import os
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from byeias.backend.config_loader import get_backend_config, get_logger

BACKEND_CONFIG = get_backend_config()
logger = get_logger("byeias.bias_model", BACKEND_CONFIG)
CLASSIFICATION_CONFIG = BACKEND_CONFIG.classification


class MultiTaskDeberta(nn.Module):
    """Multi-task architecture based on DeBERTa-v3."""

    def __init__(
        self,
        model_name: str,
        dropout_rate: float,
        sexism_num_labels: int,
        racism_num_labels: int,
    ):
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.sexism_head = nn.Linear(hidden_size, sexism_num_labels)
        self.racism_head = nn.Linear(hidden_size, racism_num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Extract [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        sexism_logits = self.sexism_head(pooled_output)
        racism_logits = self.racism_head(pooled_output)

        return sexism_logits, racism_logits


class BiasDetectionPipeline:
    """End-to-end pipeline for training, evaluation, and inference with context support."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or CLASSIFICATION_CONFIG.model_name
        self.max_length = CLASSIFICATION_CONFIG.tokenizer_max_length
        self.dropout_rate = CLASSIFICATION_CONFIG.dropout_rate

        configured_device = device or CLASSIFICATION_CONFIG.default_device
        resolved_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu" if configured_device == "auto" else configured_device
        )
        self.device = torch.device(resolved_device)

        logger.info(
            "Pipeline initialized | device=%s model=%s max_length=%d",
            self.device,
            self.model_name,
            self.max_length,
        )

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.model = MultiTaskDeberta(
            model_name=self.model_name,
            dropout_rate=self.dropout_rate,
            sexism_num_labels=CLASSIFICATION_CONFIG.sexism_num_labels,
            racism_num_labels=CLASSIFICATION_CONFIG.racism_num_labels,
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=CLASSIFICATION_CONFIG.loss_ignore_index
        )

    def _tokenize_fn(self, batch: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        # Use both context (previous sentence) and target text for classification.
        return self.tokenizer(
            batch["context"],
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def prepare_dataloader(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        effective_batch_size = batch_size or CLASSIFICATION_CONFIG.default_batch_size
        effective_shuffle = (
            CLASSIFICATION_CONFIG.train_shuffle if shuffle is None else shuffle
        )

        # Fill empty context fields with configured fallback text.
        df["context"] = df["context"].fillna(CLASSIFICATION_CONFIG.fillna_context)

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            self._tokenize_fn, batched=True, remove_columns=["context", "text"]
        )

        required_cols = CLASSIFICATION_CONFIG.required_columns
        available_cols = [col for col in required_cols if col in dataset.column_names]

        dataset.set_format(type="torch", columns=available_cols)
        return DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=effective_shuffle
        )

    def train(
        self,
        train_df_sexism: pd.DataFrame,
        train_df_racism: pd.DataFrame,
        val_df_sexism: pd.DataFrame,
        val_df_racism: pd.DataFrame,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        epochs = epochs or CLASSIFICATION_CONFIG.default_epochs
        lr = lr or CLASSIFICATION_CONFIG.default_learning_rate
        batch_size = batch_size or CLASSIFICATION_CONFIG.default_batch_size

        logger.info("Preparing and merging training and validation datasets...")

        # Datensätze vorbereiten und jeweils die andere Kategorie ignorieren (-1)
        train_s = train_df_sexism.copy()
        train_s["racism_label"] = CLASSIFICATION_CONFIG.loss_ignore_index

        train_r = train_df_racism.copy()
        train_r["sexism_label"] = CLASSIFICATION_CONFIG.loss_ignore_index

        train_df = pd.concat([train_s, train_r], ignore_index=True)

        val_s = val_df_sexism.copy()
        val_s["racism_label"] = CLASSIFICATION_CONFIG.loss_ignore_index

        val_r = val_df_racism.copy()
        val_r["sexism_label"] = CLASSIFICATION_CONFIG.loss_ignore_index

        val_df = pd.concat([val_s, val_r], ignore_index=True)

        train_loader = self.prepare_dataloader(
            train_df,
            batch_size=batch_size,
            shuffle=CLASSIFICATION_CONFIG.train_shuffle,
        )
        val_loader = self.prepare_dataloader(
            val_df,
            batch_size=batch_size,
            shuffle=CLASSIFICATION_CONFIG.eval_shuffle,
        )

        logger.info(
            "Training started | total_train_samples=%d total_val_samples=%d epochs=%d batch_size=%d lr=%s",
            len(train_df),
            len(val_df),
            epochs,
            batch_size,
            lr,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sexism_labels = batch["sexism_label"].to(self.device)
                racism_labels = batch["racism_label"].to(self.device)

                s_logits, r_logits = self.model(input_ids, attention_mask)

                # Berechnung vor NaN-Fehlern schützen, falls der Batch zufällig nur Labels EINER Sorte enthält
                ignore_idx = CLASSIFICATION_CONFIG.loss_ignore_index

                if (sexism_labels != ignore_idx).any():
                    loss_s = self.loss_fn(s_logits, sexism_labels)
                else:
                    loss_s = torch.tensor(0.0, device=self.device)

                if (racism_labels != ignore_idx).any():
                    loss_r = self.loss_fn(r_logits, racism_labels)
                else:
                    loss_r = torch.tensor(0.0, device=self.device)

                loss = loss_s + loss_r

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info("Epoch %d - Avg Train Loss: %.4f", epoch + 1, avg_train_loss)

            val_loss, metrics = self.evaluate(val_loader)
            logger.info("Epoch %d - Val Loss: %.4f", epoch + 1, val_loss)
            logger.info(
                "Validation metrics | sexism_f1=%.4f racism_f1=%.4f",
                metrics["sexism_f1"],
                metrics["racism_f1"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(CLASSIFICATION_CONFIG.best_model_path)
                logger.info("New best model saved | val_loss=%.4f", best_val_loss)

        torch.cuda.empty_cache()
        logger.info("Training completed")

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_s_preds, all_s_labels, all_r_preds, all_r_labels = [], [], [], []
        ignore_idx = CLASSIFICATION_CONFIG.loss_ignore_index

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sexism_labels = batch["sexism_label"].to(self.device)
                racism_labels = batch["racism_label"].to(self.device)

                s_logits, r_logits = self.model(input_ids, attention_mask)

                # Loss Berechnung ebenfalls schützen
                loss_s = (
                    self.loss_fn(s_logits, sexism_labels)
                    if (sexism_labels != ignore_idx).any()
                    else torch.tensor(0.0, device=self.device)
                )
                loss_r = (
                    self.loss_fn(r_logits, racism_labels)
                    if (racism_labels != ignore_idx).any()
                    else torch.tensor(0.0, device=self.device)
                )
                total_loss += (loss_s + loss_r).item()

                s_preds = torch.argmax(s_logits, dim=1).cpu().numpy()
                r_preds = torch.argmax(r_logits, dim=1).cpu().numpy()

                s_mask = sexism_labels.cpu().numpy() != ignore_idx
                r_mask = racism_labels.cpu().numpy() != ignore_idx

                all_s_preds.extend(s_preds[s_mask])
                all_s_labels.extend(sexism_labels.cpu().numpy()[s_mask])
                all_r_preds.extend(r_preds[r_mask])
                all_r_labels.extend(racism_labels.cpu().numpy()[r_mask])

        avg_loss = total_loss / len(val_loader)
        metrics = {
            "sexism_f1": f1_score(
                all_s_labels, all_s_preds, average="macro", zero_division=0
            ),
            "racism_f1": f1_score(
                all_r_labels, all_r_preds, average="macro", zero_division=0
            ),
        }
        return avg_loss, metrics

    def predict(self, context_texts: List[str], target_texts: List[str]) -> List[Dict]:
        """Inference method for new texts with context."""
        if len(context_texts) != len(target_texts):
            raise ValueError(
                "context_texts and target_texts must have the same length."
            )

        logger.info("Inference started | samples=%d", len(target_texts))
        self.model.eval()

        inputs = self.tokenizer(
            context_texts,
            target_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            s_logits, r_logits = self.model(**inputs)

        s_preds = torch.argmax(s_logits, dim=1).cpu().tolist()
        r_preds = torch.argmax(r_logits, dim=1).cpu().tolist()

        results = []
        for i in range(len(target_texts)):
            results.append(
                {
                    "context": context_texts[i],
                    "text": target_texts[i],
                    "sexism_prediction": s_preds[i],
                    "racism_prediction": r_preds[i],
                }
            )
        logger.info("Inference completed | samples=%d", len(results))
        return results

    def save_model(self, path: Optional[str] = None):
        path = path or CLASSIFICATION_CONFIG.best_model_path
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(self.model.state_dict(), path)
        logger.info("Model weights saved to: %s", path)

    def load_model(self, path: Optional[str] = None):
        path = path or CLASSIFICATION_CONFIG.best_model_path
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("Model weights loaded from: %s", path)
