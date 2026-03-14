import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class LoggingConfig:
    path: Path
    level: int
    max_bytes: int
    backup_count: int
    format: str
    date_format: str


@dataclass(frozen=True)
class ClassificationConfig:
    model_name: str
    tokenizer_max_length: int
    dropout_rate: float
    sexism_num_labels: int
    racism_num_labels: int
    loss_ignore_index: int
    default_batch_size: int
    train_shuffle: bool
    eval_shuffle: bool
    default_epochs: int
    default_learning_rate: float
    best_model_path: str
    default_device: str
    fillna_context: str
    required_columns: List[str]


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    max_tokens: int
    temperature: float
    api_key: str
    system_prompt_path: Path


@dataclass(frozen=True)
class BackendConfig:
    config_path: Path
    logging: LoggingConfig
    classification: ClassificationConfig
    llm: LLMConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    return _project_root() / "configs" / "config.yaml"


def _resolve_from_project_root(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _parse_log_level(value: str) -> int:
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level in config: {value}")
    return level


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}. Create configs/config.yaml or set BYEIAS_CONFIG_PATH."
        )

    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Invalid YAML structure in {path}. Expected a mapping at root level."
        )

    return loaded


@lru_cache(maxsize=1)
def get_backend_config() -> BackendConfig:
    configured_path = os.getenv("BYEIAS_CONFIG_PATH")
    config_path = Path(configured_path) if configured_path else _default_config_path()
    if not config_path.is_absolute():
        config_path = (_project_root() / config_path).resolve()
    yaml_config = _load_yaml(config_path)

    backend_cfg = yaml_config["backend"]
    logging_cfg = backend_cfg["logging"]
    classification_cfg = backend_cfg["classification"]
    dataloader_cfg = classification_cfg["dataloader"]
    training_cfg = classification_cfg["training"]
    dataset_cfg = classification_cfg["dataset"]
    llm_cfg = backend_cfg["llm"]

    return BackendConfig(
        config_path=config_path,
        logging=LoggingConfig(
            path=_resolve_from_project_root(str(logging_cfg["path"])),
            level=_parse_log_level(logging_cfg["level"]),
            max_bytes=int(logging_cfg["max_bytes"]),
            backup_count=int(logging_cfg["backup_count"]),
            format=str(logging_cfg["format"]),
            date_format=str(logging_cfg["date_format"]),
        ),
        classification=ClassificationConfig(
            model_name=str(classification_cfg["model_name"]),
            tokenizer_max_length=int(classification_cfg["tokenizer_max_length"]),
            dropout_rate=float(classification_cfg["dropout_rate"]),
            sexism_num_labels=int(classification_cfg["sexism_num_labels"]),
            racism_num_labels=int(classification_cfg["racism_num_labels"]),
            loss_ignore_index=int(classification_cfg["loss_ignore_index"]),
            default_batch_size=int(dataloader_cfg["default_batch_size"]),
            train_shuffle=bool(dataloader_cfg["train_shuffle"]),
            eval_shuffle=bool(dataloader_cfg["eval_shuffle"]),
            default_epochs=int(training_cfg["default_epochs"]),
            default_learning_rate=float(training_cfg["default_learning_rate"]),
            best_model_path=str(
                _resolve_from_project_root(str(training_cfg["best_model_path"]))
            ),
            default_device=str(classification_cfg["default_device"]),
            fillna_context=str(dataset_cfg["fillna_context"]),
            required_columns=list(dataset_cfg["required_columns"]),
        ),
        llm=LLMConfig(
            model_name=str(llm_cfg["model_name"]),
            max_tokens=int(llm_cfg["max_tokens"]),
            temperature=float(llm_cfg["temperature"]),
            api_key=str(llm_cfg["api_key"]),
            system_prompt_path=_resolve_from_project_root(
                str(llm_cfg["system_prompt"])
            ),
        ),
    )


def get_logger(name: str, config: Optional[BackendConfig] = None) -> logging.Logger:
    config = config or get_backend_config()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(config.logging.level)
    logger.propagate = False

    config.logging.path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=config.logging.format,
        datefmt=config.logging.date_format,
    )

    file_handler = RotatingFileHandler(
        filename=config.logging.path,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(config.logging.level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger
