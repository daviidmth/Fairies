from byeias.backend.classification.model_bias import BiasDetectionPipeline
from byeias.backend.extraction.text_extracter import PDFTextExtractor
from byeias.backend.llm_explanation.llm_communicator import LLMCommunicator


class BackendController:
    """
    Zentrale Steuerung für das Backend: Klassifikation, PDF-Extraktion, LLM-Kommunikation.
    """

    def __init__(self, model_name=None, device=None, llm_model=None, llm_api_key=None):
        # Klassifikations-Pipeline
        self.classifier = BiasDetectionPipeline(model_name=model_name, device=device)
        # PDF-Extraktion
        self.pdf_extractor = PDFTextExtractor(language="german")
        # LLM-Kommunikation
        self.llm = LLMCommunicator(model_name=llm_model, api_key=llm_api_key)

    # --- Klassifikation ---
    def train_classifier(self, **kwargs):
        return self.classifier.train(**kwargs)

    def predict_bias(self, context_texts, target_texts):
        return self.classifier.predict(
            context_texts=context_texts, target_texts=target_texts
        )

    # --- PDF-Extraktion ---
    def extract_pdf_sentences(self, pdf_path):
        return self.pdf_extractor.extract_sentences(pdf_path)

    # --- LLM-Kommunikation ---
    def explain_bias(self, context_before, flagged_sentence, context_after):
        return self.llm.explain_bias(context_before, flagged_sentence, context_after)
