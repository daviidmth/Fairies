import argparse
import sys
from pathlib import Path
from typing import List, Optional

import nltk
import pypdf


def _ensure_src_on_path() -> Path:
    src_path = Path(__file__).resolve().parents[3]
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_path


_SRC_PATH = _ensure_src_on_path()

from byeias.backend.config_loader import get_backend_config, get_logger  # noqa: E402

BACKEND_CONFIG = get_backend_config()
logger = get_logger("byeias.text_extractor", BACKEND_CONFIG)


class PDFTextExtractor:
    """Extract text from PDF files and split it into sentences."""

    def __init__(self, language: str = "english"):
        self.language = language
        self._ensure_nltk_resources()
        logger.info("PDF text extractor initialized | language=%s", self.language)

    @staticmethod
    def _ensure_nltk_resources() -> None:
        resources = (
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
        )

        for resource_path, package_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info("Downloading NLTK resource: %s", package_name)
                nltk.download(package_name, quiet=True)

    def extract_sentences(self, pdf_path: str) -> List[str]:
        file_path = Path(pdf_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The file at {file_path} was not found.")

        extracted_text_parts: List[str] = []

        with file_path.open("rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            logger.info(
                "Starting PDF extraction | pages=%d path=%s",
                len(pdf_reader.pages),
                file_path,
            )

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text_parts.append(page_text)

        if not extracted_text_parts:
            logger.warning("No extractable text found in PDF: %s", file_path)
            return []

        clean_text = " ".join(" ".join(extracted_text_parts).split())
        sentences = nltk.sent_tokenize(clean_text, language=self.language)
        logger.info("Sentence extraction completed | sentences=%d", len(sentences))
        return sentences


def extract_sentences(pdf_path: str, language: str = "english") -> List[str]:
    """Compatibility wrapper for existing call sites."""
    return PDFTextExtractor(language=language).extract_sentences(pdf_path)


def _run_cli(pdf_path: str, limit: Optional[int]) -> None:
    extractor = PDFTextExtractor()
    sentences = extractor.extract_sentences(pdf_path)

    if not sentences:
        print("No sentences extracted.")
        return

    output_sentences = sentences[:limit] if limit is not None else sentences
    print(f"Extracted sentences: {len(sentences)}")
    for index, sentence in enumerate(output_sentences):
        print(f"[{index}] {sentence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sentences from a PDF file.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of printed sentences"
    )
    args = parser.parse_args()

    try:
        print(f"Running text extractor from: {_SRC_PATH}")
        _run_cli(args.pdf_path, args.limit)
    except Exception as error:
        print(f"Text extraction failed: {error}")
        raise
