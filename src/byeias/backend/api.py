from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from byeias.backend.backendcontroller import BackendController
import tempfile
import shutil

app = FastAPI()

# CORS für das Frontend erlauben
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

backend = BackendController()


class PredictRequest(BaseModel):
    context_texts: List[str]
    target_texts: List[str]


class ExplainRequest(BaseModel):
    context_before: str
    flagged_sentence: str
    context_after: str


@app.post("/predict")
def predict_bias(request: PredictRequest):
    result = backend.predict_bias(request.context_texts, request.target_texts)
    return {"predictions": result}


@app.post("/explain")
def explain_bias(request: ExplainRequest):
    result = backend.explain_bias(
        request.context_before, request.flagged_sentence, request.context_after
    )
    return result


@app.post("/extract_pdf")
def extract_pdf(file: UploadFile = File(...), language: Optional[str] = Form("german")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    sentences = backend.pdf_extractor.extract_sentences(tmp_path)
    return {"sentences": sentences}
