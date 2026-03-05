from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .case_store import CaseStore
from .config import get_settings
from .image_pipeline import (
    LesionSimilarityService,
    encode_image,
    generate_gradcam_style_heatmap,
    generate_lesion_caption,
)
from .models import (
    AnalyzeResponse,
    DiagnosisOutput,
    DoctorReviewRequest,
    EmbedRequest,
    ExplainabilityPayload,
    RetrievedChunk,
    ScrapeRequest,
    SimilarCase,
)
from .openrouter_client import OpenRouterClient
from .rag_pipeline import MedicalRAG
from .scraper import run_scraping_pipeline


settings = get_settings()
router = APIRouter()
logger = logging.getLogger("orisight.pipeline")

case_store = CaseStore(settings.cases_dir)
rag_engine = MedicalRAG(settings)
similarity_engine = LesionSimilarityService(
    settings.chroma_lesion_path,
    settings.clip_model_name,
    allow_model_downloads=settings.allow_model_downloads,
)
openrouter_client = OpenRouterClient(settings)

SUPPORTED_DISEASES = [
    "Oral Submucous Fibrosis",
    "Leukoplakia",
    "Erythroplakia",
    "Oral Lichen Planus",
    "Oral Squamous Cell Carcinoma",
]

RISK_RULES = {
    "tobacco chewing": ["tobacco", "gutkha", "smokeless", "betel quid"],
    "areca nut consumption": ["areca", "supari", "betel nut"],
    "alcohol use": ["alcohol", "drinks daily", "ethanol"],
    "burning sensation": ["burning", "stinging"],
    "restricted mouth opening": ["restricted mouth opening", "trismus", "difficulty opening mouth"],
    "persistent white or red patches": ["white patch", "red patch", "non-scrapable", "persistent patch"],
}

DIAGNOSIS_HINTS = {
    "Oral Submucous Fibrosis": [
        "submucous",
        "osmf",
        "trismus",
        "restricted mouth opening",
        "areca",
        "fibrous bands",
        "burning sensation",
    ],
    "Leukoplakia": ["leukoplakia", "white patch", "non-scrapable", "homogeneous white plaque"],
    "Erythroplakia": ["erythroplakia", "red patch", "velvety red"],
    "Oral Lichen Planus": ["lichen", "reticular", "wickham", "bilateral", "lacy white"],
    "Oral Squamous Cell Carcinoma": [
        "ulcer",
        "indurated",
        "bleeding",
        "weight loss",
        "non-healing",
        "carcinoma",
        "malignant",
        "lymph node",
    ],
}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_case(
    request: Request,
    image: UploadFile = File(...),
    symptoms: str = Form(...),
    history: str = Form(...),
    doctor_mode: bool = Form(False),
) -> AnalyzeResponse:
    case_id = str(uuid4())
    logger.info("Case %s: analyze request received", case_id)

    image_ext = Path(image.filename or "upload.jpg").suffix or ".jpg"
    image_filename = f"{case_id}{image_ext}"
    upload_path = Path(settings.uploads_dir) / image_filename

    file_bytes = await image.read()
    if not file_bytes:
        logger.warning("Case %s: uploaded image is empty", case_id)
        raise HTTPException(status_code=400, detail="Uploaded image file is empty")

    try:
        upload_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.exception("Case %s: failed to persist uploaded image: %s", case_id, exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded image") from exc

    try:
        logger.info("Case %s: stage=image_embedding start", case_id)
        image_embedding = encode_image(
            upload_path,
            settings.clip_model_name,
            allow_downloads=settings.allow_model_downloads,
        )
        logger.info("Case %s: stage=image_embedding success dim=%s", case_id, len(image_embedding))
    except Exception as exc:
        logger.exception("Case %s: stage=image_embedding failed: %s", case_id, exc)
        raise HTTPException(status_code=422, detail="Unable to process oral cavity image") from exc

    try:
        image_caption = generate_lesion_caption(
            upload_path,
            settings.blip_model_name,
            allow_downloads=settings.allow_model_downloads,
        )
    except Exception as exc:
        logger.exception("Case %s: stage=image_caption failed: %s", case_id, exc)
        image_caption = "Oral cavity image uploaded; automated caption unavailable."

    try:
        logger.info("Case %s: stage=similarity_search start", case_id)
        similar_cases_raw = similarity_engine.query_similar(image_embedding, top_k=3)
        logger.info(
            "Case %s: stage=similarity_search success matches=%s",
            case_id,
            len(similar_cases_raw),
        )
    except Exception as exc:
        logger.exception("Case %s: stage=similarity_search failed: %s", case_id, exc)
        similar_cases_raw = []

    heatmap_filename = f"{case_id}_heatmap.png"
    heatmap_path = Path(settings.heatmaps_dir) / heatmap_filename
    try:
        generate_gradcam_style_heatmap(upload_path, heatmap_path)
    except Exception as exc:
        logger.exception("Case %s: stage=heatmap failed, using original image: %s", case_id, exc)
        heatmap_path = upload_path
        heatmap_filename = image_filename

    risk_factors = _extract_risk_factors(symptoms, history, image_caption)
    rag_query = f"Symptoms: {symptoms}\nHistory: {history}\nImage caption: {image_caption}"
    try:
        logger.info("Case %s: stage=rag_retrieval start", case_id)
        retrieved_knowledge_raw = rag_engine.retrieve(rag_query, top_k=4)
        if not retrieved_knowledge_raw:
            logger.warning("Case %s: stage=rag_retrieval empty_result", case_id)
        else:
            logger.info(
                "Case %s: stage=rag_retrieval success chunks=%s",
                case_id,
                len(retrieved_knowledge_raw),
            )
    except Exception as exc:
        logger.exception("Case %s: stage=rag_retrieval failed: %s", case_id, exc)
        retrieved_knowledge_raw = []

    llm_payload = {
        "symptoms": symptoms,
        "history": history,
        "image_caption": image_caption,
        "risk_factors": risk_factors,
        "similar_cases": similar_cases_raw,
        "retrieved_knowledge": retrieved_knowledge_raw,
    }
    llm_output = None
    try:
        logger.info("Case %s: stage=llm_request start model=%s", case_id, settings.openrouter_model)
        llm_output = openrouter_client.generate_structured_assessment(llm_payload)
        if llm_output:
            logger.info("Case %s: stage=llm_request success", case_id)
        else:
            logger.warning("Case %s: stage=llm_request unavailable, using heuristic fallback", case_id)
    except Exception as exc:
        logger.exception("Case %s: stage=llm_request failed: %s", case_id, exc)

    merged_output = (
        _normalize_output(llm_output, llm_payload) if llm_output else _heuristic_output(llm_payload)
    )

    try:
        diagnosis_output = DiagnosisOutput(**merged_output)
    except Exception as exc:
        logger.exception("Case %s: stage=output_validation failed: %s", case_id, exc)
        diagnosis_output = DiagnosisOutput(**_heuristic_output(llm_payload))

    # Persist the analyzed case into lesion similarity store for future case-based retrieval.
    try:
        similarity_engine.upsert_case(
            case_id=case_id,
            image_embedding=image_embedding,
            diagnosis=diagnosis_output.diagnosis,
            metadata={
                "source": "orisight_generated_case",
                "risk_level": diagnosis_output.risk_level,
            },
        )
    except Exception as exc:
        logger.exception("Case %s: failed to upsert similarity index: %s", case_id, exc)

    base_url = f"{request.url.scheme}://{request.url.netloc}"
    heatmap_url = f"{base_url}/static/heatmaps/{heatmap_filename}"
    image_url = f"{base_url}/static/uploads/{image_filename}"

    explainability = ExplainabilityPayload(
        image_caption=image_caption,
        risk_factors=risk_factors,
        similar_cases=[SimilarCase(**row) for row in similar_cases_raw],
        retrieved_knowledge=[RetrievedChunk(**row) for row in retrieved_knowledge_raw],
        image_url=image_url,
        heatmap_url=heatmap_url,
    )

    response = AnalyzeResponse(
        case_id=case_id,
        output=diagnosis_output,
        explainability=explainability,
        doctor_mode_enabled=doctor_mode,
    )

    try:
        case_store.save_case(
            case_id=case_id,
            symptoms=symptoms,
            history=history,
            image_path=str(upload_path),
            report=response,
        )
    except Exception as exc:
        logger.exception("Case %s: failed to persist case report: %s", case_id, exc)
        raise HTTPException(status_code=500, detail="Failed to persist analyzed case") from exc

    logger.info("Case %s: analyze request completed", case_id)
    return response


@router.post("/scrape")
def scrape_docs(payload: ScrapeRequest) -> dict[str, object]:
    try:
        return run_scraping_pipeline(settings, payload.pubmed_query, payload.pubmed_max_results)
    except Exception as exc:
        logger.exception("Scrape pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="Scraping pipeline failed") from exc


@router.post("/embed")
def embed_docs(payload: EmbedRequest) -> dict[str, object]:
    try:
        rag_stats = rag_engine.index_raw_documents(reindex=payload.reindex)
        lesion_stats = similarity_engine.ingest_seed_directory(settings.lesion_seed_dir)
    except Exception as exc:
        logger.exception("Embedding pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="Embedding pipeline failed") from exc
    return {
        "medical_docs": rag_stats,
        "lesion_images": lesion_stats,
    }


@router.get("/report/{case_id}")
def get_report(case_id: str) -> dict[str, object]:
    try:
        record = case_store.get_case(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@router.post("/report/{case_id}/review")
def review_report(case_id: str, review: DoctorReviewRequest) -> dict[str, object]:
    try:
        record = case_store.review_case(case_id, review)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.model_dump(mode="json")


def _extract_risk_factors(symptoms: str, history: str, caption: str) -> list[str]:
    text = f"{symptoms} {history} {caption}".lower()
    hits: list[str] = []

    for factor, keywords in RISK_RULES.items():
        if any(keyword in text for keyword in keywords):
            hits.append(factor)

    return hits


def _heuristic_output(payload: dict[str, object]) -> dict[str, object]:
    symptoms = str(payload.get("symptoms", ""))
    history = str(payload.get("history", ""))
    caption = str(payload.get("image_caption", ""))

    combined = f"{symptoms} {history} {caption}".lower()

    scores: dict[str, int] = {disease: 0 for disease in SUPPORTED_DISEASES}
    for disease, hints in DIAGNOSIS_HINTS.items():
        for hint in hints:
            if hint in combined:
                scores[disease] += 1

    similar_cases = payload.get("similar_cases", [])
    if isinstance(similar_cases, list):
        for case in similar_cases:
            if not isinstance(case, dict):
                continue
            diag = str(case.get("diagnosis", ""))
            sim = float(case.get("similarity", 0))
            if diag in scores:
                scores[diag] += int(round(sim * 4))

    diagnosis = max(scores, key=scores.get)
    max_score = scores[diagnosis]

    risk_factor_count = len(payload.get("risk_factors", [])) if isinstance(payload.get("risk_factors", []), list) else 0
    risk_level = "Low"
    if diagnosis == "Oral Squamous Cell Carcinoma" or max_score >= 5 or risk_factor_count >= 4:
        risk_level = "High"
    elif max_score >= 3 or risk_factor_count >= 2:
        risk_level = "Moderate"

    differential = [d for d in SUPPORTED_DISEASES if d != diagnosis][:3]

    suggested_tests = [
        "Complete oral examination with lesion mapping",
        "Incisional biopsy of suspicious area",
        "Toluidine blue / adjunctive chairside staining",
    ]
    if risk_level == "High":
        suggested_tests.extend(
            [
                "Contrast-enhanced MRI/CT for extent evaluation",
                "Cervical lymph node assessment",
            ]
        )

    treatment_plan = [
        "Eliminate risk habits (tobacco/areca/alcohol) with cessation counseling",
        "Symptomatic management and lesion-specific therapy",
        "Schedule close follow-up with photographic documentation",
    ]
    if diagnosis == "Oral Squamous Cell Carcinoma":
        treatment_plan = [
            "Urgent oncology referral for staging and histopathological confirmation",
            "Multidisciplinary treatment planning (surgery +/- chemoradiation)",
            "Nutritional and pain support",
        ]

    referral = (
        "Urgent referral to oral oncology / head-and-neck specialist"
        if risk_level == "High"
        else "Refer to oral medicine specialist for confirmatory evaluation"
    )

    confidence = min(0.95, 0.55 + (max_score * 0.06))

    return {
        "diagnosis": diagnosis,
        "differential_diagnosis": differential,
        "risk_level": risk_level,
        "suggested_tests": suggested_tests,
        "treatment_plan": treatment_plan,
        "referral": referral,
        "confidence_score": f"{confidence:.2f}",
    }


def _normalize_output(
    output: dict[str, object] | None,
    fallback_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    if not output:
        return _heuristic_output({})

    # Accept output returned as stringified JSON from some models.
    if isinstance(output, dict):
        payload = output
    elif isinstance(output, str):
        payload = _extract_json_dict(output)
    else:
        payload = {}

    fallback = _heuristic_output(fallback_payload or {})

    def get_list(key: str) -> list[str]:
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return fallback[key]  # type: ignore[index]

    return {
        "diagnosis": str(payload.get("diagnosis") or fallback["diagnosis"]),
        "differential_diagnosis": get_list("differential_diagnosis"),
        "risk_level": str(payload.get("risk_level") or fallback["risk_level"]),
        "suggested_tests": get_list("suggested_tests"),
        "treatment_plan": get_list("treatment_plan"),
        "referral": str(payload.get("referral") or fallback["referral"]),
        "confidence_score": str(payload.get("confidence_score") or fallback["confidence_score"]),
    }


def _extract_json_dict(raw: str) -> dict[str, object]:
    raw = raw.strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
