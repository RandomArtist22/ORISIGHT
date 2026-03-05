from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SimilarCase(BaseModel):
    case_id: str
    diagnosis: str
    similarity: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    source: str
    chunk: str
    score: float


class DiagnosisOutput(BaseModel):
    diagnosis: str
    differential_diagnosis: list[str]
    risk_level: str
    suggested_tests: list[str]
    treatment_plan: list[str]
    referral: str
    confidence_score: str


class ExplainabilityPayload(BaseModel):
    image_caption: str
    risk_factors: list[str]
    similar_cases: list[SimilarCase]
    retrieved_knowledge: list[RetrievedChunk]
    image_url: str
    heatmap_url: str
    disclaimer: str = "Hackathon MVP only. Not a medical diagnostic device."


class AnalyzeResponse(BaseModel):
    case_id: str
    output: DiagnosisOutput
    explainability: ExplainabilityPayload
    doctor_mode_enabled: bool = False


class CaseRecord(BaseModel):
    case_id: str
    created_at: datetime
    symptoms: str
    history: str
    image_path: str
    report: AnalyzeResponse
    doctor_review: dict[str, Any] = Field(default_factory=dict)


class ScrapeRequest(BaseModel):
    pubmed_query: str = "oral potentially malignant disorders"
    pubmed_max_results: int = 8


class EmbedRequest(BaseModel):
    reindex: bool = False


class DoctorReviewRequest(BaseModel):
    diagnosis: str | None = None
    differential_diagnosis: list[str] | None = None
    risk_level: str | None = None
    suggested_tests: list[str] | None = None
    treatment_plan: list[str] | None = None
    referral: str | None = None
    confidence_score: str | None = None
    notes: str = ""
    confirmed: bool = False
