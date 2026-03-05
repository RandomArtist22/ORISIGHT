from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import AnalyzeResponse, CaseRecord, DoctorReviewRequest


class CaseStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        return self.base_dir / f"{case_id}.json"

    def save_case(
        self,
        case_id: str,
        symptoms: str,
        history: str,
        image_path: str,
        report: AnalyzeResponse,
    ) -> CaseRecord:
        record = CaseRecord(
            case_id=case_id,
            created_at=datetime.utcnow(),
            symptoms=symptoms,
            history=history,
            image_path=image_path,
            report=report,
            doctor_review={},
        )
        self._write(record)
        return record

    def get_case(self, case_id: str) -> CaseRecord:
        path = self._path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"Case {case_id} not found")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return CaseRecord.model_validate(payload)

    def review_case(self, case_id: str, review: DoctorReviewRequest) -> CaseRecord:
        record = self.get_case(case_id)
        report_output = record.report.output.model_copy()

        for field in [
            "diagnosis",
            "differential_diagnosis",
            "risk_level",
            "suggested_tests",
            "treatment_plan",
            "referral",
            "confidence_score",
        ]:
            value = getattr(review, field)
            if value is not None:
                setattr(report_output, field, value)

        doctor_review: dict[str, Any] = {
            "reviewed_at": datetime.utcnow().isoformat(),
            "notes": review.notes,
            "confirmed": review.confirmed,
        }

        record.report.output = report_output
        record.doctor_review = doctor_review
        self._write(record)
        return record

    def _write(self, record: CaseRecord) -> None:
        with self._path(record.case_id).open("w", encoding="utf-8") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2)
