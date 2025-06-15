from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

_V_FIELDS = {f"V{i}": (float, Field(..., description=f"PCA component {i}")) for i in range(1, 29)}
_TransactionFeatures = create_model("TransactionFeatures", **_V_FIELDS)  # type: ignore[call-overload]


class Transaction(_TransactionFeatures):  # type: ignore[misc,valid-type]
    transaction_id: str | None = Field(None, description="Client id; generated if omitted")
    Time: float = Field(..., ge=0, description="Seconds since the reference start of the stream")
    Amount: float = Field(..., ge=0, description="Transaction amount (EUR/USD)")


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=5000)
    explain: Literal["all", "flagged", "none"] = "flagged"


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    typical_range: list[float] | None = None
    reason: str


class Explanation(BaseModel):
    summary: str
    method: str
    top_factors: list[Factor]


class Prediction(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model_version: str
    explanation: Explanation | None = None


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n_flagged: int
    latency_ms: float


class Label(BaseModel):
    transaction_id: str
    is_fraud: bool


class FeedbackRequest(BaseModel):
    labels: list[Label] = Field(..., min_length=1)
