"""MLflow helpers (tracking + model registry)."""
from __future__ import annotations

import logging
import os
import warnings

import mlflow
from mlflow.tracking import MlflowClient

from .config import load_config, mlflow_tracking_uri, resolve

log = logging.getLogger(__name__)


def init_mlflow() -> str:
    cfg = load_config()
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    warnings.filterwarnings("ignore", module="mlflow")
    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg))
    name = cfg.mlflow.experiment
    if mlflow.get_experiment_by_name(name) is None:
        artifact_root = None
        if mlflow.get_tracking_uri().startswith("sqlite"):
            artifact_root = resolve(cfg.mlflow.artifact_dir).as_uri()
        mlflow.create_experiment(name, artifact_location=artifact_root)
    mlflow.set_experiment(name)
    return name


def log_sklearn_model(model, X_example) -> str:
    """Log a model to MLflow (native xgboost flavour, else sklearn flavour); returns the model URI."""
    from mlflow.models import infer_signature
    from xgboost import XGBClassifier

    signature = infer_signature(X_example, model.predict_proba(X_example)[:, 1])
    common = {"name": "model", "signature": signature, "input_example": X_example.head(3)}

    if isinstance(model, XGBClassifier):
        return mlflow.xgboost.log_model(model, **common).model_uri

    # MLflow 3 serialises sklearn models with skops, which only loads explicitly trusted types.
    # We trained this model in-process, so its own types are trusted; anything skops can't handle
    # (e.g. our custom AnomalyDetector) falls back to cloudpickle.
    try:
        import skops.io as sio

        trusted = sio.get_untrusted_types(data=sio.dumps(model))
        return mlflow.sklearn.log_model(model, skops_trusted_types=trusted, **common).model_uri
    except Exception as exc:  # noqa: BLE001
        log.info("skops serialisation unavailable (%s); using cloudpickle", type(exc).__name__)
        return mlflow.sklearn.log_model(model, serialization_format="cloudpickle", **common).model_uri


def register_champion(model_uri: str, tags: dict | None = None) -> str:
    """Register the model version and point the `champion` alias at it. Returns the version number."""
    cfg = load_config()
    client = MlflowClient()
    mv = mlflow.register_model(model_uri, cfg.mlflow.registered_model)
    for k, v in (tags or {}).items():
        client.set_model_version_tag(cfg.mlflow.registered_model, mv.version, k, str(v))
    client.set_registered_model_alias(cfg.mlflow.registered_model, "champion", mv.version)
    log.info("Registered %s v%s as @champion", cfg.mlflow.registered_model, mv.version)
    return str(mv.version)
