"""
predictor.py — Model loading and KPI tier inference.
No Streamlit dependency — pure Python so it's testable independently.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from features import coerce_dtypes


def load_artifact(
    model_path: str | Path = "models/model_kpi_a.joblib",
    schema_path: str | Path = "models/schema_kpi.json",
) -> tuple[dict, dict]:
    """
    Load model artifact and schema from disk.

    Returns
    -------
    artifact : dict  — model, encoder, feature_cols, kpi_tier_labels, tier_info, …
    schema   : dict  — kpi_output, features dict
    """
    artifact = joblib.load(model_path)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    # Compute low-cardinality categorical cols from schema so predict_one can force
    # category dtype even when values are NaN (which loses the object dtype signal).
    high_card_set = set(artifact.get("high_card_cols", []))
    artifact["_low_card_cats"] = [
        name for name, info in schema["features"].items()
        if info["type"] == "categorical"
        and name not in high_card_set
        and name in set(artifact["feature_cols"])
    ]

    return artifact, schema


def predict_one(artifact: dict, input_dict: dict) -> dict:
    """
    Score a single candidate.

    Parameters
    ----------
    artifact   : dict   — loaded via load_artifact()
    input_dict : dict   — {feature_name: value} — missing keys become NaN

    Returns
    -------
    dict with keys:
        tier_idx : int              — 0=At-Risk, 1=Developing, 2=High Performer
        tier     : str              — human label
        proba    : list[float]      — [P(At-Risk), P(Developing), P(High Performer)]
        labels   : list[str]        — ["At-Risk", "Developing", "High Performer"]
    """
    feature_cols   = artifact["feature_cols"]
    high_card_cols = artifact["high_card_cols"]
    encoder        = artifact["encoder"]
    model          = artifact["model"]
    tier_labels    = artifact["kpi_tier_labels"]

    X = pd.DataFrame([input_dict])
    X = X.reindex(columns=feature_cols, fill_value=np.nan)
    X_enc = coerce_dtypes(encoder.transform(X.copy()), high_card_cols)
    for col in X_enc.select_dtypes(include=["object"]).columns:
        X_enc[col] = X_enc[col].astype("category")
    for col in artifact.get("_low_card_cats", []):
        if col in X_enc.columns and X_enc[col].dtype != "category":
            X_enc[col] = X_enc[col].astype("category")

    tier_idx = int(model.predict(X_enc)[0])
    proba    = model.predict_proba(X_enc)[0].tolist()

    return {
        "tier_idx": tier_idx,
        "tier":     tier_labels[tier_idx],
        "proba":    proba,
        "labels":   tier_labels,
    }


def predict_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of candidates.
    Appends columns: predicted_tier, p_at_risk, p_developing, p_high_performer.
    """
    results = [predict_one(artifact, row.to_dict()) for _, row in df.iterrows()]
    out = df.copy()
    out["predicted_tier"] = [r["tier"] for r in results]
    out["p_at_risk"]        = [r["proba"][0] for r in results]
    out["p_developing"]     = [r["proba"][1] for r in results]
    out["p_high_performer"] = [r["proba"][2] for r in results]
    return out
