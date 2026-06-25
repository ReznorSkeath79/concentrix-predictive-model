"""
schema_utils.py — Schema parsing, field spec generation, and input validation.
All logic here is pure Python; no Streamlit dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Field groups (for UI section headers) ────────────────────────────────────
# Prefix → display group name. Unmatched fields go to "Demographics & Org Context".
_PREFIX_GROUPS = {
    "L-":   "Leadership Background",
    "S-":   "Support Type Experience",
    "ChS-": "Channel Experience",
    "CL-":  "Certifications & Licences",
    "Tr-":  "Transportation",
    "ISP":  "Internet Service Provider",
    "V-":   "Industry Vertical Experience",
}


def _field_group(name: str) -> str:
    for prefix, label in _PREFIX_GROUPS.items():
        if name.startswith(prefix):
            return label
    return "Demographics & Org Context"


def get_field_specs(schema: dict) -> list[dict]:
    """
    Build ordered field spec list from schema['features'].

    Each spec dict:
        name     : str
        type     : 'binary' | 'categorical' | 'numeric'
        group    : str           — UI section header
        default  : Any           — safe default value
        options  : list | None   — for categorical only
        min      : float | None  — for numeric only
        max      : float | None  — for numeric only
    """
    specs = []
    for name, info in schema["features"].items():
        spec: dict = {"name": name, "type": info["type"], "group": _field_group(name)}
        if info["type"] == "binary":
            spec["default"] = 0.0
            spec["options"] = None
            spec["min"]     = None
            spec["max"]     = None
        elif info["type"] == "numeric":
            spec["default"] = float(info["range"][0])
            spec["options"] = None
            spec["min"]     = float(info["range"][0])
            spec["max"]     = float(info["range"][1])
        else:
            opts = info.get("values", [])
            spec["default"] = opts[0] if opts else ""
            spec["options"] = opts
            spec["min"]     = None
            spec["max"]     = None
        specs.append(spec)
    return specs


def validate_and_coerce(user_input: dict, schema: dict) -> tuple[dict, list[str]]:
    """
    Validate and coerce raw user input (from form or CSV row) against the schema.

    Returns
    -------
    coerced : dict        — {feature_name: coerced_value} for all schema features
    errors  : list[str]   — human-readable validation messages (empty = valid)
    """
    coerced: dict = {}
    errors:  list = []

    for name, info in schema["features"].items():
        val = user_input.get(name)

        try:
            is_missing = pd.isna(val)
        except (TypeError, ValueError):
            is_missing = False
        if val is None or is_missing:
            coerced[name] = np.nan
            continue

        if info["type"] == "binary":
            coerced[name] = float(bool(val))

        elif info["type"] == "numeric":
            try:
                coerced[name] = float(val)
            except (ValueError, TypeError):
                coerced[name] = np.nan
                errors.append(f"'{name}': expected a number, got {val!r}")

        else:  # categorical
            coerced[name] = str(val) if val != "" else np.nan

    return coerced, errors


def get_csv_template(schema: dict) -> pd.DataFrame:
    """
    Return a single-row DataFrame with all feature columns set to NaN.
    Used to generate a downloadable CSV template for batch scoring.
    """
    cols = list(schema["features"].keys())
    return pd.DataFrame([{c: np.nan for c in cols}])
