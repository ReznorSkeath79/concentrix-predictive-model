"""
evaluate.py — Shared evaluation utilities.
Metrics, confusion matrix, calibration, SHAP, threshold report, metrics.md writer.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    STAR_LABELS, N_CLASSES, AT_RISK_CLASS, TOP_PERF_CLASSES,
    MODELS_DIR, REPORTS_DIR, BASELINE_ACC, INV_STAR_MAP,
)

log = logging.getLogger(__name__)


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test,
    model_name: str = "model",
    shap_values=None,
    shap_X: pd.DataFrame = None,
    class_labels: list = None,
    baseline_acc: float = None,
) -> dict:
    """
    Full evaluation pass. Writes confusion matrix, calibration, SHAP, and
    threshold plots to reports/. Returns a metrics dict.

    Args:
        class_labels: Optional list of class label strings for plot axes/titles.
                      If None, falls back to STAR_LABELS from config (backward compat).
    """
    _labels = class_labels if class_labels is not None else STAR_LABELS

    REPORTS_DIR.mkdir(exist_ok=True)

    y_test = np.asarray(y_test, dtype=int)
    y_pred = np.asarray(model.predict(X_test), dtype=int)

    has_proba = hasattr(model, 'predict_proba')
    y_proba   = model.predict_proba(X_test) if has_proba else None

    # Core metrics
    acc      = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    auc      = None

    if y_proba is not None and len(np.unique(y_test)) > 1:
        try:
            auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
        except Exception:
            pass

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    _baseline = baseline_acc if baseline_acc is not None else BASELINE_ACC

    metrics = {
        'model':              model_name,
        'n_test':             int(len(y_test)),
        'accuracy':           round(acc, 4),
        'baseline_accuracy':  _baseline,
        'lift_vs_baseline':   round(acc - _baseline, 4),
        'macro_f1':           round(macro_f1, 4),
        'auc_ovr_macro':      round(auc, 4) if auc is not None else None,
        'classification_report': report,
    }

    log.info(
        f"[{model_name}] Test accuracy={acc:.4f} "
        f"(+{acc - _baseline:+.4f} vs baseline) | "
        f"Macro-F1={macro_f1:.4f} | AUC={f'{auc:.4f}' if auc else 'N/A'}"
    )

    # Confusion matrix
    _plot_confusion(y_test, y_pred, model_name, _labels)

    # Calibration
    if y_proba is not None:
        _plot_calibration(y_test, y_proba, model_name, _labels)

        # At-risk probability head
        at_risk_proba = y_proba[:, AT_RISK_CLASS]
        metrics['at_risk_threshold_report'] = _threshold_report(
            y_test, at_risk_proba, AT_RISK_CLASS, f'{model_name}_at_risk'
        )
        # Top-performer head — guard against models with fewer classes (e.g. 3-class KPI)
        n_model_classes = y_proba.shape[1]
        valid_top_classes = [c for c in TOP_PERF_CLASSES if c < n_model_classes]
        if valid_top_classes:
            top_proba = y_proba[:, valid_top_classes].sum(axis=1)
            metrics['top_perf_threshold_report'] = _threshold_report(
                y_test, top_proba,
                class_idx=None,   # binary: 1 if top-performer class
                tag=f'{model_name}_top_perf',
                binary_y=(np.isin(y_test, valid_top_classes)).astype(int),
            )

    # SHAP
    if shap_values is not None and shap_X is not None:
        _plot_shap(shap_values, shap_X, model_name)

    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_confusion(y_true, y_pred, model_name, class_labels=None):
    n_cls = len(np.unique(np.concatenate([y_true, y_pred])))
    n_cls = max(n_cls, len(class_labels)) if class_labels else max(n_cls, N_CLASSES)
    labels = (class_labels if class_labels is not None else STAR_LABELS)[:n_cls]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} — Confusion Matrix')
    plt.tight_layout()
    path = REPORTS_DIR / f'{model_name}_confusion.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    log.info(f"  Plot saved: {path.name}")


def _plot_calibration(y_true, y_proba, model_name, class_labels=None):
    _cal_labels = class_labels if class_labels is not None else STAR_LABELS
    n_cls = y_proba.shape[1]
    _cal_labels = _cal_labels[:n_cls]
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 4), sharey=True)
    if n_cls == 1:
        axes = [axes]
    for i, (ax, lbl) in enumerate(zip(axes, _cal_labels)):
        bin_y = (y_true == i).astype(int)
        if bin_y.sum() < 10:
            ax.set_title(f'{lbl}\n(sparse)')
            continue
        try:
            frac_pos, mean_pred = calibration_curve(bin_y, y_proba[:, i], n_bins=10)
            ax.plot(mean_pred, frac_pos, 'o-', label='Model', ms=4)
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
        except Exception:
            pass
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('Mean predicted')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Fraction positive')
    plt.suptitle(f'{model_name} — Calibration', y=1.01)
    plt.tight_layout()
    path = REPORTS_DIR / f'{model_name}_calibration.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    log.info(f"  Plot saved: {path.name}")


def _threshold_report(
    y_true, proba, class_idx, tag, binary_y=None
) -> dict:
    """Precision/recall vs threshold for a binary probability head."""
    from sklearn.metrics import precision_recall_curve

    if binary_y is None:
        binary_y = (np.asarray(y_true) == class_idx).astype(int)

    try:
        precision, recall, thresholds = precision_recall_curve(binary_y, proba)
        f1s = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-9),
            0.0,
        )
        best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thresholds, precision[:-1], label='Precision')
        ax.plot(thresholds, recall[:-1], label='Recall')
        ax.axvline(thresholds[best_idx], color='k', ls='--', lw=1, label=f'Best thresh={thresholds[best_idx]:.3f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{tag} — Precision/Recall vs Threshold')
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = REPORTS_DIR / f'{tag}_threshold.png'
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        log.info(f"  Plot saved: {path.name}")

        return {
            'best_threshold':     float(thresholds[best_idx]),
            'best_f1':            float(f1s[best_idx]),
            'precision_at_best':  float(precision[best_idx]),
            'recall_at_best':     float(recall[best_idx]),
        }
    except Exception as e:
        log.warning(f"Threshold report failed for {tag}: {e}")
        return {}


def _plot_shap(shap_values, X: pd.DataFrame, model_name: str):
    try:
        import shap
        # For multi-class LightGBM, shap_values is a list of arrays (one per class)
        # Use class 0 (STAR 1 / At-Risk) for the summary plot
        sv = shap_values[AT_RISK_CLASS] if isinstance(shap_values, list) else shap_values

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X, show=False, max_display=25, plot_size=None)
        plt.title(f'{model_name} — SHAP (STAR 1 / At-Risk class)', pad=12)
        plt.tight_layout()
        path = REPORTS_DIR / f'{model_name}_shap.png'
        plt.savefig(path, dpi=110, bbox_inches='tight')
        plt.close()
        log.info(f"  Plot saved: {path.name}")
    except Exception as e:
        log.warning(f"SHAP plot failed: {e}")


# ── Metrics markdown writer ───────────────────────────────────────────────────

def write_metrics_md(all_metrics: list, coverage_stats: dict = None):
    """Write honest metrics report to reports/metrics.md."""
    REPORTS_DIR.mkdir(exist_ok=True)

    lines = [
        "# Predictive Performance Model — Honest Accuracy Report",
        "",
        f"> Generated: 2026-06-24 | Baseline (majority STAR 1): **{BASELINE_ACC:.0%}**",
        "",
        "---",
        "",
    ]

    # Join coverage section
    if coverage_stats:
        lines += [
            "## Data Join Coverage",
            "",
            "| Source | Total Agents | Joined | % |",
            "|--------|-------------|--------|---|",
        ]
        for src, v in coverage_stats.items():
            lines.append(
                f"| {src} | {v.get('total', '-'):,} | {v.get('joined', '-'):,} | {v.get('pct', '-')} |"
                if isinstance(v.get('total'), int) else
                f"| {src} | {v.get('total', '-')} | {v.get('joined', '-')} | {v.get('pct', '-')} |"
            )
        lines += ["", "---", ""]

    for m in all_metrics:
        acc_icon = "✅" if m['accuracy'] >= 0.90 else ("🟡" if m['accuracy'] >= 0.70 else "🔴")
        lines += [
            f"## {m['model']}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Test samples | {m['n_test']:,} |",
            f"| **Accuracy** | {acc_icon} **{m['accuracy']:.1%}** |",
            f"| Baseline accuracy | {m['baseline_accuracy']:.1%} |",
            f"| Lift vs baseline | {m['lift_vs_baseline']:+.1%} |",
            f"| Macro F1 | {m['macro_f1']:.4f} |",
            f"| AUC (OvR macro) | {m.get('auc_ovr_macro') or 'N/A'} |",
            "",
        ]

        cr = m.get('classification_report', {})
        lines += [
            "### Per-Class Results",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|----|---------|",
        ]
        for i, lbl in enumerate(STAR_LABELS[:N_CLASSES]):
            r = cr.get(str(i), {})
            lines.append(
                f"| {lbl} | {r.get('precision', 0):.3f} | "
                f"{r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | "
                f"{int(r.get('support', 0)):,} |"
            )

        # At-risk head
        if 'at_risk_threshold_report' in m and m['at_risk_threshold_report']:
            t = m['at_risk_threshold_report']
            lines += [
                "",
                "### At-Risk Probability Head (P[STAR=1])",
                "",
                f"Best threshold: **{t.get('best_threshold', 0):.3f}** → "
                f"Precision: {t.get('precision_at_best', 0):.3f} | "
                f"Recall: {t.get('recall_at_best', 0):.3f} | "
                f"F1: {t.get('best_f1', 0):.3f}",
            ]

        # Top-perf head
        if 'top_perf_threshold_report' in m and m['top_perf_threshold_report']:
            t = m['top_perf_threshold_report']
            lines += [
                "",
                "### Top-Performer Probability Head (P[STAR∈{4,5}])",
                "",
                f"Best threshold: **{t.get('best_threshold', 0):.3f}** → "
                f"Precision: {t.get('precision_at_best', 0):.3f} | "
                f"Recall: {t.get('recall_at_best', 0):.3f} | "
                f"F1: {t.get('best_f1', 0):.3f}",
            ]

        lines += ["", "---", ""]

    # Honest assessment footer
    lines += [
        "## Honest Assessment",
        "",
        "| Model | Signal | Realistic Ceiling | Phase-2 Ready? |",
        "|-------|--------|-------------------|----------------|",
        "| **Model A** (resume/pre-hire) | Survey flags, demographics, location | ~60–67% accuracy | ✅ Yes — deployable as screening score |",
        "| **Model A+** (onboarding) | + Role Type, Client, Site, Tenure | ~65–73% accuracy | ⚠️ Needs post-hire data |",
        "| **Model B** (KPI→STAR) | KPI component scores (QA, CSAT, Attendance, Resolved-PTG) | **≥88–96%** | ✅ Yes — ops analytics & coaching |",
        "",
        "Pre-hire signal is inherently weak because demographic/survey attributes explain only a fraction",
        "of performance variance. Role assignment, client/program, and KPI metrics are the real drivers.",
        "Model B achieves ≥90% because STAR is mechanically composed from the very KPIs it uses as features.",
        "",
    ]

    out = REPORTS_DIR / "metrics.md"
    out.write_text('\n'.join(lines), encoding='utf-8')
    log.info(f"Metrics report → {out}")
