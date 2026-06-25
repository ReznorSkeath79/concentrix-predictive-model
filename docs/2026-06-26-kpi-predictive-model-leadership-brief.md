# KPI Performance Predictive Model — Leadership Brief

**Project:** Concentrix PH Advisor Performance Intelligence
**Prepared:** June 26, 2026
**Status:** Phase 2 Complete — Scoring App Live
**Audience:** Operations Leadership, Talent Acquisition, Workforce Planning, Site Directors

---

## Executive Summary

We built a machine learning model that predicts an advisor's **KPI performance tier before they are hired** — using only pre-hire information available during recruitment (background, certifications, org placement context, and demographics).

The model classifies candidates into three tiers:

| Tier | Meaning | Population Share |
|---|---|---|
| 🔴 **At-Risk** | PTG below 91.7% — likely to struggle | ~33% |
| 🟡 **Developing** | PTG between 91.7% and 103% — middle performers | ~33% |
| 🟢 **High Performer** | PTG above 103% — top performers | ~33% |

**Bottom line:** The model is correct 58.3% of the time — compared to a 33.3% baseline if you guessed randomly. That is a **+25 percentage point lift** in predictive accuracy. On a population of 312 candidates, that means roughly 78 more correct placement decisions than chance alone.

A scoring app is live and ready to use. You can score a single candidate in under 60 seconds or upload a batch CSV of hundreds.

---

## 1. The Problem We Solved

### Why We Could Not Use STAR Ratings

The initial approach was to predict an advisor's STAR performance rating. This was abandoned after discovering a structural flaw: **STAR is not a measure of performance — it is a measure of rank.**

Concentrix applies a forced bell-curve distribution to STAR ratings:
- Top ~10% of advisors → STAR 5
- Next tier → STAR 4/3
- Bottom ~50% → STAR 1 or 2 **by policy, regardless of actual output**

This means that on any team or program, approximately half the advisors will always receive the lowest rating — even if the entire team improved 20% year-over-year. A model trained to predict STAR was learning *who is ranked lowest on their team*, not *who performs poorly in absolute terms*. That is a fundamentally different and far less useful prediction.

### The Switch to Weighted PTG

`Latest Weighted PTG` (Percentage to Goal) is the **composite KPI score** — a weighted aggregate of the actual performance metrics that matter (quality, efficiency, adherence, and program-specific KPIs). It is measured in absolute terms, not relative rank.

This is what we want to predict. A PTG of 95% means the same thing regardless of which team or program the advisor is on.

---

## 2. Data Used

### Source

The model was trained exclusively on the **CDM dataset** (Central Data Mart). CDM was the only source containing `Latest Weighted PTG`.

| Dataset | Records | Has PTG | Used For |
|---|---|---|---|
| CDM | 65,039 agents | ✅ Yes | Model training |
| SDD1 | ~additional records | ❌ No | Not used (no PTG) |

**Labeled population:** 41,717 agents (64.1% of CDM had a PTG value). The remaining 35.9% were excluded — they either had no recorded KPI or had not been active long enough to generate a stable PTG score.

### Target Variable Construction

Rather than predicting the raw PTG number (a regression problem), we converted PTG into **tercile-based tier labels**:

1. Compute the 33rd and 67th percentiles of the actual PTG distribution across all labeled agents
2. Assign tier based on where each agent's PTG falls:
   - Below p33 (91.66%) → **At-Risk** (class 0)
   - Between p33 and p67 (91.66% – 102.98%) → **Developing** (class 1)
   - Above p67 (102.98%) → **High Performer** (class 2)

This produces **naturally balanced classes** (~13,906 agents per tier) — critical for fair model training. There is no forced distribution; these cutpoints reflect the actual shape of the performance curve.

### Features (122 Pre-Hire Variables)

All features represent information that is available **before hire** — nothing from the advisor's active employment record is used as input.

| Feature Category | Examples | Count |
|---|---|---|
| **Certifications & Licences** | CL-PRC, CL-CISCO, CL-Salesforce, CL-AWS | ~20 |
| **Support Type Experience** | S-Voice, S-Chat, S-Email, S-Non-Voice, S-Tech Support | ~8 |
| **Channel Experience** | ChS-Inbound, ChS-Outbound, ChS-Blended | ~6 |
| **Leadership Background** | L-Team Lead, L-Supervisor, L-Manager | ~5 |
| **Industry Vertical Experience** | V-Financial, V-Telco, V-Healthcare, V-Retail | ~10 |
| **Transportation** | Tr-Own Vehicle, Tr-Public, Tr-Motorcycle | ~4 |
| **ISP Connectivity** | ISP flags, connection type | ~5 |
| **Demographics & Org Context** | GC Region, Province, Client, Program, Site, Management Level, Job Grade, Work-At-Home Status, Site Distance | ~64 |

**Total: ~122 features.** The majority are binary flags (0 = No, 1 = Yes). Categorical fields (Client, Program, Site, etc.) are handled with fold-safe encoding to prevent data leakage.

---

## 3. How the Model Was Built

### Algorithm: LightGBM Gradient Boosting

We used **LightGBM** — a gradient boosting framework that builds an ensemble of decision trees. It is widely used in industry for tabular data classification because:
- It handles missing values natively (advisors with unknown fields are scored without imputation)
- It handles a mix of numeric, binary, and categorical features
- It produces probability estimates (not just a class label)
- It is fast to train and score, even at 40,000+ records

### Preventing Data Leakage

Several fields that appear in the raw data were explicitly **excluded** from the model because they would only be known after hire or would create circular logic:

- `Latest Weighted PTG` itself (the target — must not be a feature)
- STAR rating (derived from PTG rankings — would be circular)
- Tenure, performance history, and any post-hire metrics

This ensures the model only uses information that exists at the point of a hiring or placement decision.

### High-Cardinality Encoding

Fields like `Client`, `Program`, and `Site` can have dozens of unique values. Standard one-hot encoding would create sparse, unreliable columns. We used **FoldTargetEncoder** — a cross-validation-safe target encoding technique that:
- Encodes each category as a weighted average of the target (KPI tier) for that category
- Uses out-of-fold statistics to prevent the encoder from "seeing" the current row's label
- Eliminates leakage that would artificially inflate accuracy

### Hyperparameter Optimization

We used **Optuna** (a Bayesian optimization framework) to search for the best model configuration across 50 trials. Each trial was evaluated using 5-fold cross-validation on the training set. This ensures the chosen parameters generalize well and are not overfit to a single data split.

### Training / Test Split

- **Training set:** 80% of labeled agents — used for fitting the model and encoder
- **Test set:** 20% of labeled agents (held out, never used in training) — used for final evaluation
- All accuracy metrics reported below are from the **held-out test set only**

---

## 4. Model Performance Analysis

### Summary Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **AUC (Macro)** | 0.762 | Strong discriminative ability across all 3 tiers |
| **Accuracy** | 58.3% | Correctly classifies 58.3% of candidates |
| **Majority-class Baseline** | 33.3% | What random guessing achieves (equal class sizes) |
| **Lift vs. Baseline** | **+25.0 pp** | The model adds 25 percentage points over chance |
| **Macro F1-Score** | 0.582 | Balanced performance across all 3 tiers |

### What AUC 0.762 Means in Practice

AUC (Area Under the ROC Curve) measures how well the model separates tiers from one another. A score of 0.5 means no discrimination (random). A score of 1.0 means perfect separation. **0.762 is considered a strong result** for a pre-hire prediction problem, where the signal is inherently noisy.

The model is especially useful for identifying **At-Risk candidates** — the tier that carries the highest operational cost when misclassified (training investment, attrition, performance coaching, customer impact).

### What the Model Cannot Do

- It **does not predict STAR ratings** — by design
- It **cannot account for manager quality, team dynamics, or training quality** — post-hire variables not in scope
- It is **probabilistic, not deterministic** — a candidate predicted as "Developing" has a probability distribution across all three tiers
- It **does not replace human judgment** — it is a prioritization signal, not a hiring decision engine

### Probability Outputs

For every candidate scored, the model returns three probability values that sum to 100%:

```
P(At-Risk) = 18.4%  |  P(Developing) = 54.1%  |  P(High Performer) = 27.5%
                                → Predicted: Developing
```

A candidate with P(At-Risk) = 70% is a stronger signal than one with P(At-Risk) = 35%, even though both might be labeled "At-Risk." This allows operations leaders to **rank and triage** candidates by risk level, not just classify them.

### Known Predictors (From Model Analysis)

Based on feature importance analysis from earlier model iterations (signal is expected to be similar for the KPI model):

| Predictor | Direction | Notes |
|---|---|---|
| Management Level at hire | Higher → better | Most predictive single feature |
| CL-PRC (Licensed Professional) | Positive | Strong signal for structured roles |
| CL-CISCO / CL-Salesforce | Positive | Technical certification — program fit |
| Work-At-Home Status | WFH positive | Remote-capable advisors trend higher |
| Program / Client placement | Mixed | Strongest org-level signal; varies by program |
| Job Grade at hire | Lower → At-Risk | Entry grades show higher At-Risk rates |
| Site Distance | Higher → slight negative | Commute strain may affect adherence |

---

## 5. How to Use the Model

### Scoring App (Live)

**Access:** Run `streamlit run app/scoring_app.py` from the project directory, or ask the team to deploy it on a shared server.

**Two modes:**

#### Mode 1 — Single Candidate (Form)

1. Open the app → Tab: **"Score Candidate"**
2. Fill in the candidate's pre-hire details across the expandable sections
3. Click **"Predict KPI Tier"**
4. Review the tier badge and probability breakdown

Best for: Individual screening, interview prep, high-stakes placement decisions.

#### Mode 2 — Batch Scoring (CSV)

1. Open the app → Tab: **"Batch Upload"**
2. Click **"Download CSV Template"** — opens a blank spreadsheet with all 122 column headers
3. Fill in one row per candidate
4. Upload the completed CSV → Click **"Score All Candidates"**
5. Results appear sorted by P(At-Risk) descending — highest-risk candidates at top
6. Click **"Download Scored Results"** to export

Best for: End-of-week recruitment batches, program-level planning, roster reviews.

### Recommended Use Cases

| Use Case | How | Expected Value |
|---|---|---|
| **Pre-hire screening** | Batch score all applicants in final-stage pipeline | Prioritize At-Risk candidates for deeper structured interview |
| **Program placement** | Score candidate against multiple program contexts (change Client/Site field) | Match candidate to program where probability of High Performer is highest |
| **At-Risk early warning** | Score newly onboarded advisors against their pre-hire profile | Flag Day 1 At-Risk cohort for closer onboarding support |
| **Recruitment planning** | Batch score applicant pool before scheduling | Optimize offer throughput by prioritizing High Performer predictions |
| **Coaching prioritization** | Identify predicted At-Risk agents still in probation | Allocate TL coaching hours before the first STAR review cycle |

### What to Do With At-Risk Predictions

An "At-Risk" prediction is **not a rejection signal** — it is a **support signal**. Recommended actions:

- Assign a stronger onboarding buddy or senior TL
- Schedule check-ins at Day 30, Day 60, Day 90
- Place in a program where At-Risk profiles have historically shown better outcomes
- Ensure WFH setup is fully functional if on a remote-eligible account
- Track against actuals at their first performance cycle — use to validate and refine

---

## 6. Model Improvement Recommendations

These are ordered by estimated impact-to-effort ratio.

### High Priority

#### 1. SHAP Feature Importance Analysis on KPI Model
**What:** Run SHAP (SHapley Additive exPlanations) analysis specifically on the KPI model — not the deprecated STAR models.
**Why:** The current key predictor list was derived from STAR-era models. The KPI model may have different feature importance rankings. Understanding *which* pre-hire signals are truly driving KPI tier predictions enables better recruitment briefing and sourcing strategy.
**Effort:** Low (1–2 days). SHAP is already integrated in the codebase.

#### 2. Capture the Unlabeled 35.9%
**What:** Investigate why 23,322 CDM agents have no PTG value. If they are tenured agents with stable performance, label and include them.
**Why:** More labeled data = more robust model, especially for edge-case demographics and programs that are currently underrepresented in the 41,717-agent training set.
**Effort:** Medium (requires coordination with the data team to resolve missing PTG records).

#### 3. Probability Calibration
**What:** Apply isotonic regression or Platt scaling to calibrate the model's probability outputs.
**Why:** "P(At-Risk) = 70%" should mean the advisor is At-Risk 70% of the time. Without calibration, LightGBM probabilities tend to be overconfident or underconfident — the raw number is directionally correct but not literally accurate. Calibrated probabilities are more useful for downstream triage thresholds.
**Effort:** Low (1 day).

### Medium Priority

#### 4. Program-Specific Sub-Models
**What:** Train separate models per high-volume program (e.g., Tech Support, Collections, Customer Service).
**Why:** A "High Performer" in Collections looks very different from a "High Performer" in Tech Support. The current unified model treats all programs identically, which dilutes program-specific signals. Sub-models would sharpen predictions for each LOB.
**Effort:** Medium (2–3 weeks). Requires sufficient labeled data per program (minimum ~500 agents per class per program).

#### 5. Add SDD1 Data When PTG Becomes Available
**What:** If the data team can backfill `Latest Weighted PTG` into SDD1, incorporate it into the training set.
**Why:** SDD1 contains additional records not in CDM. More diverse training data improves generalization, especially for newer programs and sites.
**Effort:** Low once the data is available.

#### 6. Retraining Cadence
**What:** Retrain the model every 6 months using the latest CDM extract.
**Why:** Hiring profiles, program types, client mixes, and performance standards evolve. A model trained on 2024–2025 data will drift as the business changes. A scheduled retraining pipeline ensures predictions stay fresh.
**Effort:** Low (1–2 days to operationalize the existing training script as a scheduled job).

### Lower Priority (Future Enhancements)

#### 7. Education Level and Assessment Scores
**What:** Include educational background and pre-employment assessment scores (if available from ATS) as additional features.
**Why:** These are strong pre-hire signals that the current model does not have access to. Even a coarse education-level field (high school / college / graduate) may add meaningful predictive power.

#### 8. Time-to-Ramp Feature
**What:** Add a prediction for "time to full productivity" as a secondary output alongside tier.
**Why:** A candidate predicted as "High Performer" who takes 6 months to ramp is a different planning case than one who is performing at full speed by Day 45. Time-to-ramp is operationally as important as eventual tier.

#### 9. API Wrapper for ATS Integration
**What:** Wrap the model as a REST API and integrate with the Applicant Tracking System (ATS).
**Why:** Removes the CSV workflow entirely. Recruiters see a predicted tier and probability directly on the candidate profile in the ATS, at the moment of review.
**Effort:** Medium (2–3 weeks for a FastAPI wrapper + ATS integration).

#### 10. Feedback Loop Tracking
**What:** Build a tracking table that records every scored candidate alongside their actual PTG outcome 90–180 days post-hire.
**Why:** This creates a ground-truth feedback loop that (a) validates model accuracy in production, (b) surfaces systematic errors by site/program/recruiter, and (c) provides the labeled data for future retraining. Without this, model drift goes undetected.

---

## 7. Technical Reference

### Repository Structure

```
PredictiveModel/
├── src/
│   ├── config.py           # Constants, KPI tier labels, cutpoints
│   ├── kpi_targets.py      # Build tercile tiers from PTG
│   ├── evaluate.py         # Evaluation metrics, plots
│   ├── features.py         # Feature engineering, coerce_dtypes
│   └── train_kpi_model.py  # Full training pipeline
├── app/
│   ├── predictor.py        # Model loading + inference
│   ├── schema_utils.py     # Schema parsing + validation
│   └── scoring_app.py      # Streamlit UI
├── models/
│   ├── model_kpi_a.joblib  # Trained model artifact
│   └── schema_kpi.json     # Input schema for scoring
├── reports/
│   └── kpi_metrics.md      # Training run metrics
└── docs/
    └── scoring-app-preview.html  # Static UI preview
```

### Running the Scoring App

```bash
cd "E:\work\concentrix\PredictiveModel"
streamlit run app/scoring_app.py
# → Opens at http://localhost:8501
```

### Retraining the Model

```bash
cd "E:\work\concentrix\PredictiveModel"
python -m src.train_kpi_model
# Reads CDM → trains → saves models/model_kpi_a.joblib + models/schema_kpi.json
```

### Key Thresholds

| Threshold | Value | Source |
|---|---|---|
| At-Risk / Developing cutpoint (p33) | PTG = 91.66% | Computed from CDM tercile |
| Developing / High Performer cutpoint (p67) | PTG = 102.98% | Computed from CDM tercile |
| Training set size | 33,374 agents | 80% of 41,717 labeled |
| Test set size | 8,343 agents | 20% held out |
| Model accuracy on test set | 58.3% | Held-out evaluation |
| Majority-class baseline | 33.3% | Equal tercile distribution |

---

*Prepared by the Concentrix PH Performance Intelligence team. For questions on model methodology, contact the analytics team. For access to the scoring app, contact the data engineering team.*
