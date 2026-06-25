# Demo Experiments — STAR Prediction Comparison

> Three experiments testing how much demographic signal exists in the data,
> and whether a human-readable model can match gradient boosting.

---

## Summary Table

| Model | Features | Accuracy | vs Baseline | Macro-F1 | AUC (OvR) |
|-------|----------|----------|-------------|----------|-----------|
| Model A (reference) | 107 | 🔴 0.4764 | -0.0661 | 0.4040 | 0.7012 |
| EXP 1 — Pure Demographics (17 features) | 17 | 🔴 0.4964 | -0.0461 | 0.4333 | 0.7375 |
| EXP 2 — Full CDM Demographics (220 features) | 220 | 🔴 0.5350 | -0.0075 | 0.4624 | 0.7632 |
| exp3_lgb | 107 | 🔴 0.4946 | -0.0479 | 0.4302 | 0.7320 |
| EXP 3 — LogReg (interpretable) | 107 | 🔴 0.3348 | -0.2077 | 0.2803 | 0.5752 |

---

## What Each Experiment Answers

### EXP 1 — Pure Demographics
> Features: Education, Region, Province, City, Barangay, xSite, Work At Home,
> Job Grade, Management Level, MSA Fusion, Site Distance, Lat/Lon (~16 features).
> **No survey flags. No org context.**

This is the floor: how much does *who you are and where you live* predict STAR?
If EXP 1 ≈ baseline, demographics alone carry no actionable signal.
If EXP 1 > baseline, geography / background has independent predictive value.

### EXP 2 — Full CDM Demographics
> All non-leakage CDM columns: Client, Program, Site, Role Type, Job Title,
> Support Type, Person Status, City, Country, tenure, and all demographic fields.

This is the org-context ceiling: how much does *where you work and what you do*
add on top of pure demographics? CDM has rich assignment and role data.
If EXP 2 >> EXP 1, role/program placement is a stronger driver than background.

### EXP 3 — Interpretability
> Same 107 features as Model A, two model types side by side.
> LightGBM shows the performance ceiling; LogisticRegression shows *what drives it*.

The coefficient table below shows which specific features the model learned
as predictors for each STAR class.

---

## EXP 3 — Logistic Regression Coefficients

> OvR (One-vs-Rest) coefficients after StandardScaler + OrdinalEncoder.
> Positive = feature pushes prediction toward this STAR class.
> Negative = feature pushes prediction away from this STAR class.


### STAR 1 (At-Risk) — top drivers

| Direction | Feature | Coefficient |
|-----------|---------|-------------|
| ↑ pushes toward | `Job Grade` | +0.0673 |
| ↑ pushes toward | `ChS-Voice - Inbound` | +0.0338 |
| ↑ pushes toward | `CL-Domo` | +0.0325 |
| ↑ pushes toward | `dist_bucket` | +0.0260 |
| ↑ pushes toward | `V-Healthcare` | +0.0188 |
| ↑ pushes toward | `S-TS` | +0.0172 |
| ↑ pushes toward | `Site Latitude` | +0.0171 |
| ↑ pushes toward | `cnt_V` | +0.0169 |
| ↑ pushes toward | `Tr-Bus` | +0.0159 |
| ↑ pushes toward | `GC Education` | +0.0154 |
| ↑ pushes toward | `S-KYC` | +0.0132 |
| ↑ pushes toward | `Tr-Jeepney` | +0.0130 |
| ↓ pushes away   | `Work At Home Sub Status` | -0.0556 |
| ↓ pushes away   | `Tr-Tricycle` | -0.0333 |
| ↓ pushes away   | `Work At Home Status` | -0.0291 |
| ↓ pushes away   | `CL-PRC` | -0.0252 |
| ↓ pushes away   | `ChS-Back Office` | -0.0222 |
| ↓ pushes away   | `cnt_ISP` | -0.0185 |
| ↓ pushes away   | `L-Quality` | -0.0176 |
| ↓ pushes away   | `CL-Public Service` | -0.0172 |
| ↓ pushes away   | `Zip Code` | -0.0150 |
| ↓ pushes away   | `ChS-Chat - Outbound` | -0.0142 |
| ↓ pushes away   | `ISP Converge` | -0.0138 |
| ↓ pushes away   | `CL-CPA` | -0.0116 |

### STAR 3 — top drivers

| Direction | Feature | Coefficient |
|-----------|---------|-------------|
| ↑ pushes toward | `Zip Code` | +0.0579 |
| ↑ pushes toward | `Job Grade` | +0.0477 |
| ↑ pushes toward | `Work At Home Sub Status` | +0.0337 |
| ↑ pushes toward | `V-Healthcare` | +0.0250 |
| ↑ pushes toward | `GC Region` | +0.0218 |
| ↑ pushes toward | `S-Retention` | +0.0215 |
| ↑ pushes toward | `ChS-Chat - Outbound` | +0.0198 |
| ↑ pushes toward | `cnt_ISP` | +0.0192 |
| ↑ pushes toward | `Tr-Tricycle` | +0.0171 |
| ↑ pushes toward | `L-Supervisor` | +0.0169 |
| ↑ pushes toward | `S-Specialized` | +0.0168 |
| ↑ pushes toward | `ISP Red Fiber` | +0.0151 |
| ↓ pushes away   | `Management Level Description` | -0.0832 |
| ↓ pushes away   | `Site Latitude` | -0.0607 |
| ↓ pushes away   | `Work At Home Status` | -0.0301 |
| ↓ pushes away   | `CL-Salesforce` | -0.0194 |
| ↓ pushes away   | `S-Escalations` | -0.0194 |
| ↓ pushes away   | `Tr-Bus` | -0.0187 |
| ↓ pushes away   | `V-Consumer` | -0.0182 |
| ↓ pushes away   | `dist_bucket` | -0.0166 |
| ↓ pushes away   | `ChS-Specialized` | -0.0164 |
| ↓ pushes away   | `V-Utilities` | -0.0151 |
| ↓ pushes away   | `CL-PRC` | -0.0135 |
| ↓ pushes away   | `Tr-Grab` | -0.0129 |

### STAR 4 — top drivers

| Direction | Feature | Coefficient |
|-----------|---------|-------------|
| ↑ pushes toward | `Zip Code` | +0.0383 |
| ↑ pushes toward | `Work At Home Sub Status` | +0.0296 |
| ↑ pushes toward | `Tr-Tricycle` | +0.0291 |
| ↑ pushes toward | `CL-SixSigma` | +0.0186 |
| ↑ pushes toward | `Work At Home Status` | +0.0177 |
| ↑ pushes toward | `CL-Public Service` | +0.0173 |
| ↑ pushes toward | `V-Technical` | +0.0172 |
| ↑ pushes toward | `ChS-Specialized` | +0.0167 |
| ↑ pushes toward | `ChS-Email - Inbound` | +0.0166 |
| ↑ pushes toward | `L-Quality` | +0.0140 |
| ↑ pushes toward | `V-Consumer` | +0.0137 |
| ↑ pushes toward | `CL-BI` | +0.0133 |
| ↓ pushes away   | `Job Grade` | -0.0472 |
| ↓ pushes away   | `CL-Domo` | -0.0255 |
| ↓ pushes away   | `ChS-Voice - Inbound` | -0.0220 |
| ↓ pushes away   | `L-HR` | -0.0193 |
| ↓ pushes away   | `Tr-Jeepney` | -0.0185 |
| ↓ pushes away   | `dist_bucket` | -0.0173 |
| ↓ pushes away   | `Tr-Fast craft` | -0.0172 |
| ↓ pushes away   | `CL-CISCO` | -0.0159 |
| ↓ pushes away   | `L-Other Operations Officer` | -0.0156 |
| ↓ pushes away   | `L-Supervisor` | -0.0130 |
| ↓ pushes away   | `L-Manager` | -0.0125 |
| ↓ pushes away   | `ISP Red Fiber` | -0.0124 |

### STAR 5 (Top) — top drivers

| Direction | Feature | Coefficient |
|-----------|---------|-------------|
| ↑ pushes toward | `Management Level Description` | +0.1010 |
| ↑ pushes toward | `Site Latitude` | +0.0510 |
| ↑ pushes toward | `CL-PRC` | +0.0427 |
| ↑ pushes toward | `Work At Home Status` | +0.0414 |
| ↑ pushes toward | `L-SME` | +0.0300 |
| ↑ pushes toward | `S-Escalations` | +0.0252 |
| ↑ pushes toward | `CL-CISCO` | +0.0250 |
| ↑ pushes toward | `Valley Fault` | +0.0214 |
| ↑ pushes toward | `CL-Salesforce` | +0.0198 |
| ↑ pushes toward | `Tr-Grab` | +0.0168 |
| ↑ pushes toward | `L-HR` | +0.0162 |
| ↑ pushes toward | `ChS-Back Office` | +0.0148 |
| ↓ pushes away   | `Zip Code` | -0.0811 |
| ↓ pushes away   | `Job Grade` | -0.0678 |
| ↓ pushes away   | `V-Healthcare` | -0.0482 |
| ↓ pushes away   | `S-Retention` | -0.0284 |
| ↓ pushes away   | `V-Insurance` | -0.0245 |
| ↓ pushes away   | `cnt_V` | -0.0228 |
| ↓ pushes away   | `CL-Social Media` | -0.0202 |
| ↓ pushes away   | `V-Technical` | -0.0190 |
| ↓ pushes away   | `CL-Domo` | -0.0181 |
| ↓ pushes away   | `ISP Other` | -0.0169 |
| ↓ pushes away   | `S-Specialized` | -0.0167 |
| ↓ pushes away   | `CL-SixSigma` | -0.0166 |

---

## Interpretation Guide

- **Accuracy ≈ baseline** → model learned nothing useful from those features
- **AUC > 0.65** → model ranks agents meaningfully even if raw accuracy is low
- **LogReg AUC close to LightGBM AUC** → relationship is mostly linear; non-linear
  interactions (captured by LightGBM) are not adding much
- **LogReg AUC << LightGBM AUC** → complex interactions matter; coefficients alone
  are a partial picture