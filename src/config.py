"""
config.py — Central configuration: paths, column groups, leakage guards, target map.
All other modules import from here.
"""
from pathlib import Path

# ── Root directories ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT          # CSVs live in the project root
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ── Raw file paths ────────────────────────────────────────────────────────────
SDD1_PATH      = DATA_DIR / "Sample Demo Data 1.csv"
CDM_PATH       = DATA_DIR / "CDM Raw-smaller.csv"
KPI_PATH       = DATA_DIR / "CNX _ PH _ Advisor KPIs.csv"
SCORECARD_PATH = DATA_DIR / "current_data.csv"

# ── Target mapping: STAR {1,3,4,5} → ordinal {0,1,2,3} ──────────────────────
STAR_MAP    = {1: 0, 3: 1, 4: 2, 5: 3}
INV_STAR_MAP = {0: 1, 1: 3, 2: 4, 3: 5}
STAR_LABELS = ["STAR 1 (At-Risk)", "STAR 3", "STAR 4", "STAR 5 (Top)"]
N_CLASSES   = 4

# ── Employee number column candidates (searched case-insensitively) ───────────
EMP_NUM_COLS = [
    "game changer employee number",
    "employee number",
    "advisor id",
    "emp no",
    "employeenumber",
]

# ── Target / leakage column names ─────────────────────────────────────────────
STAR_TARGET_COL = "Latest STAR Rating"

# These columns encode the STAR outcome — never feed to any model as features
LEAKAGE_COLS = [
    "Latest STAR Rating",
    "Average PTG Last 3 Months",
    "Average STAR Last 3 Months",
    "Latest Performance Month",
    "Latest Performance Year",
    # lowercase duplicates so case-insensitive checks are easy
    "latest star rating",
    "average ptg last 3 months",
    "average star last 3 months",
    "latest performance month",
    "latest performance year",
]

# ── KPI target (new dependent variable — replaces STAR) ───────────────────────
KPI_TARGET_COL  = "Latest Weighted PTG"
KPI_TIER_LABELS = ["At-Risk", "Developing", "High Performer"]
N_KPI_CLASSES   = 3

# Probability head indices for the 3-class KPI model
KPI_AT_RISK_CLASS    = 0   # tercile 1 — PTG below p33
KPI_DEVELOPING_CLASS = 1   # tercile 2 — PTG p33..p67
KPI_HIGH_PERF_CLASS  = 2   # tercile 3 — PTG above p67

# Columns that are IDs / names / admin — never features
ALWAYS_DROP = [
    "CONSENT",
    "Record ID",
    "_BATCH_ID_",
    "Game Changer Full Name",
    "Advisor Name",
    "Date",
    "date",
    "Person ID",
    "person id",
    # internal join keys added by our pipeline
    "_emp_num",
    "_source",
    "_y",
    "_star",
    "_src_order",
    "email_lower",
    "_email_lower",
    "_emp_str",
    # Transportation noise (raw flags + engineered aggregate)
    "Tr-Jeepney", "Tr-Tricycle", "Tr-Train", "Tr-Van", "Tr-Bus",
    "Tr-Grab", "Tr-Fast craft", "cnt_Tr", "dist_bucket",
    # Geographic noise — low signal, high cardinality
    "GC Barangay", "Zip Code", "Valley Fault", "Site Longitude", "Site Latitude",
    # Individual cert flags — keep cnt_CL aggregate instead
    "CL-Public Service", "CL-Health Care", "CL-PRC", "CL-CPA", "CL-Teaching",
    "CL-Legal", "CL-ForeignLanguage", "CL-SixSigma", "CL-Project", "CL-BI",
    "CL-Social Media", "CL-COPC", "CL-ISO", "CL-Salesforce", "CL-Domo",
    "CL-Microsoft", "CL-CISCO", "CL-ORACLE", "CL-LINUX", "CL-IBM",
    "CL-Google", "CL-Adobe", "CL-Java",
    # Post-hire leaky features
    "Activity", "Campus", "MSA Fusion",
]

# ── Binary flag column prefixes (cols 29-112 in SDD1) ────────────────────────
FLAG_PREFIXES = ["L-", "S-", "ChS-", "CL-", "Tr-", "ISP", "V-"]

# ── Pre-hire categorical columns (Model A — knowable from a resume) ───────────
PREHIRE_CAT_COLS = [
    "GC Region",
    "GC Province",
    "GC City",
    "GC Barangay",
    "GC Education",
    "xSite",
    "Work At Home Status",
    "Work At Home Sub Status",
    "Management Level Description",
    "Job Grade",
    "MSA Fusion",
]

# Pre-hire numeric columns
PREHIRE_NUM_COLS = [
    "Site Distance",
    "Valley Fault",
    "Zip Code",
    "Site Longitude",
    "Site Latitude",
]

# ── Onboarding columns (Model A+ only — NOT available at resume stage) ────────
ONBOARDING_COLS = [
    "Role Type",
    "Person Status",
    "Job Title",
    "Support Type",
    "Cost Center Billable Flag",
    "Client",
    "Program",
    "Activity",
    "Site",
    "Campus",
    "City",
    "State",
    "Country",
    # tenure engineered from Hire Date
    "tenure_months",
    "tenure_bucket",
]

# ── CNX KPI file schema ───────────────────────────────────────────────────────
KPI_ADVISOR_ID   = "Advisor ID"
KPI_MONTH        = "Month year"
KPI_METRIC       = "Metric"
KPI_SCORE        = "Score"
KPI_PTG          = "PTG"
KPI_WEIGHTED_PTG = "Weighted PTG"
KPI_WEIGHTS      = "Weights"
KPI_METRICS      = ["Resolved PTG", "QA", "CSAT", "Attendance"]

# ── current_data (scorecard) schema ──────────────────────────────────────────
SCORECARD_EMAIL        = "Email Address"
SCORECARD_PERSON_ID    = "Person ID"
SCORECARD_METRIC       = "Metric Name"
SCORECARD_DATE         = "Date"
SCORECARD_PTG          = "PTG"
SCORECARD_KPI_PASSED   = "KPI Passed"
SCORECARD_FAILING      = "Failing Metrics"
SCORECARD_CONSEC_FAIL  = "Consecutive Failing Metrics"
SCORECARD_MET          = "Met Count"
SCORECARD_FAIL         = "Fail Count"
SCORECARD_AVG_PTG      = "Average PTG"
SCORECARD_RANK         = "Rank"
SCORECARD_QUARTILE     = "Quartile"

# ── Training hyper-parameters ─────────────────────────────────────────────────
RANDOM_STATE    = 42
CV_FOLDS        = 5
TEST_SIZE       = 0.20
OPTUNA_TRIALS   = 50   # reduce to 20 for quick dev runs

# ── Chunk sizes for large file streaming ─────────────────────────────────────
KPI_CHUNK_SIZE       = 100_000
SCORECARD_CHUNK_SIZE = 50_000

# ── Probability head class indices ────────────────────────────────────────────
AT_RISK_CLASS    = 0        # STAR 1 → ordinal 0
TOP_PERF_CLASSES = [2, 3]   # STAR 4, 5 → ordinal 2, 3

# ── Honest baseline (majority STAR 1 on CDM labeled) ─────────────────────────
BASELINE_ACC = 0.58
