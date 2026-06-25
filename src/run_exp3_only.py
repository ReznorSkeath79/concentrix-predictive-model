"""Resume: EXP 1 + 2 already done. Run EXP 3 then write the full report."""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_model_demo import (
    load_sdd1, load_cdm, run_exp3, write_demo_report,
)
from train_model_a import build_master_roster
from features import get_flag_cols

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# EXP 1 + 2 results from the completed run
exp1_result = {
    'label': 'EXP 1 — Pure Demographics (17 features)',
    'n_features': 17, 'accuracy': 0.4964,
    'macro_f1': 0.4333, 'auc': 0.7375, 'baseline': 0.5425,
}
exp2_result = {
    'label': 'EXP 2 — Full CDM Demographics (220 features)',
    'n_features': 220, 'accuracy': 0.5350,
    'macro_f1': 0.4624, 'auc': 0.7632, 'baseline': 0.5425,
}

log.info("Loading data for EXP 3 …")
sdd1   = load_sdd1()
cdm    = load_cdm()
roster = build_master_roster(sdd1, cdm)
flag_cols = get_flag_cols(roster)

exp3_result = run_exp3(roster, flag_cols)
write_demo_report(exp1_result, exp2_result, exp3_result)

log.info("✓ DONE  →  reports/demo_experiments.md")
