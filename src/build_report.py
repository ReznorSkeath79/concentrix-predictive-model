"""
build_report.py — Embeds model_viz_data.js inline so the HTML works from file://.

Chrome blocks <script src> relative paths from file:// URLs.
This script replaces the external script tag with an inline <script> block.
"""
from pathlib import Path

REPORTS = Path(__file__).parent.parent / "reports"
HTML_PATH = REPORTS / "executive_summary.html"
DATA_JS   = REPORTS / "model_viz_data.js"

html = HTML_PATH.read_text(encoding="utf-8")
data = DATA_JS.read_text(encoding="utf-8")

# Remove the external load comment + script tag
EXTERNAL_BLOCK = '\n<!-- ══ Load model data ══ -->\n<script src="./model_viz_data.js" onload="initSimulator()" onerror="document.getElementById(\'sim-load-msg\').innerHTML=\'<strong style=color:var(--red)>model_viz_data.js not found.</strong> Run <code>python src/export_for_viz.py</code> from the project root first.\'"></script>\n'

if EXTERNAL_BLOCK in html:
    html = html.replace(EXTERNAL_BLOCK, "")
    print("Removed external script tag.")
else:
    # Try softer match
    import re
    html = re.sub(r'\n<!-- ══ Load model data ══ -->.*?</script>\n', '', html, flags=re.DOTALL)
    print("Removed external script tag (regex).")

# Inject inline data right before the simulator JS block
INJECT_BEFORE = '<script>\n/* ── Simulator state ── */'
if INJECT_BEFORE not in html:
    print("ERROR: could not find simulator script anchor. Aborting.")
    exit(1)

inline = f'<script>\n{data}\n</script>\n{INJECT_BEFORE}'
html = html.replace(INJECT_BEFORE, inline, 1)
print("Injected inline data.")

# Fix the loading message if it still shows the external path hint
html = html.replace(
    "Loading model data… (requires <code>model_viz_data.js</code> in the same folder)",
    "Initialising…"
)

# Auto-call initSimulator on window load as fallback
# (in case onload on script tag was the only trigger)
BEFORE_BODY = "</body>"
if "window.addEventListener('load', initSimulator)" not in html:
    html = html.replace(
        BEFORE_BODY,
        "<script>window.addEventListener('load', function(){ if(window.MODEL_DATA) initSimulator(); });</script>\n</body>"
    )
    print("Added window load fallback.")

HTML_PATH.write_text(html, encoding="utf-8")
size_mb = HTML_PATH.stat().st_size / 1e6
print(f"Done -> {HTML_PATH}  ({size_mb:.1f} MB)")
