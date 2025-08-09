"""
Crop Command AI — Streamlit MVP (Streamlit)
-------------------------------------------------
Run locally:
  pip install streamlit pandas numpy pyyaml
  streamlit run app.py

Deploy to Streamlit Community Cloud:
  1) Push this file to a GitHub repo
  2) Create a new Streamlit app, point it at app.py

Notes:
- This is an MVP for soil/tissue/water analysis uploads, manual entry, and rule-based flags.
- Target ranges included here are placeholders. UPDATE THESE to your agronomy standards before production use.
- All outputs are decision-support, not a substitute for a licensed agronomist.
"""

import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

##############################
# --- App Config / Branding
##############################
APP_NAME = "Crop Command AI"
st.set_page_config(page_title=APP_NAME, page_icon="🌱", layout="wide")

# Simple brand header
st.markdown(
    f"""
    <div style='display:flex;align-items:center;gap:12px'>
      <div style='font-size:28px'>🌱 <b>{APP_NAME}</b></div>
      <div style='opacity:0.7'>MVP · v0.1</div>
    </div>
    <div style='opacity:0.8;margin-top:4px'>Upload soil / tissue / water results, pick a crop, and get quick flags & a shareable report.</div>
    <hr/>
    """,
    unsafe_allow_html=True,
)

##############################
# --- Supported Crops & Defaults
##############################
SUPPORTED_CROPS = ["Pecans", "Pistachios", "Almonds", "Potatoes", "Citrus"]
SAMPLE_TYPES = ["Soil", "Plant tissue", "Water"]
GROWTH_STAGES = [
    "Early/Establishment",
    "Vegetative",
    "Flowering",
    "Bulking / Tuber fill / Kernel fill",
    "Post-harvest / Recovery",
]

# Placeholders — replace with your standards. Values are illustrative only.
# Structure: targets[crop][sample_type][growth_stage][analyte] = (min_ppm, max_ppm)
DEFAULT_TARGETS = {
    "Generic": {
        "Soil": {
            "Vegetative": {
                "pH": (6.2, 7.8),
                "EC": (0, 2),  # dS/m (handled separately)
                "NO3-N": (10, 30),
                "P": (20, 60),
                "K": (120, 250),
                "Ca": (1000, 3000),
                "Mg": (150, 400),
                "S": (10, 50),
                "Na": (0, 100),
                "Cl": (0, 75),
                "Fe": (2, 10),
                "Mn": (2, 10),
                "Zn": (1, 5),
                "Cu": (0.5, 3),
                "B": (0.5, 2),
            }
        },
        "Plant tissue": {
            "Vegetative": {
                "N": (25000, 45000),  # ppm = % * 10000
                "P": (2000, 4500),
                "K": (15000, 35000),
                "Ca": (10000, 30000),
                "Mg": (2500, 7000),
                "S": (1500, 4000),
                "Fe": (50, 200),
                "Mn": (20, 300),
                "Zn": (15, 100),
                "Cu": (5, 30),
                "B": (20, 80),
            }
        },
        "Water": {
            "Vegetative": {
                "pH": (6.0, 7.5),
                "EC": (0, 1.2),  # dS/m (handled separately)
                "Alkalinity (as CaCO3)": (0, 120),
                "HCO3": (0, 100),
                "Na": (0, 70),
                "Cl": (0, 100),
                "SAR": (0, 6),  # handled separately
                "NO3-N": (0, 20),
                "Fe": (0, 1),
                "Mn": (0, 0.2),
                "B": (0, 0.5),
            }
        },
    }
}

##############################
# --- Utilities
##############################
CANON = {
    # normalize keys (case-insensitive)
    "n": "N",
    "no3-n": "NO3-N",
    "p": "P",
    "k": "K",
    "ca": "Ca",
    "mg": "Mg",
    "s": "S",
    "na": "Na",
    "cl": "Cl",
    "fe": "Fe",
    "mn": "Mn",
    "zn": "Zn",
    "cu": "Cu",
    "b": "B",
    "mo": "Mo",
    "ph": "pH",
    "ec": "EC",
    "sar": "SAR",
    "alkalinity": "Alkalinity (as CaCO3)",
    "hco3": "HCO3",
}

ANALYTE_ORDER = [
    "pH", "EC", "SAR", "Alkalinity (as CaCO3)", "HCO3",
    "NO3-N", "N", "P", "K", "Ca", "Mg", "S", "Na", "Cl",
    "Fe", "Mn", "Zn", "Cu", "B", "Mo"
]


@dataclass
class AnalysisResult:
    analyte: str
    value: float
    unit: str
    target_min: Optional[float]
    target_max: Optional[float]
    status: str  # Low / OK / High / Info
    note: str


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        key = c.strip().lower()
        new_cols[c] = CANON.get(key, c)
    out = df.rename(columns=new_cols)
    return out


def load_table_from_upload(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    name = upload.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(upload)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(upload)
    else:
        st.warning("Only CSV/XLSX supported in MVP. PDF parsing coming later.")
        return pd.DataFrame()
    df = normalize_columns(df)
    return df


def template_dataframe(sample_type: str) -> pd.DataFrame:
    cols = ["Analyte", "Value (PPM)"]
    base = [
        "pH", "EC", "SAR", "Alkalinity (as CaCO3)", "HCO3",
        "NO3-N", "N", "P", "K", "Ca", "Mg", "S", "Na", "Cl",
        "Fe", "Mn", "Zn", "Cu", "B", "Mo",
    ]
    if sample_type == "Plant tissue":
        # Tissue rarely has SAR/Alkalinity/HCO3
        base = [x for x in base if x not in ("SAR", "Alkalinity (as CaCO3)", "HCO3")]
    if sample_type == "Water":
        # Water rarely has total N in ppm; keep NO3-N, etc.
        pass
    return pd.DataFrame({"Analyte": base, "Value (PPM)": [np.nan]*len(base)})


def get_targets(crop: str, sample_type: str, growth_stage: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    # For now, use Generic until crop-specific tables are provided.
    safe = DEFAULT_TARGETS.get("Generic", {}).get(sample_type, {})
    # fall back: use Vegetative if stage not present
    table = safe.get(growth_stage) or safe.get("Vegetative", {})
    # Convert to dict of analyte -> (min,max)
    return {k: v for k, v in table.items()}


def analyze(df: pd.DataFrame, sample_type: str, crop: str, stage: str, unit: str) -> List[AnalysisResult]:
    df = normalize_columns(df)
    # unify: expect columns either [Analyte, Value (PPM)] or a wide table; try both
    long_df = None
    if {"Analyte", "Value (PPM)"}.issubset(set(df.columns)):
        long_df = df[["Analyte", "Value (PPM)"]].copy()
        long_df.rename(columns={"Value (PPM)": "value"}, inplace=True)
    else:
        # try to pivot a one-row wide table
        numeric_cols = [c for c in df.columns if c in ANALYTE_ORDER]
        if len(numeric_cols) == 0:
            # last resort: try all numeric columns
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(df) == 1:
            s = df.iloc[0][numeric_cols]
            long_df = pd.DataFrame({"Analyte": s.index, "value": s.values})
        else:
            st.error("Couldn't understand the table layout. Use the template or upload a single-row sheet.")
            return []

    targets = get_targets(crop, sample_type, stage)

    results: List[AnalysisResult] = []
    for _, row in long_df.iterrows():
        analyte = str(row["Analyte"]).strip()
        key = CANON.get(analyte.lower(), analyte)
        val = row.get("value")
        try:
            value = float(val)
        except Exception:
            value = np.nan

        tmin, tmax = None, None
        if key in targets:
            tmin, tmax = targets[key]

        status = "Info"
        note = ""

        if not np.isnan(value) and (tmin is not None or tmax is not None):
            if tmin is not None and value < tmin:
                status = "Low"
                note = "Below target range"
            elif tmax is not None and value > tmax:
                status = "High"
                note = "Above target range"
            else:
                status = "OK"
                note = "Within target range"
        else:
            status = "Info"
            note = "No target configured (update targets in sidebar)."

        # Special calcs
        if key == "EC":
            note += " | EC shown in dS/m; high EC can indicate salinity stress."
        if key == "SAR":
            note += " | SAR is sodium adsorption ratio; high SAR can affect soil structure."
        if key in ("pH",):
            note += " | pH affects nutrient availability."

        results.append(AnalysisResult(key, value, "PPM" if key not in ("pH", "EC", "SAR") else key, tmin, tmax, status, note))

    # Sort by category/order
    def sort_key(r: AnalysisResult):
        try:
            return ANALYTE_ORDER.index(r.analyte)
        except ValueError:
            return 999
    results.sort(key=sort_key)
    return results


def results_table(results: List[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Analyte": r.analyte,
                "Value": r.value,
                "Unit": r.unit,
                "Target Min": r.target_min,
                "Target Max": r.target_max,
                "Status": r.status,
                "Notes": r.note,
            }
        )
    return pd.DataFrame(rows)


def make_recommendations(results: List[AnalysisResult], crop: str, sample_type: str, stage: str) -> List[str]:
    recs: List[str] = []
    lows = [r for r in results if r.status == "Low"]
    highs = [r for r in results if r.status == "High"]

    if lows:
        names = ", ".join(r.analyte for r in lows)
        recs.append(f"Address low: {names}. Consider adjusting your fertility plan to lift these into range.")
    if highs:
        names = ", ".join(r.analyte for r in highs)
        recs.append(f"Watch high: {names}. Risk of toxicity or antagonism; consider reducing inputs or leaching if appropriate.")

    # Generic crop-aware nudges
    if crop in ("Pecans", "Pistachios", "Almonds") and sample_type == "Plant tissue":
        recs.append("For tree nuts, maintain balanced K:Ca:Mg; imbalances can affect kernel fill and quality.")
    if crop == "Potatoes" and sample_type == "Soil":
        recs.append("For potatoes, monitor chloride and EC closely to avoid quality penalties.")
    if crop == "Citrus" and sample_type == "Water":
        recs.append("For citrus under micro-sprinkler, check bicarbonates and alkalinity; acidification may be beneficial if high.")

    if not recs:
        recs.append("All values within configured targets. Maintain current program and monitor per schedule.")

    recs.append("Reminder: These are general suggestions; consult your agronomist for rates and product selection.")
    return recs


def download_report(results_df: pd.DataFrame, meta: Dict[str, str], recs: List[str]) -> bytes:
    # Build a simple Markdown report
    buf = io.StringIO()
    buf.write(f"# {APP_NAME} Report\n\n")
    for k, v in meta.items():
        buf.write(f"**{k}:** {v}  \n")
    buf.write("\n## Results\n\n")
    buf.write(results_df.to_markdown(index=False))
    buf.write("\n\n## Recommendations\n\n")
    for r in recs:
        buf.write(f"- {r}\n")
    buf.write("\n---\nThis report is decision-support only and not a substitute for professional advice.\n")
    return buf.getvalue().encode("utf-8")


##############################
# --- Sidebar: Controls & Targets
##############################
with st.sidebar:
    st.header("Setup")
    crop = st.selectbox("Crop", SUPPORTED_CROPS, index=0)
    sample_type = st.selectbox("Sample type", SAMPLE_TYPES, index=0)
    growth_stage = st.selectbox("Growth stage", GROWTH_STAGES, index=1)
    unit = st.selectbox("Units", ["PPM"], index=0)

    st.divider()
    st.header("Targets (edit as needed)")

    # Let users edit targets for their scenario
    targets = get_targets(crop, sample_type, growth_stage)
    # Display editable table
    tdf = pd.DataFrame([
        {"Analyte": k, "Min": v[0], "Max": v[1]} for k, v in targets.items()
    ])
    tdf = st.data_editor(tdf, num_rows="dynamic", key="targets_editor")

    # Save updated targets into session for use
    session_targets = {row["Analyte"]: (row["Min"], row["Max"]) for _, row in tdf.iterrows()}

##############################
# --- Main: Uploads / Manual Entry
##############################
st.subheader("1) Upload your lab results (CSV/XLSX) or use the template")
up = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

col1, col2 = st.columns(2)
with col1:
    st.caption("If you don't have a file, fill this out:")
    tmpl = template_dataframe(sample_type)
    editable = st.data_editor(tmpl, num_rows="dynamic", key="manual_table")
with col2:
    st.caption("Or download a blank template to share with your lab or fill offline:")
    template_buffer = io.StringIO()
    tmpl.to_csv(template_buffer, index=False)
    st.download_button("Download CSV template", data=template_buffer.getvalue(), file_name=f"{sample_type.lower()}_template.csv")


##############################
# --- Analysis
##############################
st.subheader("2) Analyze")

if up is not None:
    df = load_table_from_upload(up)
else:
    df = editable.rename(columns={"Value (PPM)": "value"}).copy()

# Use sidebar-edited targets
if "targets_editor" in st.session_state:
    edited = st.session_state["targets_editor"]
    session_targets = {row["Analyte"]: (row["Min"], row["Max"]) for _, row in edited.iterrows()}
else:
    session_targets = get_targets(crop, sample_type, growth_stage)

# Monkey-patch get_targets for this session's edited table
def get_session_targets(*args, **kwargs):
    return session_targets

results = analyze(df, sample_type, crop, growth_stage, unit)

# Replace targets in results with session-edited ones
for r in results:
    if r.analyte in session_targets:
        r.target_min, r.target_max = session_targets[r.analyte]

res_df = results_table(results)

if res_df.empty:
    st.info("Add values to the table (or upload a file) to see analysis.")
else:
    st.dataframe(res_df, use_container_width=True)

    st.subheader("3) Recommendations")
    recs = make_recommendations(results, crop, sample_type, growth_stage)
    for r in recs:
        st.write("• ", r)

    st.subheader("4) Export report")
    meta = {
        "Crop": crop,
        "Sample Type": sample_type,
        "Growth Stage": growth_stage,
        "Units": unit,
    }
    report_bytes = download_report(res_df, meta, recs)
    st.download_button(
        label="Download Markdown report",
        data=report_bytes,
        file_name=f"{crop}_{sample_type}_report.md",
        mime="text/markdown",
    )

st.divider()
st.markdown(
    """
    **Roadmap**
    - PDF parsing (common lab formats)
    - Crop-specific target libraries (pecan, pistachio, almond, potato, citrus) by growth stage
    - Multi-field projects, saved history, and team sharing
    - AI narrative explanations & what-if analysis (requires an API key)
    - Prescription rates after you provide local standards & product list
    """
)
