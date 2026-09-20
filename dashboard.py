"""Interactive project dashboard for the HER3 knockout HEK293F 12-week cell culture run.

Run with:  streamlit run dashboard.py
"""

import re
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_PATH = Path(__file__).parent / "data" / "HER3_KO_HEK293F_12Week_Data.xlsx"
SHEET_NAME = "12-Week Daily Bench Data"
VIABILITY_MIN_PCT = 80.0  # QC minimum; below this the CONTINGENCY_MAP fallback applies

# ROI assumptions (illustrative planning inputs, not measured from the bench data).
DEFAULT_SCIENTISTS = 25
HOURS_SAVED_PER_SCIENTIST_WEEK = 12  # conservative: logging, ELN entry, presentation generation
LOADED_COST_PER_HOUR = 45            # USD, average fully loaded cost
HOURS_PER_FTE_WEEK = 40

REQUIRED_COLUMNS = [
    "Date", "Week", "Day", "Project_ID", "Client_Name", "Cell_Line", "Target_Gene",
    "Passage_No", "Cell_Density_M_mL", "Viability_Pct", "HER3_Knockout_Pct",
    "Instrument_QC_Status", "SOP_Protocol", "ELN_Record_ID", "Scientist_Notes",
]

# Chart colours: blue / aqua are categorical slots 1 and 3 (validated in light and dark);
# red is the reserved "critical" status colour, kept off the series so a breach never
# reads as a series.
COLORS = {
    "light": {"viability": "#2a78d6", "density": "#1baf7a", "critical": "#d03b3b", "surface": "#fcfcfb"},
    "dark": {"viability": "#3987e5", "density": "#199e70", "critical": "#e66767", "surface": "#1a1a19"},
}

# Row highlight tints are translucent so they read on both light and dark themes.
SEVERITY_TINT = {
    2: "background-color: rgba(208, 59, 59, 0.25)",   # contingency trigger / viability breach
    1: "background-color: rgba(250, 178, 25, 0.28)",  # QC warning, calibration, follow-up
}

PASSING_QC = {"passed", "pass", "ok"}
CALIBRATION_RE = re.compile(r"calibrat|drift", re.I)
WARNING_RE = re.compile(r"\b(warning|alert|deviation|out[- ]of[- ]spec\w*)\b", re.I)
CONTINGENCY_FOLLOW_UP_RE = re.compile(r"post[- ]contingency", re.I)
CONTINGENCY_RE = re.compile(r"contingency", re.I)


@st.cache_data(show_spinner=False)
def load_data(path: str, modified: float) -> pd.DataFrame:
    """Read the bench-data sheet; `modified` (file mtime) busts the cache when the file changes."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{SHEET_NAME}' is missing columns: {', '.join(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Week_No"] = pd.to_numeric(df["Week"].astype(str).str.extract(r"(\d+)", expand=False))
    df = df.dropna(subset=["Week_No"]).astype({"Week_No": int})
    return df.sort_values("Date").reset_index(drop=True)


def classify_row(row: pd.Series) -> tuple[str, int]:
    """Return (flag reasons, severity) for a bench-data row. Severity: 0 none, 1 warning, 2 critical."""
    status = "" if pd.isna(row["Instrument_QC_Status"]) else str(row["Instrument_QC_Status"]).strip()
    notes = "" if pd.isna(row["Scientist_Notes"]) else str(row["Scientist_Notes"])
    reasons: list[str] = []
    severity = 0

    if status and status.lower() not in PASSING_QC:
        reasons.append(f"QC status: {status}")
        severity = max(severity, 1)
    if CALIBRATION_RE.search(notes):
        reasons.append("Calibration issue")
        severity = max(severity, 1)
    if WARNING_RE.search(notes):
        reasons.append("Warning noted")
        severity = max(severity, 1)
    if CONTINGENCY_FOLLOW_UP_RE.search(notes):
        reasons.append("Contingency follow-up")
        severity = max(severity, 1)
    elif CONTINGENCY_RE.search(notes):
        reasons.append("Contingency trigger")
        severity = max(severity, 2)
    if row["Viability_Pct"] < VIABILITY_MIN_PCT:
        reasons.append(f"Viability < {VIABILITY_MIN_PCT:.0f}%")
        severity = max(severity, 2)

    return "; ".join(reasons), severity


def is_dark_theme() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def build_chart(df: pd.DataFrame) -> go.Figure:
    colors = COLORS["dark" if is_dark_theme() else "light"]
    breaches = df[df["Viability_Pct"] < VIABILITY_MIN_PCT]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Viability_Pct"], name="Viability (%)", mode="lines",
            line=dict(color=colors["viability"], width=2),
            hovertemplate="%{y:.1f}%<extra>Viability</extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Cell_Density_M_mL"], name="Cell density (M/mL)", mode="lines",
            line=dict(color=colors["density"], width=2),
            hovertemplate="%{y:.2f} M cells/mL<extra>Density</extra>",
        ),
        secondary_y=True,
    )

    # 80% threshold line, always drawn so the QC floor is visible even with no breach.
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1, yref="y", y0=VIABILITY_MIN_PCT, y1=VIABILITY_MIN_PCT,
        line=dict(color=colors["critical"], width=1.5, dash="dash"),
    )
    fig.add_annotation(
        xref="paper", x=1, xanchor="right", yref="y", y=VIABILITY_MIN_PCT, yanchor="bottom",
        text=f"{VIABILITY_MIN_PCT:.0f}% viability minimum", showarrow=False,
        font=dict(color=colors["critical"], size=12),
    )

    if not breaches.empty:
        notes = breaches["Scientist_Notes"].fillna("").map(lambda n: textwrap.fill(n, 60).replace("\n", "<br>"))
        fig.add_trace(
            go.Scatter(
                x=breaches["Date"], y=breaches["Viability_Pct"], name=f"Viability < {VIABILITY_MIN_PCT:.0f}%",
                mode="markers+text", text=breaches["Viability_Pct"].map("{:.1f}%".format),
                textposition="bottom center",
                marker=dict(color=colors["critical"], size=13, line=dict(color=colors["surface"], width=2)),
                customdata=list(zip(breaches["Day"], notes)),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra>Below minimum</extra>",
            ),
            secondary_y=False,
        )

    viab_floor = min(df["Viability_Pct"].min(), VIABILITY_MIN_PCT)
    viab_lo = max(0, (int(viab_floor) // 5) * 5 - 5)
    dens_lo = max(0.0, int((df["Cell_Density_M_mL"].min() - 0.2) * 10) / 10)
    dens_hi = int((df["Cell_Density_M_mL"].max() + 0.2) * 10 + 1) / 10

    fig.update_yaxes(
        title_text="Viability (%)", range=[viab_lo, 101], showline=True, linewidth=2,
        linecolor=colors["viability"], ticks="outside", tickcolor=colors["viability"],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Cell density (M cells/mL)", range=[dens_lo, dens_hi], showgrid=False,
        showline=True, linewidth=2, linecolor=colors["density"], ticks="outside",
        tickcolor=colors["density"], secondary_y=True,
    )
    fig.update_xaxes(title_text=None, showgrid=False)
    fig.update_layout(
        height=470, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def flagged_table(flagged: pd.DataFrame):
    """Styled view of the flagged rows, tinted by severity (the Flag column carries the meaning in text)."""
    columns = [
        "Date", "Week", "Day", "Passage_No", "Viability_Pct", "Instrument_QC_Status",
        "Flag", "Scientist_Notes", "SOP_Protocol", "ELN_Record_ID",
    ]
    view = flagged[columns].copy()
    view["Date"] = view["Date"].dt.strftime("%Y-%m-%d")
    severity = flagged["Severity"]

    def tint(row: pd.Series) -> list[str]:
        return [SEVERITY_TINT[severity.loc[row.name]]] * len(row)

    return view.style.apply(tint, axis=1).format({"Viability_Pct": "{:.1f}"})


def metric_card(label, value, delta=None, delta_color="normal", caption=None, help=None, history=None):
    with st.container(border=True):
        kwargs = {}
        if history is not None and len(history) >= 2:
            kwargs = dict(chart_data=list(history), chart_type="line")
        st.metric(label, value, delta, delta_color=delta_color, help=help, **kwargs)
        st.caption(caption or " ")


def deviation_summary(data: pd.DataFrame) -> str:
    """Narrative on real-time SOP deviation flagging, grounded in the first viability breach in the data."""
    client = data.iloc[0]["Client_Name"]
    breaches = data[data["Viability_Pct"] < VIABILITY_MIN_PCT]
    if breaches.empty:
        return (
            f"No viability breach occurred in this campaign. Real-time flagging keeps it that way: any day "
            f"below {VIABILITY_MIN_PCT:.0f}% viability triggers the fallback steps in docs/CONTINGENCY_MAP.md "
            f"the same day, before it can reach a {client} shipment."
        )

    first, worst = breaches.iloc[0], breaches["Viability_Pct"].min()
    recovered = data[(data["Date"] > breaches.iloc[-1]["Date"]) & (data["Viability_Pct"] >= VIABILITY_MIN_PCT)]
    recovery = (
        f" Viability was back above {VIABILITY_MIN_PCT:.0f}% by {recovered.iloc[0]['Day']} "
        f"({recovered.iloc[0]['Viability_Pct']:.1f}%) and the campaign closed at "
        f"{data.iloc[-1]['Viability_Pct']:.1f}% viability."
        if not recovered.empty else ""
    )
    return (
        f"**{first['Day']} ({first['Date']:%d %b %Y}):** viability fell to {worst:.1f}%, below the "
        f"{VIABILITY_MIN_PCT:.0f}% minimum that docs/SOPS.md requires for downstream drug-testing deliverables. "
        f"Because the deviation was flagged the same day, the contingency was activated immediately (FBS raised "
        f"from 10% to 15% and a backup culture plate seeded) instead of at the next review.{recovery}\n\n"
        f"Without real-time flagging, a drop like this can go unnoticed until end-of-week review or "
        f"pre-shipment QC. At that point the material fails the viability requirement, the culture must be "
        f"re-expanded and re-banked, and a {client} shipment is re-scheduled. Catching it on the day it "
        f"happens keeps recovery inside the campaign timeline and protects the client delivery date."
    )


def render_roi_tab(data: pd.DataFrame) -> None:
    weeks = int(data["Week_No"].max())
    st.subheader("Executive Business Case & ROI")

    scientists = st.slider(
        "Number of Scientists on Team", min_value=1, max_value=100, value=DEFAULT_SCIENTISTS, step=1,
        help="Team size used to scale the estimated AI-automation savings.",
    )
    total_hours = scientists * HOURS_SAVED_PER_SCIENTIST_WEEK * weeks
    ftes = total_hours / (HOURS_PER_FTE_WEEK * weeks)
    cost_avoided = total_hours * LOADED_COST_PER_HOUR

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(
            "Total Hours Saved", f"{total_hours:,.0f} hrs",
            caption=f"{scientists} scientists × {HOURS_SAVED_PER_SCIENTIST_WEEK} h/week × {weeks} weeks",
            help="Hours returned to science across the campaign by automating logging, ELN entry and presentation generation.",
        )
    with c2:
        metric_card(
            "Equivalent FTEs Unlocked", f"{ftes:,.1f} FTEs",
            caption=f"{total_hours:,.0f} h ÷ ({HOURS_PER_FTE_WEEK} h/week × {weeks} weeks)",
            help=f"Full-time equivalents, assuming a {HOURS_PER_FTE_WEEK}-hour work week.",
        )
    with c3:
        metric_card(
            "Estimated Cost Avoidance", f"${cost_avoided:,.0f}",
            caption=f"{total_hours:,.0f} h × ${LOADED_COST_PER_HOUR}/h loaded cost",
            help="Labor cost only. Avoided re-shipment delays are not included.",
        )
    st.caption(
        f"Assumptions: {HOURS_SAVED_PER_SCIENTIST_WEEK} hours saved per scientist per week (a conservative planning "
        f"estimate for AI-assisted logging, ELN entry and presentation generation), ${LOADED_COST_PER_HOUR}/hour "
        f"average loaded cost, {HOURS_PER_FTE_WEEK}-hour FTE week. These are planning inputs, not measurements "
        f"from the bench data."
    )

    st.subheader("Real-time SOP deviation flagging prevents re-shipment delays")
    st.markdown(deviation_summary(data))
    st.caption("The delay avoided is described qualitatively. It is not converted into dollars in the cost figure above.")


def main() -> None:
    st.set_page_config(page_title="HER3 KO Project Dashboard", page_icon="🧬", layout="wide")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    try:
        data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:  # unreadable workbook, missing sheet or columns
        st.error(f"Could not load '{SHEET_NAME}': {exc}")
        st.stop()

    data[["Flag", "Severity"]] = data.apply(lambda r: pd.Series(classify_row(r)), axis=1)

    # ---- Sidebar: week filter ----
    week_labels = data.drop_duplicates("Week_No").sort_values("Week_No")["Week"].tolist()
    st.sidebar.header("Filters")
    selected_weeks = st.sidebar.multiselect("Week", week_labels, default=week_labels)
    df = data[data["Week"].isin(selected_weeks)]
    st.sidebar.caption(f"Showing {len(df)} of {len(data)} days (applies to the Project Dashboard tab)")

    first = data.iloc[0]
    st.title(f"{first['Target_Gene']} Knockout · {first['Cell_Line']}")
    st.caption(
        f"{first['Project_ID']} · {first['Client_Name']} · "
        f"{data['Date'].min():%d %b %Y} – {data['Date'].max():%d %b %Y}"
    )

    tab_project, tab_roi = st.tabs(["Project Dashboard", "Executive Business Case & ROI"])
    with tab_project:
        render_project_tab(df)
    with tab_roi:
        render_roi_tab(data)


def render_project_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Select at least one week in the sidebar.")
        return

    # ---- Metric cards (latest values within the current selection) ----
    latest = df.iloc[-1]
    ko = df.dropna(subset=["HER3_Knockout_Pct"])
    st.subheader("Latest status")
    c1, c2, c3 = st.columns(3)
    with c1:
        passage_change = int(latest["Passage_No"] - df.iloc[0]["Passage_No"])
        metric_card(
            "Latest Passage Number", f"P{int(latest['Passage_No'])}",
            f"{passage_change:+d} passages in selection", delta_color="off",
            caption=f"{latest['Day']} · {latest['Date']:%d %b %Y}",
            help="Passage number on the most recent day in the selected weeks.",
            history=df["Passage_No"],
        )
    with c2:
        delta = None
        if len(df) > 1:
            delta = f"{latest['Viability_Pct'] - df.iloc[-2]['Viability_Pct']:+.1f} pts vs previous day"
        below = latest["Viability_Pct"] < VIABILITY_MIN_PCT
        metric_card(
            "Latest Viability", f"{latest['Viability_Pct']:.1f}%", delta,
            caption=("🔴 Below the 80% QC minimum" if below else "✅ Above the 80% QC minimum")
            + f" · {latest['Day']}",
            help="Viability must stay above 80% for downstream drug-testing deliverables (docs/SOPS.md).",
            history=df["Viability_Pct"],
        )
    with c3:
        if ko.empty:
            metric_card(
                "Latest HER3 Knockout %", "—", caption="No knockout measurement in selected weeks",
                help="HER3 knockout % is only measured on flow-cytometry days.",
            )
        else:
            last_ko = ko.iloc[-1]
            delta = None
            if len(ko) > 1:
                delta = f"{last_ko['HER3_Knockout_Pct'] - ko.iloc[-2]['HER3_Knockout_Pct']:+.1f} pts vs previous measurement"
            metric_card(
                "Latest HER3 Knockout %", f"{last_ko['HER3_Knockout_Pct']:.1f}%", delta,
                caption=f"Measured {last_ko['Day']} · {last_ko['Date']:%d %b %Y}",
                help="HER3 knockout % is only measured on flow-cytometry days, so this is the most "
                     "recent measurement in the selected weeks, not necessarily the latest day.",
                history=ko["HER3_Knockout_Pct"],
            )

    if latest["Viability_Pct"] < VIABILITY_MIN_PCT:
        st.error(
            f"Latest viability is {latest['Viability_Pct']:.1f}% (< {VIABILITY_MIN_PCT:.0f}%). "
            "Trigger the fallback steps in docs/CONTINGENCY_MAP.md."
        )

    # ---- Dual-axis trend chart ----
    st.subheader("Viability and cell density")
    n_breach = int((df["Viability_Pct"] < VIABILITY_MIN_PCT).sum())
    st.caption(
        f"{n_breach} day(s) below {VIABILITY_MIN_PCT:.0f}% viability in the selection."
        if n_breach else f"No days below {VIABILITY_MIN_PCT:.0f}% viability in the selection."
    )
    st.plotly_chart(build_chart(df), width="stretch")

    # ---- Flagged rows ----
    flagged = df[df["Severity"] > 0]
    with st.expander(f"⚠️ Flagged rows: QC warnings, calibration issues, contingencies ({len(flagged)})"):
        if flagged.empty:
            st.success("No warnings, calibration issues or contingency triggers in the selected weeks.")
        else:
            st.caption("Red = contingency trigger or viability below 80%. Amber = QC warning, calibration issue or contingency follow-up.")
            st.dataframe(
                flagged_table(flagged), hide_index=True, width="stretch",
                column_config={
                    "Scientist_Notes": st.column_config.TextColumn("Scientist_Notes", width="large"),
                    "Flag": st.column_config.TextColumn("Flag", width="medium"),
                },
            )

    with st.expander("All daily data in selection"):
        table = df.drop(columns=["Week_No", "Severity"]).copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(table, hide_index=True, width="stretch")


main()
