"""
CSV EDA Studio - a Streamlit app that profiles any uploaded CSV and
turns it into interactive charts.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="CSV EDA Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#6C63FF", "#FF6584", "#22C3A6", "#FFB84D", "#4DA3FF", "#B36BFF"]
PLOT_TEMPLATE = "plotly_white"
SAMPLE_PATH = Path(__file__).parent / "sample_data.csv"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      .hero {
        background: linear-gradient(120deg, #6C63FF 0%, #4DA3FF 55%, #22C3A6 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(108, 99, 255, .25);
      }
      .hero h1 {margin: 0; font-size: 2rem; color: white;}
      .hero p {margin: .3rem 0 0; opacity: .92; font-size: 1.02rem;}
      .kpi {
        background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18);
        border-radius: 16px; padding: 1rem 1.2rem;
      }
      .kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
      .kpi .value {font-size: 1.8rem; font-weight: 700; line-height: 1.2;}
      .kpi .sub {font-size: .8rem; opacity: .65;}
      .insight {
        border-left: 4px solid #6C63FF; background: rgba(108,99,255,.08);
        padding: .7rem 1rem; border-radius: 0 12px 12px 0; margin-bottom: .6rem;
      }
      .stTabs [data-baseweb="tab-list"] {gap: .4rem;}
      .stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: .5rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --------------------------------------------------------------------------- #
# Data loading & cleaning
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
    raise ValueError("Could not decode the file.")


def split_columns(df: pd.DataFrame):
    """Return (numeric, categorical, id_like) column lists."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    id_like, categorical = [], []
    for c in df.columns.difference(numeric, sort=False):
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        n_unique = s.nunique(dropna=True)
        if n_unique == len(s.dropna()) and n_unique > 20:
            id_like.append(c)
        elif n_unique <= 30:
            categorical.append(c)
        else:
            id_like.append(c)
    # Integer columns with very few distinct values behave like categories too.
    return numeric, categorical, id_like


def percent_like_columns(df: pd.DataFrame, numeric: list[str]) -> list[str]:
    return [c for c in numeric if any(k in c.lower() for k in ("pct", "percent", "%"))]


def clean(df: pd.DataFrame, drop_dupes: bool, pct_invalid_to_nan: bool, missing: str):
    log = []
    out = df.copy()
    numeric, _, _ = split_columns(out)

    if drop_dupes:
        n = int(out.duplicated().sum())
        if n:
            out = out.drop_duplicates()
            log.append(f"Removed **{n}** duplicate row(s).")

    if pct_invalid_to_nan:
        for c in percent_like_columns(out, numeric):
            bad = (out[c] < 0) | (out[c] > 100)
            if bad.any():
                out.loc[bad, c] = np.nan
                log.append(f"Set **{int(bad.sum())}** value(s) outside 0-100 in `{c}` to missing.")

    numeric, categorical, _ = split_columns(out)
    if missing == "Drop rows with missing values":
        n = int(out.isna().any(axis=1).sum())
        if n:
            out = out.dropna()
            log.append(f"Dropped **{n}** row(s) containing missing values.")
    elif missing == "Fill numeric with median":
        for c in numeric:
            k = int(out[c].isna().sum())
            if k:
                out[c] = out[c].fillna(out[c].median())
                log.append(f"Filled **{k}** missing value(s) in `{c}` with the median.")
    return out.reset_index(drop=True), log


def iqr_outliers(s: pd.Series):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return lo, hi, s[(s < lo) | (s > hi)]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📊 CSV EDA Studio")
    st.caption("Upload a CSV and explore it instantly.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv", "txt"])
    use_sample = st.checkbox(
        "Use bundled sample data",
        value=False,
        disabled=not SAMPLE_PATH.exists(),
        help="Student-support demo dataset included with the app.",
    )

    st.divider()
    st.markdown("### Cleaning options")
    opt_dupes = st.toggle("Remove duplicate rows", value=True)
    opt_pct = st.toggle(
        "Treat % values outside 0-100 as missing",
        value=True,
        help="Applies to numeric columns whose name contains 'pct', 'percent' or '%'.",
    )
    opt_missing = st.radio(
        "Missing values",
        ["Keep as is", "Drop rows with missing values", "Fill numeric with median"],
    )

# --------------------------------------------------------------------------- #
# Hero + load
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>CSV EDA Studio</h1>
      <p>Automatic profiling, data-quality checks and interactive visuals for any CSV.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

raw_df = None
source_name = ""
if uploaded is not None:
    try:
        raw_df = read_csv(uploaded.getvalue())
        source_name = uploaded.name
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read that file: {e}")
        st.stop()
elif use_sample and SAMPLE_PATH.exists():
    raw_df = read_csv(SAMPLE_PATH.read_bytes())
    source_name = SAMPLE_PATH.name

if raw_df is None:
    st.info("👈 Upload a CSV in the sidebar (or tick **Use bundled sample data**) to begin.")
    st.stop()

if raw_df.empty:
    st.warning("The file has no rows.")
    st.stop()

df, clean_log = clean(raw_df, opt_dupes, opt_pct, opt_missing)
numeric_cols, cat_cols, id_cols = split_columns(df)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs(
    [
        "🏠 Overview",
        "🧹 Data Quality",
        "📈 Distributions",
        "🔗 Relationships",
        "🎯 Target Analysis",
        "🚨 Outliers",
        "🗂 Data Explorer",
    ]
)

# ---------------------------- Overview ------------------------------------- #
with tabs[0]:
    st.caption(f"Source: `{source_name}`")
    miss_total = int(df.isna().sum().sum())
    miss_pct = miss_total / df.size * 100 if df.size else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Rows", f"{len(df):,}", f"{len(raw_df):,} in raw file"), unsafe_allow_html=True)
    c2.markdown(kpi("Columns", f"{df.shape[1]}", f"{len(numeric_cols)} numeric · {len(cat_cols)} categorical"), unsafe_allow_html=True)
    c3.markdown(kpi("Missing cells", f"{miss_total:,}", f"{miss_pct:.2f}% of all cells"), unsafe_allow_html=True)
    c4.markdown(kpi("Duplicate rows", f"{int(raw_df.duplicated().sum())}", "in raw file"), unsafe_allow_html=True)
    c5.markdown(kpi("Memory", f"{df.memory_usage(deep=True).sum() / 1024:,.1f} KB", ""), unsafe_allow_html=True)

    if clean_log:
        with st.expander("Cleaning applied", expanded=True):
            for line in clean_log:
                st.markdown(f"- {line}")

    st.subheader("Key insights")
    insights = []
    if numeric_cols:
        skew = df[numeric_cols].skew(numeric_only=True).abs().sort_values(ascending=False)
        if len(skew) and skew.iloc[0] > 1:
            insights.append(f"`{skew.index[0]}` is the most skewed numeric column (|skew| = {skew.iloc[0]:.2f}).")
        if len(numeric_cols) >= 2:
            signed_corr = df[numeric_cols].corr()
            abs_corr = signed_corr.abs().to_numpy(copy=True)
            np.fill_diagonal(abs_corr, 0)
            abs_corr = np.nan_to_num(abs_corr)
            if abs_corr.max() > 0:
                a, b = np.unravel_index(abs_corr.argmax(), abs_corr.shape)
                signed = signed_corr.iloc[a, b]
                insights.append(
                    f"Strongest numeric relationship: `{numeric_cols[a]}` & `{numeric_cols[b]}` (r = {signed:.2f})."
                )
    worst = df.isna().mean().sort_values(ascending=False)
    if len(worst) and worst.iloc[0] > 0:
        insights.append(f"`{worst.index[0]}` has the most missing data ({worst.iloc[0] * 100:.1f}%).")
    for c in cat_cols:
        vc = df[c].value_counts(normalize=True)
        if len(vc) and vc.iloc[0] > 0.7 and len(vc) > 1:
            insights.append(f"`{c}` is imbalanced: '{vc.index[0]}' makes up {vc.iloc[0] * 100:.0f}% of rows.")
    if not insights:
        insights.append("No major issues detected - the data looks clean.")
    for t in insights:
        st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)

    st.subheader("Summary statistics")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc["skew"] = df[numeric_cols].skew()
        desc["missing"] = df[numeric_cols].isna().sum()
        st.dataframe(desc.round(2), width="stretch")
    else:
        st.info("No numeric columns found.")

    st.subheader("Column types")
    types = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "role": [
                "numeric" if c in numeric_cols else "categorical" if c in cat_cols else "identifier / text"
                for c in df.columns
            ],
            "unique": [df[c].nunique() for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.dataframe(types, width="stretch", hide_index=True)

# ---------------------------- Data quality --------------------------------- #
with tabs[1]:
    st.subheader("Missing values")
    miss = df.isna().sum().rename("missing").to_frame()
    miss["percent"] = (miss["missing"] / len(df) * 100).round(2)
    if miss["missing"].sum() == 0:
        st.success("No missing values after cleaning.")
    else:
        m = miss[miss["missing"] > 0].reset_index().rename(columns={"index": "column"})
        fig = px.bar(m, x="percent", y="column", orientation="h", text="missing",
                     color_discrete_sequence=[PALETTE[1]], title="Missing values by column (%)")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

    st.subheader("Raw data issues")
    left, right = st.columns(2)
    with left:
        dup_mask = raw_df.duplicated(keep=False)
        st.markdown(f"**Duplicate rows in raw file:** {int(raw_df.duplicated().sum())}")
        if dup_mask.any():
            st.dataframe(raw_df[dup_mask], width="stretch", hide_index=True)
        else:
            st.caption("None found.")
    with right:
        raw_num = raw_df.select_dtypes(include="number").columns.tolist()
        issues = []
        for c in percent_like_columns(raw_df, raw_num):
            bad = raw_df[(raw_df[c] < 0) | (raw_df[c] > 100)]
            for i, row in bad.iterrows():
                issues.append({"row": i, "column": c, "value": row[c]})
        st.markdown(f"**Out-of-range percentages:** {len(issues)}")
        if issues:
            st.dataframe(pd.DataFrame(issues), width="stretch", hide_index=True)
        else:
            st.caption("None found.")

    miss_rows = raw_df[raw_df.isna().any(axis=1)]
    st.markdown(f"**Rows with missing values in raw file:** {len(miss_rows)}")
    if len(miss_rows):
        st.dataframe(miss_rows, width="stretch", hide_index=True)

# ---------------------------- Distributions -------------------------------- #
with tabs[2]:
    if numeric_cols:
        st.subheader("Numeric distributions")
        c1, c2, c3 = st.columns([2, 2, 1])
        col = c1.selectbox("Column", numeric_cols, key="dist_col")
        hue_opts = ["None"] + cat_cols
        hue = c2.selectbox("Colour by", hue_opts, key="dist_hue")
        bins = c3.slider("Bins", 5, 80, 25)
        fig = px.histogram(
            df, x=col, nbins=bins, marginal="box",
            color=None if hue == "None" else hue,
            barmode="overlay", opacity=0.75, color_discrete_sequence=PALETTE,
            title=f"Distribution of {col}",
        )
        st.plotly_chart(style_fig(fig, 480), width="stretch")

        st.markdown("**All numeric columns at a glance**")
        melted = df[numeric_cols].melt(var_name="feature", value_name="value")
        fig = px.violin(melted, x="feature", y="value", color="feature", box=True,
                        color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    else:
        st.info("No numeric columns to plot.")

    if cat_cols:
        st.subheader("Categorical distributions")
        cc = st.selectbox("Categorical column", cat_cols, key="cat_col")
        vc = df[cc].value_counts(dropna=False).reset_index()
        vc.columns = [cc, "count"]
        a, b = st.columns(2)
        with a:
            fig = px.bar(vc, x=cc, y="count", color=cc, text="count", color_discrete_sequence=PALETTE,
                         title=f"Counts of {cc}")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 380), width="stretch")
        with b:
            fig = px.pie(vc, names=cc, values="count", hole=0.55, color_discrete_sequence=PALETTE,
                         title=f"Share of {cc}")
            st.plotly_chart(style_fig(fig, 380), width="stretch")

# ---------------------------- Relationships -------------------------------- #
with tabs[3]:
    if len(numeric_cols) >= 2:
        st.subheader("Correlation matrix")
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        corr = df[numeric_cols].corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale="RdBu_r", title=f"{method.title()} correlation")
        st.plotly_chart(style_fig(fig, 480), width="stretch")

        st.subheader("Scatter explorer")
        c1, c2, c3, c4 = st.columns(4)
        x = c1.selectbox("X axis", numeric_cols, index=0, key="sx")
        y = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="sy")
        color = c3.selectbox("Colour", ["None"] + cat_cols + numeric_cols, key="sc")
        trend = c4.checkbox("Trend line", value=True)
        plot_df = df.dropna(subset=[x, y]).copy()
        fig = px.scatter(
            plot_df, x=x, y=y,
            color=None if color == "None" else color,
            color_discrete_sequence=PALETTE, opacity=0.8,
            hover_data=[c for c in id_cols[:1]],
            title=f"{y} vs {x}",
        )
        if trend and len(plot_df) > 2:
            slope, intercept = np.polyfit(plot_df[x], plot_df[y], 1)
            xs = np.linspace(plot_df[x].min(), plot_df[x].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                     name="Linear fit", line=dict(color="#888", dash="dash")))
        fig.update_traces(marker=dict(size=10), selector=dict(mode="markers"))
        st.plotly_chart(style_fig(fig, 480), width="stretch")

        if len(numeric_cols) >= 3:
            st.subheader("Scatter matrix")
            sm_color = st.selectbox("Colour by", ["None"] + cat_cols, key="sm_color")
            fig = px.scatter_matrix(
                df, dimensions=numeric_cols[:6],
                color=None if sm_color == "None" else sm_color,
                color_discrete_sequence=PALETTE, opacity=0.7,
            )
            fig.update_traces(diagonal_visible=False, marker=dict(size=4))
            st.plotly_chart(style_fig(fig, 650), width="stretch")
    else:
        st.info("Need at least two numeric columns for relationship plots.")

# ---------------------------- Target analysis ------------------------------ #
with tabs[4]:
    target_candidates = [c for c in cat_cols if 2 <= df[c].nunique() <= 10]
    if not target_candidates or not numeric_cols:
        st.info("Add a categorical column with 2-10 classes (e.g. Yes/No) to unlock target analysis.")
    else:
        default_idx = next(
            (i for i, c in enumerate(target_candidates) if "need" in c.lower() or "target" in c.lower() or "label" in c.lower()),
            0,
        )
        target = st.selectbox("Target column", target_candidates, index=default_idx)
        tdf = df.dropna(subset=[target])
        classes = sorted(tdf[target].astype(str).unique())

        a, b = st.columns([1, 2])
        with a:
            vc = tdf[target].value_counts().reset_index()
            vc.columns = [target, "count"]
            fig = px.pie(vc, names=target, values="count", hole=0.6, color_discrete_sequence=PALETTE,
                         title="Class balance")
            st.plotly_chart(style_fig(fig, 380), width="stretch")
        with b:
            means = tdf.groupby(target)[numeric_cols].mean().reset_index().melt(
                id_vars=target, var_name="feature", value_name="mean")
            fig = px.bar(means, x="feature", y="mean", color=target, barmode="group",
                         color_discrete_sequence=PALETTE, title="Average of each feature by class")
            st.plotly_chart(style_fig(fig, 380), width="stretch")

        st.subheader("Feature comparison across classes")
        feat = st.selectbox("Feature", numeric_cols, key="t_feat")
        kind = st.radio("Chart", ["Box", "Violin", "Histogram"], horizontal=True)
        if kind == "Box":
            fig = px.box(tdf, x=target, y=feat, color=target, points="all", color_discrete_sequence=PALETTE)
        elif kind == "Violin":
            fig = px.violin(tdf, x=target, y=feat, color=target, box=True, points="all",
                            color_discrete_sequence=PALETTE)
        else:
            fig = px.histogram(tdf, x=feat, color=target, barmode="overlay", opacity=0.7,
                               color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False if kind != "Histogram" else True)
        st.plotly_chart(style_fig(fig, 440), width="stretch")

        if len(classes) == 2:
            st.subheader("Risk curve")
            pos = st.selectbox("Positive class", classes,
                               index=classes.index("Yes") if "Yes" in classes else 0)
            nb = st.slider("Number of bins", 3, 12, 6, key="risk_bins")
            fig = go.Figure()
            for i, f in enumerate(numeric_cols):
                s = tdf[[f, target]].dropna()
                if s[f].nunique() < 3:
                    continue
                s["bin"] = pd.qcut(s[f], q=nb, duplicates="drop")
                g = s.groupby("bin", observed=True).agg(
                    rate=(target, lambda v: (v.astype(str) == pos).mean() * 100),
                    mid=(f, "mean"),
                )
                fig.add_trace(go.Scatter(x=g["mid"], y=g["rate"], mode="lines+markers", name=f,
                                         line=dict(color=PALETTE[i % len(PALETTE)], width=3)))
            fig.update_layout(title=f"% of '{pos}' across quantile bins of each feature",
                              xaxis_title="Feature value (bin mean)", yaxis_title=f"% {pos}")
            st.plotly_chart(style_fig(fig, 440), width="stretch")

            enc = (tdf[target].astype(str) == pos).astype(int)
            cors = tdf[numeric_cols].corrwith(enc).dropna().sort_values()
            fig = px.bar(cors, orientation="h", color=cors.values, color_continuous_scale="RdBu_r",
                         range_color=[-1, 1], title=f"Correlation with target = '{pos}'")
            fig.update_layout(coloraxis_showscale=False, showlegend=False,
                              xaxis_title="correlation", yaxis_title="")
            st.plotly_chart(style_fig(fig, 320), width="stretch")

            st.subheader("Interactive threshold check")
            tf = st.selectbox("Feature", numeric_cols, key="thr_feat")
            lo, hi = float(tdf[tf].min()), float(tdf[tf].max())
            thr = st.slider("Flag rows where feature is below", lo, hi, float(tdf[tf].median()))
            flagged = tdf[tdf[tf] < thr]
            is_pos = flagged[target].astype(str) == pos
            tot_pos = int((tdf[target].astype(str) == pos).sum())
            m1, m2, m3 = st.columns(3)
            m1.markdown(kpi("Rows flagged", f"{len(flagged)}", f"{len(flagged) / len(tdf) * 100:.0f}% of data"), unsafe_allow_html=True)
            m2.markdown(kpi("Precision", f"{(is_pos.mean() * 100 if len(flagged) else 0):.0f}%", f"flagged rows that are '{pos}'"), unsafe_allow_html=True)
            m3.markdown(kpi("Recall", f"{(is_pos.sum() / tot_pos * 100 if tot_pos else 0):.0f}%", f"of all '{pos}' rows caught"), unsafe_allow_html=True)

# ---------------------------- Outliers ------------------------------------- #
with tabs[5]:
    if numeric_cols:
        rows = []
        for c in numeric_cols:
            lo, hi, out = iqr_outliers(df[c])
            rows.append({"column": c, "lower fence": round(lo, 2), "upper fence": round(hi, 2),
                         "outliers": len(out), "% of rows": round(len(out) / len(df) * 100, 2)})
        st.subheader("IQR outlier summary (1.5 x IQR)")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        oc = st.selectbox("Inspect column", numeric_cols, key="out_col")
        lo, hi, out = iqr_outliers(df[oc])
        fig = px.box(df, x=oc, points="all", color_discrete_sequence=[PALETTE[0]], title=f"Box plot: {oc}")
        st.plotly_chart(style_fig(fig, 300), width="stretch")
        if len(out):
            st.markdown(f"**{len(out)} outlier row(s):**")
            st.dataframe(df.loc[out.index], width="stretch", hide_index=True)
        else:
            st.success("No outliers detected by the IQR rule.")
    else:
        st.info("No numeric columns.")

# ---------------------------- Data explorer -------------------------------- #
with tabs[6]:
    st.subheader("Filter & browse")
    view = df.copy()
    with st.expander("Filters", expanded=False):
        for c in numeric_cols:
            s = df[c].dropna()
            if s.empty or s.min() == s.max():
                continue
            lo, hi = float(s.min()), float(s.max())
            rng = st.slider(c, lo, hi, (lo, hi), key=f"f_{c}")
            view = view[view[c].between(rng[0], rng[1]) | view[c].isna()]
        for c in cat_cols:
            opts = sorted(df[c].dropna().astype(str).unique())
            pick = st.multiselect(c, opts, default=opts, key=f"f_{c}")
            view = view[view[c].astype(str).isin(pick) | view[c].isna()]
    st.caption(f"Showing {len(view):,} of {len(df):,} rows")
    st.dataframe(view, width="stretch", hide_index=True, height=420)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode("utf-8"),
                       "cleaned_data.csv", "text/csv", width="stretch")
    d2.download_button("⬇️ Download filtered view", view.to_csv(index=False).encode("utf-8"),
                       "filtered_data.csv", "text/csv", width="stretch")
