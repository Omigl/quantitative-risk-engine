"""
app.py — Streamlit Dashboard Orchestration (5-Tab Layout)
============================================================
Business Purpose:
    Main entry point for the Quantitative Risk & Performance Analytics Engine.
    Orchestrates the full pipeline across 5 interactive tabs:
    1. Portfolio Overview — holdings, weights, allocation, exposure
    2. Risk Analytics — volatility, correlation, covariance, drawdown
    3. VaR & ES — Parametric, Historical, Monte Carlo comparison
    4. Stress Testing — historical, hypothetical, custom scenario builder
    5. Monitoring — rolling metrics, drawdown tracking, limit breaches

    All state is driven from the YAML configuration file.

Launch:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Project root path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.performance import PerformanceAnalyzer
from src.data.downloader import MarketDataDownloader
from src.data.transformer import ReturnTransformer
from src.data.validator import DataValidator
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.exposure import ExposureAnalyzer
from src.risk.engine import RiskEngine
from src.risk.limits import RiskLimitMonitor
from src.risk.monitoring import RiskMonitor
from src.risk.stress import StressTestEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

CONFIG_PATH: Path = PROJECT_ROOT / "config" / "portfolio_config.yaml"


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: Path) -> Dict[str, Any]:
    """Load and parse the YAML configuration file."""
    if not path.exists():
        st.error(f"Configuration file not found: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quantitative Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h4 {
        color: #8892b0; font-size: 0.8rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px;
    }
    .metric-card .value {
        color: #ccd6f6; font-size: 1.8rem; font-weight: 700;
    }
    .metric-card .value.positive { color: #64ffda; }
    .metric-card .value.negative { color: #ff6b6b; }

    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #ccd6f6;
        border-bottom: 2px solid #64ffda;
        padding-bottom: 8px; margin-top: 24px; margin-bottom: 16px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a192f 0%, #112240 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 { color: #64ffda; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

config: Dict[str, Any] = load_config(CONFIG_PATH)

portfolio_cfg = config["portfolio"]
data_cfg = config["data"]
perf_cfg = config["performance"]
risk_cfg = config["risk"]
mc_cfg = config.get("monte_carlo", {})
monitoring_cfg = config.get("monitoring", {})
limits_cfg = config.get("risk_limits", {})
stress_cfg = config.get("stress_scenarios", {})

tickers: List[str] = portfolio_cfg["tickers"]
shares: Dict[str, int] = portfolio_cfg["shares"]
sectors: Dict[str, str] = portfolio_cfg.get("sectors", {})


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 📊 Risk Engine")
    st.markdown("---")

    st.markdown("### Portfolio Holdings")
    holdings_df = pd.DataFrame(
        {"Ticker": list(shares.keys()), "Shares": list(shares.values())}
    )
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Risk Parameters")
    st.markdown(f"**Confidence:** {risk_cfg['confidence_levels']}")
    st.markdown(f"**Holding Period:** {risk_cfg['holding_period_days']}d")
    st.markdown(f"**Risk-Free Rate:** {perf_cfg['risk_free_rate']*100:.2f}%")
    st.markdown(f"**MC Simulations:** {mc_cfg.get('n_simulations', 10000):,}")

    st.markdown("---")
    refresh: bool = st.button("🔄 Refresh Data", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def run_data_pipeline(
    _tickers: List[str], lookback_years: int,
    ffill_limit: int, anomaly_threshold: float, missing_alert: float,
) -> Dict[str, Any]:
    downloader = MarketDataDownloader(tickers=_tickers, lookback_years=lookback_years)
    prices = downloader.fetch_prices()

    validator = DataValidator(
        anomaly_threshold_pct=anomaly_threshold,
        missing_data_alert_pct=missing_alert,
    )
    validation = validator.validate(prices)

    transformer = ReturnTransformer(forward_fill_limit=ffill_limit)
    filled_prices = transformer.fill_missing(prices)
    simple_returns = transformer.transform(filled_prices, method="simple")
    log_returns = transformer.transform(filled_prices, method="log")

    return {
        "prices": filled_prices,
        "simple_returns": simple_returns,
        "log_returns": log_returns,
        "anomalies": validation["anomalies"],
        "missing_data": validation["missing_data"],
    }


if refresh:
    st.cache_data.clear()

try:
    pipeline = run_data_pipeline(
        _tickers=tickers,
        lookback_years=data_cfg["lookback_years"],
        ffill_limit=data_cfg["forward_fill_limit"],
        anomaly_threshold=data_cfg["anomaly_threshold_pct"],
        missing_alert=data_cfg["missing_data_alert_pct"],
    )
except Exception as e:
    st.error(f"❌ Data pipeline failed: {e}")
    st.stop()

prices = pipeline["prices"]
simple_returns = pipeline["simple_returns"]
log_returns = pipeline["log_returns"]


# ═══════════════════════════════════════════════════════════════════════════
# BUILD PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════

constructor = PortfolioConstructor(tickers=tickers, shares=shares, sectors=sectors)
snapshot = constructor.build_snapshot(prices)
weights = snapshot.weights
portfolio_value = snapshot.total_value
market_values = snapshot.market_values

port_returns = constructor.compute_portfolio_returns(simple_returns, weights)

# Instantiate all engines
perf_analyzer = PerformanceAnalyzer(
    risk_free_rate=perf_cfg["risk_free_rate"],
    trading_days_per_year=perf_cfg["trading_days_per_year"],
)
risk_engine = RiskEngine(
    confidence_levels=risk_cfg["confidence_levels"],
    holding_period_days=risk_cfg["holding_period_days"],
    trading_days_per_year=perf_cfg["trading_days_per_year"],
    n_simulations=mc_cfg.get("n_simulations", 10000),
    random_seed=mc_cfg.get("random_seed", 42),
)
exposure_analyzer = ExposureAnalyzer(sectors=sectors)
stress_engine = StressTestEngine(portfolio_value=portfolio_value)
risk_monitor = RiskMonitor(
    rolling_window=monitoring_cfg.get("rolling_window_days", 63),
    confidence_levels=risk_cfg["confidence_levels"],
    trading_days_per_year=perf_cfg["trading_days_per_year"],
)
limit_monitor = RiskLimitMonitor(
    var_limits=limits_cfg.get("var_limits", {}),
    exposure_limits=limits_cfg.get("exposure_limits", {}),
    concentration_limits=limits_cfg.get("concentration_limits", {}),
)

# Plotly theme defaults
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8892b0"),
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified",
)


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style='text-align: center; padding: 12px 0 4px 0;'>
        <h1 style='color: #ccd6f6; font-weight: 700; margin-bottom: 4px;'>
            Quantitative Risk & Performance Engine
        </h1>
        <p style='color: #8892b0; font-size: 1.05rem;'>
            Cross-Asset Portfolio Analytics Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top-line metrics
ann_ret = perf_analyzer.annualized_return(simple_returns)
ann_vol = perf_analyzer.annualized_volatility(simple_returns)
sharpe = perf_analyzer.sharpe_ratio(simple_returns)
mdd = perf_analyzer.max_drawdown(simple_returns)

port_ann_ret = float(ann_ret.dot(weights))
port_ann_vol = float(ann_vol.dot(weights))
port_sharpe = float(sharpe.dot(weights))
port_mdd = float(mdd.max())

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, fmt, css_class in [
    (c1, "Portfolio Value", portfolio_value, "${:,.0f}", ""),
    (c2, "Ann. Return", port_ann_ret * 100, "{:.2f}%", "positive" if port_ann_ret >= 0 else "negative"),
    (c3, "Ann. Volatility", port_ann_vol * 100, "{:.2f}%", ""),
    (c4, "Sharpe Ratio", port_sharpe, "{:.3f}", "positive" if port_sharpe > 0.5 else "negative"),
    (c5, "Max Drawdown", port_mdd * 100, "{:.2f}%", "negative"),
]:
    col.markdown(
        f"<div class='metric-card'><h4>{label}</h4>"
        f"<div class='value {css_class}'>{fmt.format(value)}</div></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Portfolio Overview",
    "📊 Risk Analytics",
    "⚠️ VaR & ES",
    "🔥 Stress Testing",
    "📈 Monitoring",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>🥧 Portfolio Allocation</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(data=go.Pie(
            labels=market_values.index.tolist(),
            values=market_values.values,
            hole=0.45, textinfo="label+percent",
            textfont=dict(size=12, color="#ccd6f6"),
            marker=dict(colors=px.colors.qualitative.Set2, line=dict(color="#0a192f", width=2)),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>📊 Sector Exposure</div>", unsafe_allow_html=True)
        sector_report = exposure_analyzer.sector_exposure(weights)
        fig_sector = px.bar(
            sector_report, x="Sector", y="Weight (%)",
            color="Sector", color_discrete_sequence=px.colors.qualitative.Set2,
            text="Weight (%)",
        )
        fig_sector.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        fig_sector.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_sector, use_container_width=True)

    # Holdings table
    st.markdown("<div class='section-header'>📋 Per-Asset Performance Summary</div>", unsafe_allow_html=True)
    perf_summary = perf_analyzer.summary(simple_returns)
    perf_summary["Weight (%)"] = np.round(weights.reindex(perf_summary.index).fillna(0) * 100, 2)
    perf_summary["Market Value ($)"] = np.round(market_values.reindex(perf_summary.index).fillna(0), 2)
    perf_summary["Sector"] = pd.Series(sectors).reindex(perf_summary.index).fillna("—")
    # Reorder columns
    col_order = ["Sector", "Weight (%)", "Market Value ($)", "Annualized Return (%)",
                 "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    perf_summary = perf_summary[[c for c in col_order if c in perf_summary.columns]]
    st.dataframe(
        perf_summary.style.format({
            "Weight (%)": "{:.2f}", "Market Value ($)": "${:,.2f}",
            "Annualized Return (%)": "{:.2f}", "Annualized Volatility (%)": "{:.2f}",
            "Sharpe Ratio": "{:.3f}", "Max Drawdown (%)": "{:.2f}",
        }).background_gradient(cmap="RdYlGn", subset=["Sharpe Ratio"]),
        use_container_width=True,
    )

    # Exposure details
    st.markdown("<div class='section-header'>🎯 Concentration Metrics</div>", unsafe_allow_html=True)
    hhi = exposure_analyzer.herfindahl_hirschman_index(weights)
    top3 = exposure_analyzer.top_n_concentration(weights, n=3)
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("HHI", f"{hhi:.4f}", help="Herfindahl-Hirschman Index. <0.15 = diversified, >0.25 = concentrated")
    ec2.metric("Top-3 Concentration", f"{top3*100:.1f}%")
    ec3.metric("Effective # of Assets", f"{1/hhi:.1f}", help="1/HHI — equivalent number of equal-weighted positions")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    # Cumulative returns chart
    st.markdown("<div class='section-header'>📈 Cumulative Returns</div>", unsafe_allow_html=True)
    cum_returns = perf_analyzer.cumulative_returns(simple_returns)
    port_cum = ((1 + simple_returns).dot(weights.reindex(simple_returns.columns).fillna(0))).cumprod() - 1
    cum_returns["Portfolio"] = port_cum

    fig_cum = px.line(
        cum_returns * 100,
        labels={"value": "Cumulative Return (%)", "variable": "Asset"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_cum.update_layout(**PLOTLY_LAYOUT, height=420,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_cum.update_traces(line=dict(width=2))
    for trace in fig_cum.data:
        if trace.name == "Portfolio":
            trace.line = dict(width=3.5, color="#64ffda", dash="dot")
    st.plotly_chart(fig_cum, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header'>🔗 Correlation Matrix</div>", unsafe_allow_html=True)
    corr_matrix = risk_engine.compute_correlation_matrix(simple_returns)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
        textfont=dict(size=12, color="#ccd6f6"),
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Drawdown chart
    st.markdown("<div class='section-header'>📉 Portfolio Drawdown</div>", unsafe_allow_html=True)
    port_wealth = (1 + port_returns).cumprod()
    port_peak = port_wealth.cummax()
    port_dd = (port_peak - port_wealth) / port_peak * -100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=port_dd.index, y=port_dd.values,
        fill="tozeroy", fillcolor="rgba(255, 107, 107, 0.3)",
        line=dict(color="#ff6b6b", width=1.5), name="Portfolio Drawdown",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig_dd.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # MCTR
    st.markdown("<div class='section-header'>🎯 Marginal Contribution to Risk</div>", unsafe_allow_html=True)
    cov_annual = risk_engine.compute_covariance_matrix(simple_returns)
    mctr_report = exposure_analyzer.marginal_contribution_to_risk(weights, cov_annual)
    if not mctr_report.empty:
        fig_mctr = px.bar(
            mctr_report, x="Ticker", y="Risk Contribution (%)",
            color="Risk Contribution (%)", color_continuous_scale="Reds",
            text="Risk Contribution (%)",
        )
        fig_mctr.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        fig_mctr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_mctr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: VaR & ES
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='section-header'>⚠️ Value-at-Risk Comparison</div>", unsafe_allow_html=True)

    param_var = risk_engine.parametric_var(simple_returns, weights, portfolio_value)
    hist_var = risk_engine.historical_var(simple_returns, weights, portfolio_value)
    mc_var = risk_engine.monte_carlo_var(simple_returns, weights, portfolio_value)

    var_records = []
    for cl in risk_cfg["confidence_levels"]:
        cl_int = int(cl * 100)
        var_records.append({
            "Confidence": f"{cl_int}%",
            "Parametric VaR ($)": f"${param_var.get(f'Parametric_VaR_{cl_int}', 0):,.2f}",
            "Historical VaR ($)": f"${hist_var.get(f'Historical_VaR_{cl_int}', 0):,.2f}",
            "Monte Carlo VaR ($)": f"${mc_var.get(f'MonteCarlo_VaR_{cl_int}', 0):,.2f}",
        })
    st.dataframe(pd.DataFrame(var_records), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>📊 Expected Shortfall Comparison</div>", unsafe_allow_html=True)

    param_es = risk_engine.parametric_es(simple_returns, weights, portfolio_value)
    hist_es = risk_engine.historical_es(simple_returns, weights, portfolio_value)
    mc_es = risk_engine.monte_carlo_es(simple_returns, weights, portfolio_value)

    es_records = []
    for cl in risk_cfg["confidence_levels"]:
        cl_int = int(cl * 100)
        es_records.append({
            "Confidence": f"{cl_int}%",
            "Parametric ES ($)": f"${param_es.get(f'Parametric_ES_{cl_int}', 0):,.2f}",
            "Historical ES ($)": f"${hist_es.get(f'Historical_ES_{cl_int}', 0):,.2f}",
            "Monte Carlo ES ($)": f"${mc_es.get(f'MonteCarlo_ES_{cl_int}', 0):,.2f}",
        })
    st.dataframe(pd.DataFrame(es_records), use_container_width=True, hide_index=True)

    # VaR visual comparison bar chart
    st.markdown("<div class='section-header'>📊 VaR Method Comparison (95% Confidence)</div>", unsafe_allow_html=True)
    var_compare = pd.DataFrame({
        "Method": ["Parametric", "Historical", "Monte Carlo"],
        "VaR ($)": [
            param_var.get("Parametric_VaR_95", 0),
            hist_var.get("Historical_VaR_95", 0),
            mc_var.get("MonteCarlo_VaR_95", 0),
        ],
    })
    fig_var_cmp = px.bar(
        var_compare, x="Method", y="VaR ($)",
        color="Method", color_discrete_sequence=["#64ffda", "#ff6b6b", "#ffd93d"],
        text="VaR ($)",
    )
    fig_var_cmp.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
    fig_var_cmp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_var_cmp, use_container_width=True)

    # Return distribution
    st.markdown("<div class='section-header'>📈 Portfolio Return Distribution</div>", unsafe_allow_html=True)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=port_returns.dropna().values * 100, nbinsx=80,
        marker_color="#64ffda", opacity=0.7, name="Returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    # Add VaR lines
    for cl in risk_cfg["confidence_levels"]:
        pct = (1 - cl) * 100
        var_val = float(np.percentile(port_returns.dropna(), pct)) * 100
        fig_dist.add_vline(x=var_val, line_dash="dash", line_color="#ff6b6b",
                           annotation_text=f"VaR {int(cl*100)}%", annotation_position="top left")
    fig_dist.update_layout(**PLOTLY_LAYOUT, height=350,
                           xaxis_title="Daily Return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st_col1, st_col2 = st.columns(2)

    # Historical scenarios
    with st_col1:
        st.markdown("<div class='section-header'>📅 Historical Scenarios</div>", unsafe_allow_html=True)
        hist_scenarios = stress_cfg.get("historical", {})
        if hist_scenarios:
            hist_results = stress_engine.run_historical_scenarios(prices, weights, hist_scenarios)
            if not hist_results.empty:
                st.dataframe(hist_results.style.format({
                    "Portfolio Return (%)": "{:.2f}",
                    "Portfolio Loss ($)": "${:,.2f}",
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No historical scenarios matched the data range.")
        else:
            st.info("No historical scenarios configured.")

    # Hypothetical scenarios
    with st_col2:
        st.markdown("<div class='section-header'>💡 Hypothetical Scenarios</div>", unsafe_allow_html=True)
        hypo_scenarios = stress_cfg.get("hypothetical", {})
        if hypo_scenarios:
            hypo_results = stress_engine.run_hypothetical_scenarios(
                weights, market_values, hypo_scenarios
            )
            if not hypo_results.empty:
                st.dataframe(hypo_results.style.format({
                    "Portfolio Return (%)": "{:.2f}",
                    "Portfolio Loss ($)": "${:,.2f}",
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No hypothetical scenarios returned results.")
        else:
            st.info("No hypothetical scenarios configured.")

    # Custom scenario builder
    st.markdown("<div class='section-header'>🛠️ Custom Scenario Builder</div>", unsafe_allow_html=True)
    st.markdown("Define per-asset shock percentages and compute the portfolio impact in real time.")

    custom_cols = st.columns(min(len(tickers), 4))
    custom_shocks: Dict[str, float] = {}
    for i, ticker in enumerate(tickers):
        col_idx = i % len(custom_cols)
        with custom_cols[col_idx]:
            shock = st.number_input(
                f"{ticker} Shock (%)", min_value=-100.0, max_value=100.0,
                value=0.0, step=1.0, key=f"shock_{ticker}",
            )
            custom_shocks[ticker] = shock

    if st.button("🚀 Run Custom Scenario", use_container_width=True):
        custom_result = stress_engine.run_custom_scenario(
            market_values, custom_shocks, label="Custom User Scenario"
        )
        st.dataframe(custom_result, use_container_width=True, hide_index=True)

        # Per-asset breakdown
        breakdown = stress_engine.per_asset_breakdown(market_values, custom_shocks)
        st.markdown("**Per-Asset Impact Breakdown:**")
        st.dataframe(breakdown.style.format({
            "Market Value ($)": "${:,.2f}", "Shock (%)": "{:.1f}",
            "Dollar Impact ($)": "${:,.2f}", "Post-Shock Value ($)": "${:,.2f}",
        }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: MONITORING
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    # Rolling volatility
    st.markdown("<div class='section-header'>📈 Rolling Volatility</div>", unsafe_allow_html=True)
    roll_windows = monitoring_cfg.get("rolling_windows", [21, 63, 126])
    multi_vol = risk_monitor.multi_window_volatility(port_returns, roll_windows)
    fig_vol = px.line(
        multi_vol * 100,
        labels={"value": "Annualized Volatility (%)", "variable": "Window"},
        color_discrete_sequence=["#64ffda", "#ffd93d", "#ff6b6b"],
    )
    fig_vol.update_layout(**PLOTLY_LAYOUT, height=350,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_vol, use_container_width=True)

    # Rolling VaR
    st.markdown("<div class='section-header'>📉 Rolling Value-at-Risk</div>", unsafe_allow_html=True)
    rolling_var = risk_monitor.rolling_var(port_returns, portfolio_value)
    rolling_var_df = pd.DataFrame(rolling_var)
    fig_rvar = px.line(
        rolling_var_df,
        labels={"value": "VaR ($)", "variable": "Metric"},
        color_discrete_sequence=["#ff6b6b", "#ffd93d"],
    )
    fig_rvar.update_layout(**PLOTLY_LAYOUT, height=350,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_rvar, use_container_width=True)

    # Drawdown tracking
    st.markdown("<div class='section-header'>📉 Drawdown Tracking</div>", unsafe_allow_html=True)
    dd_data = risk_monitor.drawdown_tracking(port_returns)

    dd_col1, dd_col2, dd_col3 = st.columns(3)
    current_dd = float(dd_data["Drawdown (%)"].iloc[-1]) if len(dd_data) > 0 else 0
    max_dd_val = float(dd_data["Drawdown (%)"].max()) if len(dd_data) > 0 else 0
    max_dur = int(dd_data["Duration (Days)"].max()) if len(dd_data) > 0 else 0
    dd_col1.metric("Current Drawdown", f"{current_dd:.2f}%")
    dd_col2.metric("Max Drawdown", f"{max_dd_val:.2f}%")
    dd_col3.metric("Max Duration", f"{max_dur} days")

    fig_dd_track = go.Figure()
    fig_dd_track.add_trace(go.Scatter(
        x=dd_data.index, y=-dd_data["Drawdown (%)"].values,
        fill="tozeroy", fillcolor="rgba(255,107,107,0.3)",
        line=dict(color="#ff6b6b", width=1.5), name="Drawdown",
    ))
    fig_dd_track.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd_track, use_container_width=True)

    # Risk Limits
    st.markdown("<div class='section-header'>🚦 Risk Limit Status</div>", unsafe_allow_html=True)

    all_var = {**param_var, **hist_var, **mc_var}
    limit_report = limit_monitor.full_limit_report(all_var, weights, sectors)

    for limit_type, label in [
        ("var_limits", "VaR Limits"),
        ("exposure_limits", "Exposure Limits"),
        ("concentration_limits", "Concentration Limits"),
    ]:
        report_df = limit_report.get(limit_type, pd.DataFrame())
        if not report_df.empty:
            st.markdown(f"**{label}**")
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No {label.lower()} configured.")

    # Data quality
    with st.expander("🔍 Data Quality Report", expanded=False):
        dq1, dq2 = st.columns(2)
        with dq1:
            st.markdown("**Missing Data Coverage**")
            missing_report = pipeline["missing_data"]
            if not missing_report.empty:
                st.dataframe(missing_report, use_container_width=True, hide_index=True)
            else:
                st.success("No missing data issues.")
        with dq2:
            st.markdown("**Anomaly Detection**")
            anomalies = pipeline["anomalies"]
            if not anomalies.empty:
                st.warning(f"{len(anomalies)} anomalies detected")
                st.dataframe(anomalies, use_container_width=True, hide_index=True)
            else:
                st.success("No anomalous price moves detected.")


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #495670; font-size: 0.85rem;'>"
    f"Quantitative Risk & Performance Analytics Engine · "
    f"Data as of {prices.index[-1].strftime('%Y-%m-%d')} · "
    f"{len(prices)} trading days · {len(tickers)} assets · "
    f"{mc_cfg.get('n_simulations', 10000):,} MC simulations"
    f"</div>",
    unsafe_allow_html=True,
)
