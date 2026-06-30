"""
stress.py — Stress Testing Engine
====================================
Business Purpose:
    Evaluates portfolio resilience under adverse market conditions through
    three testing frameworks:
    1. Historical scenarios — replay actual crisis windows from market data
    2. Hypothetical scenarios — apply per-asset shocks from YAML config
    3. Custom scenario builder — accept runtime-defined shock vectors

Mathematical Formulas:
    Historical:     R_p = Σ(w_i × (P_end,i / P_start,i - 1))
    Hypothetical:   ΔV = Σ(MV_i × shock_i / 100)
    Custom:         Same as hypothetical with runtime-provided shocks

Output Schema:
    pd.DataFrame with columns: ['Scenario', 'Type', 'Portfolio Return (%)',
    'Portfolio Loss ($)', 'Per-Asset Impact'].
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class StressTestEngine:
    """
    Stress testing engine supporting historical replay, hypothetical shocks,
    and custom runtime scenarios.

    Parameters
    ----------
    portfolio_value : float
        Current total portfolio market value.
    """

    def __init__(self, portfolio_value: float) -> None:
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive, got {portfolio_value}."
            )
        self.portfolio_value: float = portfolio_value

    def run_historical_scenarios(
        self,
        prices: pd.DataFrame,
        weights: pd.Series,
        scenarios: Dict[str, Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Replay historical crisis windows and measure portfolio impact.

        For each scenario, computes the actual return over the crisis period
        using market data.

        Formula:
            r_i = P_end,i / P_start,i - 1
            R_p = Σ(w_i × r_i)

        Parameters
        ----------
        prices : pd.DataFrame
            Full adjusted close price history (DatetimeIndex × tickers).
        weights : pd.Series
            Portfolio weights.
        scenarios : Dict[str, Dict[str, str]]
            Historical scenario definitions with 'label', 'start_date', 'end_date'.

        Returns
        -------
        pd.DataFrame
            Results with columns: ['Scenario', 'Type', 'Start', 'End',
            'Portfolio Return (%)', 'Portfolio Loss ($)'].
        """
        records: List[Dict[str, object]] = []

        for scenario_id, params in scenarios.items():
            label: str = params.get("label", scenario_id)
            start: str = str(params.get("start_date", ""))
            end: str = str(params.get("end_date", ""))

            if not start or not end:
                logger.warning(
                    "Historical [%s]: missing start_date or end_date. Skipping.", label
                )
                continue

            try:
                window: pd.DataFrame = prices.loc[start:end]
            except KeyError:
                logger.warning(
                    "Historical [%s]: date range %s–%s not in data. Skipping.",
                    label, start, end,
                )
                continue

            if len(window) < 2:
                logger.warning(
                    "Historical [%s]: insufficient data (%d rows). Skipping.",
                    label, len(window),
                )
                continue

            # Vectorized period returns per asset
            period_returns: pd.Series = (window.iloc[-1] / window.iloc[0]) - 1

            # Portfolio weighted return
            aligned_w: pd.Series = weights.reindex(period_returns.index).fillna(0.0)
            port_return: float = float(period_returns.dot(aligned_w))
            loss_dollar: float = (
                abs(port_return) * self.portfolio_value if port_return < 0 else 0.0
            )

            records.append(
                {
                    "Scenario": label,
                    "Type": "Historical",
                    "Start": start,
                    "End": end,
                    "Portfolio Return (%)": round(port_return * 100, 2),
                    "Portfolio Loss ($)": round(loss_dollar, 2),
                }
            )
            logger.info(
                "Historical [%s]: %.2f%% → $%.2f loss",
                label, port_return * 100, loss_dollar,
            )

        result: pd.DataFrame = pd.DataFrame(records)
        logger.info(
            "Historical stress testing: %d scenarios evaluated.", len(result)
        )
        return result

    def run_hypothetical_scenarios(
        self,
        weights: pd.Series,
        market_values: pd.Series,
        scenarios: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Apply per-asset hypothetical shock scenarios.

        Each scenario defines a percentage shock for each asset. The portfolio
        impact is computed as the weighted sum of individual asset impacts.

        Formula:
            ΔV_i = MV_i × shock_i / 100
            ΔV_portfolio = Σ ΔV_i
            R_p = ΔV_portfolio / V_total

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        market_values : pd.Series
            Dollar market value per asset.
        scenarios : Dict[str, Dict[str, Any]]
            Hypothetical scenario definitions with 'label' and 'shocks' dict.

        Returns
        -------
        pd.DataFrame
            Results with columns: ['Scenario', 'Type', 'Portfolio Return (%)',
            'Portfolio Loss ($)'].
        """
        records: List[Dict[str, object]] = []

        for scenario_id, params in scenarios.items():
            label: str = params.get("label", scenario_id)
            shocks: Dict[str, float] = params.get("shocks", {})

            if not shocks:
                logger.warning(
                    "Hypothetical [%s]: no shocks defined. Skipping.", label
                )
                continue

            # Vectorized: compute dollar impact per asset
            shock_series: pd.Series = pd.Series(shocks).reindex(
                market_values.index
            ).fillna(0.0)
            dollar_impact: pd.Series = market_values * (shock_series / 100.0)
            total_impact: float = float(dollar_impact.sum())
            port_return: float = total_impact / self.portfolio_value

            loss_dollar: float = abs(total_impact) if total_impact < 0 else 0.0

            records.append(
                {
                    "Scenario": label,
                    "Type": "Hypothetical",
                    "Portfolio Return (%)": round(port_return * 100, 2),
                    "Portfolio Loss ($)": round(loss_dollar, 2),
                }
            )
            logger.info(
                "Hypothetical [%s]: %.2f%% → $%.2f impact",
                label, port_return * 100, total_impact,
            )

        result: pd.DataFrame = pd.DataFrame(records)
        logger.info(
            "Hypothetical stress testing: %d scenarios evaluated.", len(result)
        )
        return result

    def run_custom_scenario(
        self,
        market_values: pd.Series,
        shocks: Dict[str, float],
        label: str = "Custom Scenario",
    ) -> pd.DataFrame:
        """
        Apply a single custom shock scenario at runtime.

        Designed for the interactive scenario builder in the dashboard.

        Parameters
        ----------
        market_values : pd.Series
            Dollar market value per asset.
        shocks : Dict[str, float]
            Per-ticker shock percentages (e.g., {'AAPL': -15.0, 'MSFT': -10.0}).
        label : str
            Human-readable scenario label.

        Returns
        -------
        pd.DataFrame
            Single-row result with columns: ['Scenario', 'Type',
            'Portfolio Return (%)', 'Portfolio Loss ($)'] plus per-asset columns.
        """
        shock_series: pd.Series = pd.Series(shocks).reindex(
            market_values.index
        ).fillna(0.0)
        dollar_impact: pd.Series = market_values * (shock_series / 100.0)
        total_impact: float = float(dollar_impact.sum())
        port_return: float = total_impact / self.portfolio_value
        loss_dollar: float = abs(total_impact) if total_impact < 0 else 0.0

        # Build result with per-asset breakdown
        result_dict: Dict[str, object] = {
            "Scenario": label,
            "Type": "Custom",
            "Portfolio Return (%)": round(port_return * 100, 2),
            "Portfolio Loss ($)": round(loss_dollar, 2),
        }

        # Add per-asset impact columns
        for ticker in market_values.index:
            result_dict[f"{ticker} Impact ($)"] = round(
                float(dollar_impact.get(ticker, 0.0)), 2
            )

        result: pd.DataFrame = pd.DataFrame([result_dict])

        logger.info(
            "Custom scenario [%s]: %.2f%% → $%.2f loss",
            label, port_return * 100, loss_dollar,
        )
        return result

    def per_asset_breakdown(
        self,
        market_values: pd.Series,
        shocks: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Compute a detailed per-asset impact breakdown for a shock scenario.

        Parameters
        ----------
        market_values : pd.Series
            Dollar market value per asset.
        shocks : Dict[str, float]
            Per-ticker shock percentages.

        Returns
        -------
        pd.DataFrame
            Columns: ['Ticker', 'Market Value ($)', 'Shock (%)',
            'Dollar Impact ($)', 'Post-Shock Value ($)'].
        """
        shock_series: pd.Series = pd.Series(shocks).reindex(
            market_values.index
        ).fillna(0.0)
        dollar_impact: pd.Series = market_values * (shock_series / 100.0)
        post_shock: pd.Series = market_values + dollar_impact

        report: pd.DataFrame = pd.DataFrame(
            {
                "Ticker": market_values.index,
                "Market Value ($)": np.round(market_values.values, 2),
                "Shock (%)": np.round(shock_series.values, 2),
                "Dollar Impact ($)": np.round(dollar_impact.values, 2),
                "Post-Shock Value ($)": np.round(post_shock.values, 2),
            }
        )
        return report.reset_index(drop=True)

    def run_all_scenarios(
        self,
        prices: pd.DataFrame,
        weights: pd.Series,
        market_values: pd.Series,
        historical_scenarios: Optional[Dict[str, Dict[str, str]]] = None,
        hypothetical_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Run all configured stress scenarios and combine results.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history.
        weights : pd.Series
            Portfolio weights.
        market_values : pd.Series
            Per-asset market values.
        historical_scenarios : Optional dict of historical scenarios.
        hypothetical_scenarios : Optional dict of hypothetical scenarios.

        Returns
        -------
        pd.DataFrame
            Combined results from all scenario types.
        """
        frames: List[pd.DataFrame] = []

        if historical_scenarios:
            hist_results = self.run_historical_scenarios(
                prices, weights, historical_scenarios
            )
            if not hist_results.empty:
                frames.append(hist_results)

        if hypothetical_scenarios:
            hypo_results = self.run_hypothetical_scenarios(
                weights, market_values, hypothetical_scenarios
            )
            if not hypo_results.empty:
                frames.append(hypo_results)

        if not frames:
            return pd.DataFrame()

        combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
        logger.info("All stress scenarios: %d total evaluated.", len(combined))
        return combined
