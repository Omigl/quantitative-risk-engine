"""
monitoring.py — Rolling Risk Monitoring
==========================================
Business Purpose:
    Tracks portfolio risk metrics over time using rolling windows. Enables
    early detection of regime changes, volatility spikes, and drawdown
    deterioration through time-series monitoring.

Mathematical Formulas:
    Rolling Volatility:     σ_t = std(R_p[t-W:t]) × √252
    Rolling VaR:            VaR_t = -Percentile(R_p[t-W:t], (1-α)×100) × V
    Rolling ES:             ES_t  = -Mean(R_p[t-W:t] | R ≤ VaR) × V
    Drawdown:               DD_t  = (Peak_t - Wealth_t) / Peak_t
    Drawdown Duration:      Number of consecutive days in drawdown

Output Schema:
    pd.DataFrame time series for each rolling metric.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class RiskMonitor:
    """
    Computes rolling risk metrics for portfolio monitoring.

    Parameters
    ----------
    rolling_window : int
        Default rolling window size in trading days.
    confidence_levels : List[float]
        VaR/ES confidence levels.
    trading_days_per_year : int
        Annualization factor.
    """

    def __init__(
        self,
        rolling_window: int = 63,
        confidence_levels: Optional[List[float]] = None,
        trading_days_per_year: int = 252,
    ) -> None:
        if rolling_window < 2:
            raise ValueError(
                f"rolling_window must be >= 2, got {rolling_window}."
            )
        self.window: int = rolling_window
        self.confidence_levels: List[float] = confidence_levels or [0.95, 0.99]
        self.trading_days: int = trading_days_per_year

    def rolling_volatility(
        self,
        portfolio_returns: pd.Series,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute rolling annualized volatility.

        Formula:
            σ_t = std(R_p[t-W:t]) × √252

        Fully vectorized via ``pd.Series.rolling().std()``.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio return series.
        window : Optional[int]
            Override rolling window size. Defaults to self.window.

        Returns
        -------
        pd.Series
            Rolling annualized volatility indexed by date.
        """
        w: int = window or self.window
        rolling_vol: pd.Series = (
            portfolio_returns.rolling(window=w, min_periods=max(w // 2, 2)).std()
            * np.sqrt(self.trading_days)
        )
        rolling_vol.name = f"Rolling_Vol_{w}d"

        logger.info(
            "Rolling volatility (window=%d): computed %d observations",
            w, rolling_vol.dropna().shape[0],
        )
        return rolling_vol

    def rolling_var(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: float,
        window: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """
        Compute rolling Historical VaR for each confidence level.

        Formula:
            VaR_t = -Percentile(R_p[t-W:t], (1-α)×100) × V

        Uses ``pd.Series.rolling().quantile()`` for vectorized computation.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio return series.
        portfolio_value : float
            Current total portfolio market value.
        window : Optional[int]
            Override rolling window size.

        Returns
        -------
        Dict[str, pd.Series]
            Keys: 'Rolling_VaR_95', 'Rolling_VaR_99', etc.
        """
        w: int = window or self.window
        results: Dict[str, pd.Series] = {}

        for cl in self.confidence_levels:
            quantile: float = 1 - cl
            rolling_quantile: pd.Series = portfolio_returns.rolling(
                window=w, min_periods=max(w // 2, 2)
            ).quantile(quantile)

            rolling_var_dollar: pd.Series = rolling_quantile.abs() * portfolio_value
            key: str = f"Rolling_VaR_{int(cl * 100)}"
            rolling_var_dollar.name = key
            results[key] = rolling_var_dollar

            logger.info(
                "%s (window=%d): computed %d observations",
                key, w, rolling_var_dollar.dropna().shape[0],
            )

        return results

    def rolling_es(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: float,
        window: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """
        Compute rolling Historical Expected Shortfall for each confidence level.

        For each window, computes the mean of returns at or below the VaR threshold.

        Formula:
            ES_t = -Mean(R_p[t-W:t] | R ≤ VaR_threshold) × V

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio return series.
        portfolio_value : float
            Current total portfolio market value.
        window : Optional[int]
            Override rolling window size.

        Returns
        -------
        Dict[str, pd.Series]
            Keys: 'Rolling_ES_95', 'Rolling_ES_99', etc.
        """
        w: int = window or self.window
        results: Dict[str, pd.Series] = {}

        for cl in self.confidence_levels:
            quantile: float = 1 - cl

            def _es_calc(x: np.ndarray) -> float:
                """Compute ES for a single window."""
                if len(x) < 2:
                    return np.nan
                threshold: float = float(np.percentile(x, quantile * 100))
                tail: np.ndarray = x[x <= threshold]
                if len(tail) == 0:
                    return 0.0
                return abs(float(np.mean(tail))) * portfolio_value

            rolling_es_series: pd.Series = portfolio_returns.rolling(
                window=w, min_periods=max(w // 2, 2)
            ).apply(_es_calc, raw=True)

            key: str = f"Rolling_ES_{int(cl * 100)}"
            rolling_es_series.name = key
            results[key] = rolling_es_series

            logger.info(
                "%s (window=%d): computed %d observations",
                key, w, rolling_es_series.dropna().shape[0],
            )

        return results

    def drawdown_tracking(
        self, portfolio_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Compute full drawdown tracking with depth and duration metrics.

        Formula:
            Wealth_t = ∏(1 + R_p,i)
            Peak_t = max(Wealth[1:t])
            DD_t = (Peak_t - Wealth_t) / Peak_t

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio return series.

        Returns
        -------
        pd.DataFrame
            Columns: ['Wealth', 'Peak', 'Drawdown', 'DrawdownPct',
            'Duration_Days']. Indexed by date.
        """
        wealth: pd.Series = (1 + portfolio_returns).cumprod()
        peak: pd.Series = wealth.cummax()
        drawdown: pd.Series = (peak - wealth) / peak
        drawdown_pct: pd.Series = drawdown * 100

        # Compute drawdown duration: consecutive days in drawdown
        in_drawdown: pd.Series = (drawdown > 0).astype(int)
        # Vectorized duration: cumulative sum that resets at recovery
        duration: pd.Series = in_drawdown.groupby(
            (in_drawdown != in_drawdown.shift()).cumsum()
        ).cumsum()

        result: pd.DataFrame = pd.DataFrame(
            {
                "Wealth": wealth,
                "Peak": peak,
                "Drawdown": drawdown,
                "Drawdown (%)": drawdown_pct,
                "Duration (Days)": duration,
            }
        )

        current_dd: float = float(drawdown.iloc[-1] * 100) if len(drawdown) > 0 else 0
        max_dd: float = float(drawdown.max() * 100) if len(drawdown) > 0 else 0
        max_dur: int = int(duration.max()) if len(duration) > 0 else 0

        logger.info(
            "Drawdown tracking: current=%.2f%%, max=%.2f%%, "
            "max duration=%d days",
            current_dd, max_dd, max_dur,
        )
        return result

    def multi_window_volatility(
        self,
        portfolio_returns: pd.Series,
        windows: List[int],
    ) -> pd.DataFrame:
        """
        Compute rolling volatility across multiple window sizes for comparison.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio return series.
        windows : List[int]
            List of rolling window sizes (e.g., [21, 63, 126]).

        Returns
        -------
        pd.DataFrame
            Columns = one per window size, Index = dates.
        """
        vol_dict: Dict[str, pd.Series] = {}
        for w in windows:
            vol: pd.Series = self.rolling_volatility(portfolio_returns, window=w)
            vol_dict[f"{w}d"] = vol

        result: pd.DataFrame = pd.DataFrame(vol_dict)
        logger.info(
            "Multi-window volatility: %d windows, %d observations",
            len(windows), len(result),
        )
        return result

    def full_monitoring_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: float,
        windows: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """
        Generate a consolidated rolling risk monitoring report.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio return series.
        portfolio_value : float
            Current total portfolio market value.
        windows : Optional[List[int]]
            Multiple rolling windows for comparison.

        Returns
        -------
        Dict[str, object]
            Keys: 'rolling_volatility', 'rolling_var', 'rolling_es',
            'drawdown', 'multi_window_vol'.
        """
        report: Dict[str, object] = {
            "rolling_volatility": self.rolling_volatility(portfolio_returns),
            "rolling_var": self.rolling_var(portfolio_returns, portfolio_value),
            "rolling_es": self.rolling_es(portfolio_returns, portfolio_value),
            "drawdown": self.drawdown_tracking(portfolio_returns),
        }

        if windows:
            report["multi_window_vol"] = self.multi_window_volatility(
                portfolio_returns, windows
            )

        logger.info("Full monitoring report generated.")
        return report
