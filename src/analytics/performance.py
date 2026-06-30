"""
performance.py — Portfolio Performance Analytics
==================================================
Business Purpose:
    Computes key portfolio performance metrics from return series. All
    calculations are vectorized using NumPy/Pandas — no Python loops
    in the critical path.

Metrics & Formulas:
    Cumulative Return:        CR_T = ∏(1 + r_t) - 1
    Annualized Return:        AR = (1 + CR_T)^(252 / N) - 1
    Annualized Volatility:    σ_ann = σ_daily × √252
    Sharpe Ratio:             SR = (AR - R_f) / σ_ann
    Max Drawdown:             MDD = max_t [ (Peak_t - Value_t) / Peak_t ]

Output Schema:
    Dict[str, float] per asset, or pd.DataFrame summary across all assets.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Computes portfolio-level and per-asset performance analytics.

    Parameters
    ----------
    risk_free_rate : float
        Annualized risk-free rate (e.g., 0.0525 for 5.25%).
    trading_days_per_year : int
        Number of trading days used for annualization (standard: 252).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0525,
        trading_days_per_year: int = 252,
    ) -> None:
        if trading_days_per_year <= 0:
            raise ValueError(
                f"trading_days_per_year must be positive, got {trading_days_per_year}."
            )
        self.risk_free_rate: float = risk_free_rate
        self.trading_days: int = trading_days_per_year

    def cumulative_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the cumulative return series for each asset.

        Formula:
            CR_t = ∏_{i=1}^{t} (1 + r_i) - 1

        Vectorized via ``pd.DataFrame.cumprod()``.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series (simple returns preferred for compounding).

        Returns
        -------
        pd.DataFrame
            Cumulative return series with the same shape and index.
        """
        cum: pd.DataFrame = (1 + returns).cumprod() - 1
        logger.info("Cumulative returns computed: shape %s", cum.shape)
        return cum

    def annualized_return(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute the annualized return for each asset.

        Formula:
            AR = (1 + CR_T)^(252 / N) - 1

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.Series
            Annualized return per asset.
        """
        n_days: int = len(returns)
        total_return: pd.Series = (1 + returns).prod() - 1
        ann_return: pd.Series = (1 + total_return) ** (self.trading_days / n_days) - 1

        logger.info("Annualized returns computed for %d assets.", len(ann_return))
        return ann_return

    def annualized_volatility(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute annualized volatility for each asset.

        Formula:
            σ_ann = σ_daily × √252

        Vectorized via ``pd.DataFrame.std()`` and scalar multiplication.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.Series
            Annualized volatility per asset.
        """
        ann_vol: pd.Series = returns.std() * np.sqrt(self.trading_days)
        logger.info("Annualized volatility computed for %d assets.", len(ann_vol))
        return ann_vol

    def sharpe_ratio(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute the annualized Sharpe ratio for each asset.

        Formula:
            SR = (AR - R_f) / σ_ann

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.Series
            Sharpe ratio per asset.
        """
        ann_ret: pd.Series = self.annualized_return(returns)
        ann_vol: pd.Series = self.annualized_volatility(returns)

        # Guard against division by zero for zero-volatility assets
        sharpe: pd.Series = pd.Series(
            np.where(
                ann_vol > 0,
                (ann_ret - self.risk_free_rate) / ann_vol,
                0.0,
            ),
            index=returns.columns,
        )
        logger.info("Sharpe ratios computed (Rf=%.4f).", self.risk_free_rate)
        return sharpe

    def max_drawdown(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute the maximum drawdown for each asset.

        Formula:
            MDD = max_t [ (RunningMax_t - CumWealth_t) / RunningMax_t ]

        Vectorized via ``pd.DataFrame.cummax()`` for the running peak.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.Series
            Maximum drawdown per asset (expressed as a positive fraction,
            e.g., 0.25 = 25% drawdown).
        """
        cumulative_wealth: pd.DataFrame = (1 + returns).cumprod()
        running_max: pd.DataFrame = cumulative_wealth.cummax()

        # Vectorized drawdown computation
        drawdown: pd.DataFrame = (running_max - cumulative_wealth) / running_max
        mdd: pd.Series = drawdown.max()

        logger.info("Max drawdown computed for %d assets.", len(mdd))
        return mdd

    def drawdown_series(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full drawdown time series for visualization.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.DataFrame
            Drawdown fraction over time (positive values = drawdown depth).
        """
        cumulative_wealth: pd.DataFrame = (1 + returns).cumprod()
        running_max: pd.DataFrame = cumulative_wealth.cummax()
        drawdown: pd.DataFrame = (running_max - cumulative_wealth) / running_max
        return drawdown

    def summary(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a consolidated performance summary table.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.DataFrame
            Summary table with rows = assets, columns = metrics:
            [AnnualizedReturn, AnnualizedVolatility, SharpeRatio, MaxDrawdown].
        """
        summary_df: pd.DataFrame = pd.DataFrame(
            {
                "Annualized Return (%)": np.round(
                    self.annualized_return(returns) * 100, 2
                ),
                "Annualized Volatility (%)": np.round(
                    self.annualized_volatility(returns) * 100, 2
                ),
                "Sharpe Ratio": np.round(self.sharpe_ratio(returns), 3),
                "Max Drawdown (%)": np.round(self.max_drawdown(returns) * 100, 2),
            }
        )
        summary_df.index.name = "Ticker"
        logger.info("Performance summary generated for %d assets.", len(summary_df))
        return summary_df
