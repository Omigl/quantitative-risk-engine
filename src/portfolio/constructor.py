"""
constructor.py — Portfolio Construction Engine
================================================
Business Purpose:
    Builds a portfolio from raw holdings (tickers + shares) and live market
    prices. Computes market-value weights, total portfolio value, and
    per-asset market values.

Mathematical Formulas:
    Market Value:       MV_i = shares_i × P_i
    Portfolio Value:    V = Σ MV_i
    Weight:             w_i = MV_i / V
    Portfolio Return:   R_p,t = Σ(w_i × r_i,t)

Output Schema:
    PortfolioSnapshot dataclass with weights (pd.Series), market_values
    (pd.Series), total_value (float), and portfolio return series (pd.Series).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """
    Immutable snapshot of portfolio state at a point in time.

    Attributes
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker. Sums to 1.0.
    market_values : pd.Series
        Dollar market value per asset.
    total_value : float
        Total portfolio market value.
    shares : pd.Series
        Number of shares held per asset.
    """

    weights: pd.Series
    market_values: pd.Series
    total_value: float
    shares: pd.Series


class PortfolioConstructor:
    """
    Constructs a portfolio from holdings and market prices.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols in the portfolio.
    shares : Dict[str, int]
        Number of shares held per ticker.
    sectors : Optional[Dict[str, str]]
        Sector classification per ticker (for exposure analysis).
    """

    def __init__(
        self,
        tickers: List[str],
        shares: Dict[str, int],
        sectors: Optional[Dict[str, str]] = None,
    ) -> None:
        if not tickers:
            raise ValueError("Ticker list must be non-empty.")
        if not shares:
            raise ValueError("Shares dictionary must be non-empty.")

        self.tickers: List[str] = tickers
        self.shares_dict: Dict[str, int] = shares
        self.sectors: Dict[str, str] = sectors or {}

        self.shares_series: pd.Series = pd.Series(
            {t: shares.get(t, 0) for t in tickers}, dtype=np.float64
        )

    def build_snapshot(self, prices: pd.DataFrame) -> PortfolioSnapshot:
        """
        Build a portfolio snapshot from the latest available prices.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (DatetimeIndex × tickers).

        Returns
        -------
        PortfolioSnapshot
            Frozen snapshot with weights, market values, and total value.

        Raises
        ------
        ValueError
            If total portfolio value is zero or negative.
        """
        latest_prices: pd.Series = prices.iloc[-1]

        # Vectorized market value computation
        available_tickers: List[str] = [
            t for t in self.tickers if t in latest_prices.index
        ]
        shares: pd.Series = self.shares_series.reindex(available_tickers).fillna(0)
        asset_prices: pd.Series = latest_prices.reindex(available_tickers).fillna(0)

        market_values: pd.Series = shares * asset_prices
        total_value: float = float(market_values.sum())

        if total_value <= 0:
            raise ValueError(
                f"Total portfolio value is ${total_value:.2f}. "
                "Check holdings and price data."
            )

        weights: pd.Series = market_values / total_value

        logger.info(
            "Portfolio snapshot: $%.2f total value, %d assets, "
            "max weight=%.2f%% (%s)",
            total_value,
            len(available_tickers),
            weights.max() * 100,
            weights.idxmax(),
        )

        return PortfolioSnapshot(
            weights=weights,
            market_values=market_values,
            total_value=total_value,
            shares=shares,
        )

    def compute_portfolio_returns(
        self, returns: pd.DataFrame, weights: pd.Series
    ) -> pd.Series:
        """
        Compute the weighted portfolio return series.

        Formula:
            R_p,t = Σ(w_i × r_i,t)

        Vectorized via matrix dot product: returns @ weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series (observations × assets).
        weights : pd.Series
            Portfolio weights indexed by ticker.

        Returns
        -------
        pd.Series
            Portfolio return series indexed by date.
        """
        aligned_weights: pd.Series = weights.reindex(returns.columns).fillna(0.0)
        port_returns: pd.Series = returns.dot(aligned_weights)
        port_returns.name = "PortfolioReturn"

        logger.info(
            "Portfolio returns: %d obs, weight sum=%.4f, "
            "mean daily=%.6f, std daily=%.6f",
            len(port_returns),
            aligned_weights.sum(),
            port_returns.mean(),
            port_returns.std(),
        )
        return port_returns

    def compute_historical_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-varying portfolio weights across the full history.

        Useful for tracking weight drift over time.

        Parameters
        ----------
        prices : pd.DataFrame
            Full adjusted close price history.

        Returns
        -------
        pd.DataFrame
            Weights over time (DatetimeIndex × tickers).
        """
        shares: pd.Series = self.shares_series.reindex(prices.columns).fillna(0)

        # Vectorized: broadcast shares across all dates
        market_values: pd.DataFrame = prices.multiply(shares, axis=1)
        total_values: pd.Series = market_values.sum(axis=1)

        # Avoid division by zero
        weights: pd.DataFrame = market_values.div(total_values, axis=0)
        weights = weights.fillna(0.0)

        logger.info("Historical weights computed: shape %s", weights.shape)
        return weights

    def get_sector_mapping(self) -> pd.Series:
        """
        Return the sector classification for each ticker.

        Returns
        -------
        pd.Series
            Sector per ticker. Tickers without mapping are labeled 'Unclassified'.
        """
        return pd.Series(
            {t: self.sectors.get(t, "Unclassified") for t in self.tickers}
        )
