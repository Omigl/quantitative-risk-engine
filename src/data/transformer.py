"""
transformer.py — Return Computation & Data Treatment
======================================================
Business Purpose:
    Converts raw adjusted close prices into analytically useful return series.
    Handles missing data via forward-fill with a configurable limit before
    computing returns.

Mathematical Formulas:
    Simple Return:  r_t = (P_t - P_{t-1}) / P_{t-1}
    Log Return:     r_t = ln(P_t / P_{t-1})

Output Schema:
    pd.DataFrame with DatetimeIndex and one column per ticker, containing
    either simple or log returns (float64). First row is NaN (no prior price).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class ReturnTransformer:
    """
    Transforms raw price data into return series after missing-data treatment.

    Parameters
    ----------
    forward_fill_limit : int
        Maximum number of consecutive NaN values to forward-fill.
        Beyond this limit, gaps remain as NaN to avoid propagating stale data.
    """

    def __init__(self, forward_fill_limit: int = 5) -> None:
        if forward_fill_limit < 0:
            raise ValueError(
                f"forward_fill_limit must be non-negative, got {forward_fill_limit}."
            )
        self.forward_fill_limit: int = forward_fill_limit

    def fill_missing(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing price observations with a bounded limit.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw adjusted close prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Prices with NaN gaps forward-filled up to ``forward_fill_limit``
            consecutive observations.
        """
        missing_before: int = int(prices.isna().sum().sum())
        filled: pd.DataFrame = prices.ffill(limit=self.forward_fill_limit)
        missing_after: int = int(filled.isna().sum().sum())

        logger.info(
            "Forward-fill: %d NaNs before → %d NaNs after (limit=%d)",
            missing_before,
            missing_after,
            self.forward_fill_limit,
        )
        return filled

    def compute_simple_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple (arithmetic) returns from price levels.

        Formula:
            r_t = (P_t - P_{t-1}) / P_{t-1}

        Vectorized via ``pd.DataFrame.pct_change()``.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (forward-filled).

        Returns
        -------
        pd.DataFrame
            Simple returns. First row is NaN.
        """
        returns: pd.DataFrame = prices.pct_change()
        logger.info("Computed simple returns: shape %s", returns.shape)
        return returns

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute logarithmic (continuously compounded) returns from price levels.

        Formula:
            r_t = ln(P_t / P_{t-1})

        Vectorized via ``np.log()`` on the price ratio array.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (forward-filled).

        Returns
        -------
        pd.DataFrame
            Log returns. First row is NaN.
        """
        # Vectorized: element-wise log of the ratio P_t / P_{t-1}
        log_returns: pd.DataFrame = pd.DataFrame(
            data=np.log(prices.values[1:] / prices.values[:-1]),
            index=prices.index[1:],
            columns=prices.columns,
        )
        # Prepend a NaN row to maintain index alignment with the original prices
        nan_row: pd.DataFrame = pd.DataFrame(
            data=np.full((1, prices.shape[1]), np.nan),
            index=prices.index[:1],
            columns=prices.columns,
        )
        log_returns = pd.concat([nan_row, log_returns])
        logger.info("Computed log returns: shape %s", log_returns.shape)
        return log_returns

    def transform(
        self, prices: pd.DataFrame, method: str = "log"
    ) -> pd.DataFrame:
        """
        End-to-end pipeline: fill missing data → compute returns.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw adjusted close prices.
        method : str
            Return computation method — ``'simple'`` or ``'log'`` (default).

        Returns
        -------
        pd.DataFrame
            Return series with first row dropped (NaN from differencing).

        Raises
        ------
        ValueError
            If ``method`` is not one of ``'simple'`` or ``'log'``.
        """
        valid_methods = {"simple", "log"}
        if method not in valid_methods:
            raise ValueError(
                f"Unknown return method '{method}'. Must be one of {valid_methods}."
            )

        filled: pd.DataFrame = self.fill_missing(prices)

        if method == "simple":
            returns = self.compute_simple_returns(filled)
        else:
            returns = self.compute_log_returns(filled)

        # Drop the first NaN row resulting from differencing
        returns = returns.iloc[1:]
        logger.info(
            "Transform complete (method=%s): %d observations × %d assets",
            method,
            len(returns),
            returns.shape[1],
        )
        return returns
