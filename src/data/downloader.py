"""
downloader.py — Market Data Downloader
========================================
Business Purpose:
    Wraps the yfinance API to fetch historical adjusted close prices for a
    configurable list of tickers. Acts as the single entry point for all
    external market data dependencies.

Output Schema:
    pd.DataFrame with DatetimeIndex (trading days) and one column per ticker
    containing adjusted close prices (float64).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger: logging.Logger = logging.getLogger(__name__)


class MarketDataDownloader:
    """
    Fetches historical adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of equity ticker symbols (e.g., ['AAPL', 'MSFT']).
    lookback_years : int
        Number of years of historical data to retrieve.
    """

    def __init__(self, tickers: List[str], lookback_years: int = 3) -> None:
        if not tickers:
            raise ValueError("Ticker list must be non-empty.")
        if lookback_years <= 0:
            raise ValueError(f"lookback_years must be positive, got {lookback_years}.")

        self.tickers: List[str] = tickers
        self.lookback_years: int = lookback_years
        self._end_date: datetime = datetime.today()
        self._start_date: datetime = self._end_date - timedelta(days=lookback_years * 365)

    def fetch_prices(self) -> pd.DataFrame:
        """
        Download adjusted close prices for all configured tickers.

        Returns
        -------
        pd.DataFrame
            Columns = tickers, Index = DatetimeIndex of trading days,
            Values = adjusted close prices (float64).

        Raises
        ------
        RuntimeError
            If the yfinance download returns an empty frame (API failure,
            network issue, or all tickers invalid).
        """
        logger.info(
            "Fetching prices for %d tickers from %s to %s",
            len(self.tickers),
            self._start_date.strftime("%Y-%m-%d"),
            self._end_date.strftime("%Y-%m-%d"),
        )

        try:
            raw: pd.DataFrame = yf.download(
                tickers=self.tickers,
                start=self._start_date.strftime("%Y-%m-%d"),
                end=self._end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)
            raise RuntimeError(f"Market data download failed: {exc}") from exc

        # yfinance returns multi-level columns when >1 ticker is requested.
        # Extract the 'Close' level to get a flat DataFrame.
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                prices: pd.DataFrame = raw["Close"]
            else:
                prices = raw.droplevel(level=0, axis=1)
        else:
            prices = raw

        if prices.empty:
            raise RuntimeError(
                "Download returned an empty DataFrame. "
                "Verify tickers and network connectivity."
            )

        # Drop columns that are completely empty/NaN
        nan_cols = prices.columns[prices.isna().all()]
        if len(nan_cols) > 0:
            logger.warning("Dropping tickers that returned all NaNs: %s", list(nan_cols))
            prices = prices.drop(columns=nan_cols)

        # Ensure column order matches the configured ticker list
        available: List[str] = [t for t in self.tickers if t in prices.columns]
        missing: List[str] = [t for t in self.tickers if t not in prices.columns]

        if missing:
            logger.warning("Tickers not found in download response: %s", missing)

        if not available:
            raise RuntimeError(
                f"None of the requested tickers {self.tickers} returned data."
            )

        prices = prices[available]
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "Date"

        logger.info(
            "Successfully fetched %d rows × %d tickers", len(prices), len(available)
        )
        return prices
