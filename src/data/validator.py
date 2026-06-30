"""
validator.py — Data Quality Validation
========================================
Business Purpose:
    Enforces data integrity constraints on raw price data before it enters
    the analytics pipeline. Implements fail-fast assertions for corrupt data
    and generates structured anomaly reports.

Validation Rules:
    1. Non-negative pricing: All price values must be >= 0.
    2. Anomaly detection: Flag single-day moves exceeding a configurable threshold.
    3. Missing data coverage: Alert when a ticker's missing-data percentage
       exceeds a configurable limit.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates raw price DataFrames for quality and integrity.

    Parameters
    ----------
    anomaly_threshold_pct : float
        Maximum allowable single-day percentage move (absolute value).
        Moves exceeding this are flagged as anomalies.
    missing_data_alert_pct : float
        If a ticker has more than this percentage of missing data points,
        a warning alert is raised.
    """

    def __init__(
        self,
        anomaly_threshold_pct: float = 50.0,
        missing_data_alert_pct: float = 5.0,
    ) -> None:
        if anomaly_threshold_pct <= 0:
            raise ValueError(
                f"anomaly_threshold_pct must be positive, got {anomaly_threshold_pct}."
            )
        if missing_data_alert_pct < 0:
            raise ValueError(
                f"missing_data_alert_pct must be non-negative, got {missing_data_alert_pct}."
            )

        self.anomaly_threshold_pct: float = anomaly_threshold_pct
        self.missing_data_alert_pct: float = missing_data_alert_pct

    def assert_non_negative_prices(self, prices: pd.DataFrame) -> None:
        """
        Fail-fast assertion: all price values must be non-negative.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw adjusted close prices.

        Raises
        ------
        ValueError
            If any non-NaN price value is negative, indicating corrupt data.
        """
        # Vectorized check across the entire DataFrame
        negative_mask: pd.DataFrame = prices < 0
        if negative_mask.any().any():
            # Identify the offending tickers and dates for diagnostics
            offending: List[Tuple[str, str]] = []
            for col in prices.columns:
                bad_dates = prices.index[negative_mask[col]]
                for d in bad_dates:
                    offending.append((col, str(d.date())))

            detail: str = "; ".join(
                f"{ticker} on {date}" for ticker, date in offending[:10]
            )
            raise ValueError(
                f"Negative prices detected (corrupt data). "
                f"First occurrences: {detail}"
            )
        logger.info("✓ Non-negative price assertion passed.")

    def detect_anomalies(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Detect single-day price moves exceeding the anomaly threshold.

        Uses vectorized percentage change computation — no Python loops.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            DataFrame of flagged anomalies with columns:
            ['Ticker', 'Date', 'PreviousClose', 'Close', 'DailyMovePct'].
            Empty if no anomalies found.
        """
        # Vectorized daily percentage change
        daily_pct: pd.DataFrame = prices.pct_change().abs() * 100.0

        # Boolean mask for anomalies
        anomaly_mask: pd.DataFrame = daily_pct > self.anomaly_threshold_pct

        if not anomaly_mask.any().any():
            logger.info(
                "✓ No single-day anomalies detected (threshold=%.1f%%).",
                self.anomaly_threshold_pct,
            )
            return pd.DataFrame(
                columns=["Ticker", "Date", "PreviousClose", "Close", "DailyMovePct"]
            )

        # Extract anomaly records using vectorized masking
        records: List[Dict[str, object]] = []
        for col in prices.columns:
            col_anomalies = anomaly_mask[col]
            if col_anomalies.any():
                anomaly_dates = prices.index[col_anomalies]
                for date in anomaly_dates:
                    idx: int = prices.index.get_loc(date)
                    if idx == 0:
                        continue  # Skip first row (no previous close)
                    records.append(
                        {
                            "Ticker": col,
                            "Date": date,
                            "PreviousClose": float(prices[col].iloc[idx - 1]),
                            "Close": float(prices[col].iloc[idx]),
                            "DailyMovePct": float(daily_pct[col].iloc[idx]),
                        }
                    )

        anomalies: pd.DataFrame = pd.DataFrame(records)
        logger.warning(
            "⚠ %d anomalies detected (threshold=%.1f%%): %s",
            len(anomalies),
            self.anomaly_threshold_pct,
            anomalies["Ticker"].unique().tolist(),
        )
        return anomalies

    def report_missing_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-ticker missing data percentages and flag those exceeding
        the alert threshold.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw adjusted close prices.

        Returns
        -------
        pd.DataFrame
            Summary with columns: ['Ticker', 'MissingCount', 'TotalRows',
            'MissingPct', 'Alert']. Sorted by MissingPct descending.
        """
        total_rows: int = len(prices)
        # Vectorized missing count per column
        missing_counts: pd.Series = prices.isna().sum()

        report: pd.DataFrame = pd.DataFrame(
            {
                "Ticker": missing_counts.index,
                "MissingCount": missing_counts.values,
                "TotalRows": total_rows,
                "MissingPct": np.round(
                    (missing_counts.values / total_rows) * 100.0, 2
                ),
            }
        )
        report["Alert"] = report["MissingPct"] > self.missing_data_alert_pct
        report = report.sort_values("MissingPct", ascending=False).reset_index(
            drop=True
        )

        alerted: List[str] = report.loc[report["Alert"], "Ticker"].tolist()
        if alerted:
            logger.warning(
                "⚠ Tickers exceeding %.1f%% missing data: %s",
                self.missing_data_alert_pct,
                alerted,
            )
        else:
            logger.info(
                "✓ All tickers within missing data threshold (%.1f%%).",
                self.missing_data_alert_pct,
            )

        return report

    def validate(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run the full validation suite on a price DataFrame.

        Parameters
        ----------
        prices : pd.DataFrame
            Raw adjusted close prices.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Keys: 'anomalies' (anomaly report), 'missing_data' (coverage report).

        Raises
        ------
        ValueError
            If negative prices are detected (fail-fast).
        """
        logger.info("Running full data validation suite...")
        self.assert_non_negative_prices(prices)
        anomalies: pd.DataFrame = self.detect_anomalies(prices)
        missing_report: pd.DataFrame = self.report_missing_data(prices)

        return {"anomalies": anomalies, "missing_data": missing_report}
