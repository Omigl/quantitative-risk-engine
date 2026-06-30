"""
limits.py — Risk Limit Framework
====================================
Business Purpose:
    Enforces portfolio risk constraints by comparing live risk metrics
    against configurable thresholds. Generates structured breach reports
    for VaR limits, exposure limits, and concentration limits.

Limit Categories:
    1. VaR Limits:          Current VaR vs. max allowable VaR
    2. Exposure Limits:     Single-name and sector weight caps
    3. Concentration Limits: HHI threshold and top-N holding caps

Output Schema:
    pd.DataFrame breach report with columns: ['Limit Type', 'Metric',
    'Current Value', 'Threshold', 'Status', 'Severity'].
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class RiskLimitMonitor:
    """
    Monitors portfolio risk metrics against configurable limits.

    Parameters
    ----------
    var_limits : Dict[str, float]
        Maximum allowable VaR values (e.g., {'VaR_95_max': 50000}).
    exposure_limits : Dict[str, float]
        Exposure constraints (e.g., {'max_single_name_weight': 0.25}).
    concentration_limits : Dict[str, float]
        Concentration constraints (e.g., {'max_hhi': 0.25}).
    """

    def __init__(
        self,
        var_limits: Optional[Dict[str, float]] = None,
        exposure_limits: Optional[Dict[str, float]] = None,
        concentration_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        self.var_limits: Dict[str, float] = var_limits or {}
        self.exposure_limits: Dict[str, float] = exposure_limits or {}
        self.concentration_limits: Dict[str, float] = concentration_limits or {}

    def check_var_limits(
        self,
        current_var: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Check current VaR metrics against configured dollar thresholds.

        Parameters
        ----------
        current_var : Dict[str, float]
            Current VaR values (e.g., {'Parametric_VaR_95': 45000}).

        Returns
        -------
        pd.DataFrame
            Breach report with columns: ['Limit Type', 'Metric',
            'Current ($)', 'Threshold ($)', 'Status', 'Utilization (%)'].
        """
        records: List[Dict[str, object]] = []

        # Map limit keys to VaR result keys
        limit_mapping: Dict[str, List[str]] = {
            "VaR_95_max": [
                "Parametric_VaR_95", "Historical_VaR_95", "MonteCarlo_VaR_95"
            ],
            "VaR_99_max": [
                "Parametric_VaR_99", "Historical_VaR_99", "MonteCarlo_VaR_99"
            ],
        }

        for limit_key, threshold in self.var_limits.items():
            var_keys: List[str] = limit_mapping.get(limit_key, [])

            for var_key in var_keys:
                if var_key not in current_var:
                    continue

                current: float = current_var[var_key]
                utilization: float = (current / threshold * 100) if threshold > 0 else 0
                breached: bool = current > threshold
                status: str = "🔴 BREACH" if breached else (
                    "🟡 WARNING" if utilization > 80 else "🟢 OK"
                )

                records.append(
                    {
                        "Limit Type": "VaR",
                        "Metric": var_key,
                        "Current ($)": round(current, 2),
                        "Threshold ($)": round(threshold, 2),
                        "Status": status,
                        "Utilization (%)": round(utilization, 2),
                    }
                )

                if breached:
                    logger.warning(
                        "VaR BREACH: %s=$%.2f > limit=$%.2f (%.1f%% utilized)",
                        var_key, current, threshold, utilization,
                    )

        result: pd.DataFrame = pd.DataFrame(records)
        breaches: int = len(result[result["Status"].str.contains("BREACH")])
        logger.info("VaR limits checked: %d metrics, %d breaches", len(result), breaches)
        return result

    def check_exposure_limits(
        self,
        weights: pd.Series,
        sectors: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Check single-name and sector exposure against configured limits.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        sectors : Optional[Dict[str, str]]
            Ticker → sector mapping.

        Returns
        -------
        pd.DataFrame
            Breach report for exposure limits.
        """
        records: List[Dict[str, object]] = []

        # Single-name exposure check
        max_single_name: float = self.exposure_limits.get(
            "max_single_name_weight", 1.0
        )
        for ticker, weight in weights.items():
            w: float = float(weight)
            if w > max_single_name:
                records.append(
                    {
                        "Limit Type": "Single-Name Exposure",
                        "Metric": f"{ticker} Weight",
                        "Current (%)": round(w * 100, 2),
                        "Threshold (%)": round(max_single_name * 100, 2),
                        "Status": "🔴 BREACH",
                        "Utilization (%)": round(w / max_single_name * 100, 2),
                    }
                )
                logger.warning(
                    "Exposure BREACH: %s weight=%.2f%% > limit=%.2f%%",
                    ticker, w * 100, max_single_name * 100,
                )
            elif w > max_single_name * 0.8:
                records.append(
                    {
                        "Limit Type": "Single-Name Exposure",
                        "Metric": f"{ticker} Weight",
                        "Current (%)": round(w * 100, 2),
                        "Threshold (%)": round(max_single_name * 100, 2),
                        "Status": "🟡 WARNING",
                        "Utilization (%)": round(w / max_single_name * 100, 2),
                    }
                )

        # Sector exposure check
        if sectors:
            max_sector: float = self.exposure_limits.get("max_sector_weight", 1.0)
            sector_map: pd.Series = pd.Series(
                {t: sectors.get(t, "Unclassified") for t in weights.index}
            )
            sector_weights: pd.Series = weights.groupby(sector_map).sum()

            for sector, sw in sector_weights.items():
                s: float = float(sw)
                if s > max_sector:
                    records.append(
                        {
                            "Limit Type": "Sector Exposure",
                            "Metric": f"{sector} Weight",
                            "Current (%)": round(s * 100, 2),
                            "Threshold (%)": round(max_sector * 100, 2),
                            "Status": "🔴 BREACH",
                            "Utilization (%)": round(s / max_sector * 100, 2),
                        }
                    )
                    logger.warning(
                        "Sector BREACH: %s weight=%.2f%% > limit=%.2f%%",
                        sector, s * 100, max_sector * 100,
                    )
                elif s > max_sector * 0.8:
                    records.append(
                        {
                            "Limit Type": "Sector Exposure",
                            "Metric": f"{sector} Weight",
                            "Current (%)": round(s * 100, 2),
                            "Threshold (%)": round(max_sector * 100, 2),
                            "Status": "🟡 WARNING",
                            "Utilization (%)": round(s / max_sector * 100, 2),
                        }
                    )

        result: pd.DataFrame = pd.DataFrame(records)
        logger.info("Exposure limits checked: %d items flagged", len(result))
        return result

    def check_concentration_limits(
        self,
        weights: pd.Series,
    ) -> pd.DataFrame:
        """
        Check portfolio concentration against HHI and top-N thresholds.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.

        Returns
        -------
        pd.DataFrame
            Breach report for concentration limits.
        """
        records: List[Dict[str, object]] = []

        # HHI check
        max_hhi: float = self.concentration_limits.get("max_hhi", 1.0)
        hhi: float = float(np.sum(weights.values ** 2))
        hhi_util: float = (hhi / max_hhi * 100) if max_hhi > 0 else 0

        if hhi > max_hhi:
            status = "🔴 BREACH"
        elif hhi > max_hhi * 0.8:
            status = "🟡 WARNING"
        else:
            status = "🟢 OK"

        records.append(
            {
                "Limit Type": "Concentration",
                "Metric": "HHI",
                "Current": round(hhi, 4),
                "Threshold": round(max_hhi, 4),
                "Status": status,
                "Utilization (%)": round(hhi_util, 2),
            }
        )

        # Top-3 concentration check
        max_top3: float = self.concentration_limits.get("max_top3_weight", 1.0)
        sorted_w: pd.Series = weights.sort_values(ascending=False)
        top3_weight: float = float(sorted_w.iloc[:3].sum())
        top3_util: float = (top3_weight / max_top3 * 100) if max_top3 > 0 else 0

        if top3_weight > max_top3:
            status = "🔴 BREACH"
        elif top3_weight > max_top3 * 0.8:
            status = "🟡 WARNING"
        else:
            status = "🟢 OK"

        records.append(
            {
                "Limit Type": "Concentration",
                "Metric": f"Top-3 ({', '.join(sorted_w.index[:3].tolist())})",
                "Current": round(top3_weight, 4),
                "Threshold": round(max_top3, 4),
                "Status": status,
                "Utilization (%)": round(top3_util, 2),
            }
        )

        result: pd.DataFrame = pd.DataFrame(records)
        breaches: int = len(result[result["Status"].str.contains("BREACH")])
        logger.info(
            "Concentration limits: HHI=%.4f, Top3=%.2f%%, %d breaches",
            hhi, top3_weight * 100, breaches,
        )
        return result

    def full_limit_report(
        self,
        current_var: Dict[str, float],
        weights: pd.Series,
        sectors: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all limit checks and return a consolidated report.

        Parameters
        ----------
        current_var : Dict[str, float]
            Combined VaR metrics from all methods.
        weights : pd.Series
            Portfolio weights.
        sectors : Optional[Dict[str, str]]
            Ticker → sector mapping.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Keys: 'var_limits', 'exposure_limits', 'concentration_limits'.
        """
        report: Dict[str, pd.DataFrame] = {
            "var_limits": self.check_var_limits(current_var),
            "exposure_limits": self.check_exposure_limits(weights, sectors),
            "concentration_limits": self.check_concentration_limits(weights),
        }

        total_breaches: int = sum(
            len(df[df["Status"].str.contains("BREACH")])
            for df in report.values()
            if not df.empty and "Status" in df.columns
        )
        logger.info(
            "Full limit report: %d total breaches across all categories",
            total_breaches,
        )
        return report
