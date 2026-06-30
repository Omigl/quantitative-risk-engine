"""
exposure.py — Portfolio Exposure & Concentration Analysis
===========================================================
Business Purpose:
    Analyzes portfolio concentration risk across single-name, sector,
    and aggregate dimensions. Computes the Herfindahl-Hirschman Index
    (HHI) and marginal contribution to risk (MCTR) per asset.

Mathematical Formulas:
    HHI:                HHI = Σ(w_i²)         ∈ [1/N, 1]
    Top-N Weight:       Σ top N weights
    Sector Weight:      Σ w_i for i ∈ sector
    MCTR:               MCTR_i = (Σ × w)_i / σ_p   (marginal contrib to portfolio vol)

Output Schema:
    pd.DataFrame summaries for single-name, sector, and concentration reports.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class ExposureAnalyzer:
    """
    Analyzes portfolio exposure and concentration across multiple dimensions.

    Parameters
    ----------
    sectors : Dict[str, str]
        Mapping of ticker → sector name.
    """

    def __init__(self, sectors: Optional[Dict[str, str]] = None) -> None:
        self.sectors: Dict[str, str] = sectors or {}

    def single_name_exposure(self, weights: pd.Series) -> pd.DataFrame:
        """
        Compute per-asset exposure (weight) sorted by concentration.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.

        Returns
        -------
        pd.DataFrame
            Columns: ['Ticker', 'Weight (%)', 'Cumulative Weight (%)'].
            Sorted descending by weight.
        """
        sorted_weights: pd.Series = weights.sort_values(ascending=False)
        cumulative: np.ndarray = np.cumsum(sorted_weights.values) * 100

        report: pd.DataFrame = pd.DataFrame(
            {
                "Ticker": sorted_weights.index,
                "Weight (%)": np.round(sorted_weights.values * 100, 2),
                "Cumulative Weight (%)": np.round(cumulative, 2),
            }
        )
        report = report.reset_index(drop=True)

        logger.info(
            "Single-name exposure: top holding=%s (%.2f%%)",
            sorted_weights.index[0],
            sorted_weights.iloc[0] * 100,
        )
        return report

    def sector_exposure(self, weights: pd.Series) -> pd.DataFrame:
        """
        Aggregate portfolio weights by sector.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.

        Returns
        -------
        pd.DataFrame
            Columns: ['Sector', 'Weight (%)', 'Num Holdings'].
            Sorted descending by weight.
        """
        sector_map: pd.Series = pd.Series(
            {t: self.sectors.get(t, "Unclassified") for t in weights.index}
        )

        # Vectorized groupby aggregation
        sector_weights: pd.Series = weights.groupby(sector_map).sum()
        sector_counts: pd.Series = weights.groupby(sector_map).count()

        report: pd.DataFrame = pd.DataFrame(
            {
                "Sector": sector_weights.index,
                "Weight (%)": np.round(sector_weights.values * 100, 2),
                "Num Holdings": sector_counts.values.astype(int),
            }
        )
        report = report.sort_values("Weight (%)", ascending=False).reset_index(
            drop=True
        )

        logger.info(
            "Sector exposure: %d sectors, top=%s (%.2f%%)",
            len(report),
            report.iloc[0]["Sector"],
            report.iloc[0]["Weight (%)"],
        )
        return report

    def herfindahl_hirschman_index(self, weights: pd.Series) -> float:
        """
        Compute the Herfindahl-Hirschman Index (HHI) for concentration.

        Formula:
            HHI = Σ(w_i²)

        Interpretation:
            - HHI = 1/N → perfectly diversified (equal weight)
            - HHI = 1.0 → fully concentrated in one asset
            - HHI > 0.25 → considered highly concentrated

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights.

        Returns
        -------
        float
            HHI value in [1/N, 1].
        """
        hhi: float = float(np.sum(weights.values ** 2))
        n: int = len(weights)
        min_hhi: float = 1.0 / n if n > 0 else 0.0

        logger.info(
            "HHI=%.4f (min possible=%.4f for %d assets, %s concentrated)",
            hhi,
            min_hhi,
            n,
            "highly" if hhi > 0.25 else "moderately" if hhi > 0.15 else "well",
        )
        return hhi

    def top_n_concentration(self, weights: pd.Series, n: int = 3) -> float:
        """
        Compute the combined weight of the top-N holdings.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights.
        n : int
            Number of top holdings to aggregate.

        Returns
        -------
        float
            Combined weight of top-N holdings (fraction, e.g., 0.65 = 65%).
        """
        sorted_weights: pd.Series = weights.sort_values(ascending=False)
        top_n_weight: float = float(sorted_weights.iloc[:n].sum())

        logger.info(
            "Top-%d concentration: %.2f%% (%s)",
            n,
            top_n_weight * 100,
            sorted_weights.index[:n].tolist(),
        )
        return top_n_weight

    def marginal_contribution_to_risk(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute Marginal Contribution to Risk (MCTR) per asset.

        Formula:
            σ_p = √(w' Σ w)
            MCTR_i = (Σ × w)_i / σ_p
            Component Risk_i = w_i × MCTR_i
            % Contribution_i = Component Risk_i / σ_p

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        cov_matrix : pd.DataFrame
            Covariance matrix (assets × assets).

        Returns
        -------
        pd.DataFrame
            Per-asset risk decomposition with columns:
            ['Ticker', 'Weight (%)', 'MCTR', 'Component Risk', 'Risk Contribution (%)'].
        """
        # Align weights and covariance
        aligned_w: pd.Series = weights.reindex(cov_matrix.columns).fillna(0)
        w: np.ndarray = aligned_w.values

        # Portfolio variance and std dev
        port_var: float = float(w @ cov_matrix.values @ w)
        port_std: float = np.sqrt(port_var)

        if port_std == 0:
            logger.warning("Portfolio std dev is zero. MCTR undefined.")
            return pd.DataFrame()

        # Marginal contribution: (Σ × w) / σ_p
        marginal: np.ndarray = (cov_matrix.values @ w) / port_std

        # Component risk: w_i × MCTR_i
        component: np.ndarray = w * marginal

        # Percentage contribution
        pct_contrib: np.ndarray = component / port_std * 100

        report: pd.DataFrame = pd.DataFrame(
            {
                "Ticker": aligned_w.index,
                "Weight (%)": np.round(w * 100, 2),
                "MCTR": np.round(marginal, 6),
                "Component Risk": np.round(component, 6),
                "Risk Contribution (%)": np.round(pct_contrib, 2),
            }
        )
        report = report.sort_values("Risk Contribution (%)", ascending=False)
        report = report.reset_index(drop=True)

        logger.info(
            "MCTR computed: portfolio σ=%.6f, top risk contributor=%s (%.2f%%)",
            port_std,
            report.iloc[0]["Ticker"],
            report.iloc[0]["Risk Contribution (%)"],
        )
        return report

    def full_exposure_report(
        self,
        weights: pd.Series,
        cov_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, object]:
        """
        Generate a consolidated exposure and concentration report.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights.
        cov_matrix : Optional[pd.DataFrame]
            Covariance matrix (needed for MCTR). If None, MCTR is skipped.

        Returns
        -------
        Dict[str, object]
            Keys: 'single_name', 'sector', 'hhi', 'top3_concentration', 'mctr'.
        """
        report: Dict[str, object] = {
            "single_name": self.single_name_exposure(weights),
            "sector": self.sector_exposure(weights),
            "hhi": self.herfindahl_hirschman_index(weights),
            "top3_concentration": self.top_n_concentration(weights, n=3),
        }

        if cov_matrix is not None:
            report["mctr"] = self.marginal_contribution_to_risk(weights, cov_matrix)
        else:
            report["mctr"] = pd.DataFrame()

        logger.info("Full exposure report generated.")
        return report
