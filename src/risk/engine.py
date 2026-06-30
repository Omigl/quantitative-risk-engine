"""
engine.py — Portfolio Risk Engine
====================================
Business Purpose:
    Quantifies portfolio-level risk through covariance analysis, Value-at-Risk
    (Parametric, Historical, Monte Carlo), Expected Shortfall (Parametric,
    Historical, Monte Carlo), and portfolio return computation.

Mathematical Formulas:
    Portfolio Variance:      σ²_p = w' Σ w
    Parametric VaR:          VaR_α = z_α × σ_p × √t  (normal assumption)
    Historical VaR:          VaR_α = Percentile(portfolio_returns, 1 - α)
    Monte Carlo VaR:         VaR_α = Percentile(simulated_returns, 1 - α)
    Parametric ES:           ES_α  = σ_p × φ(z_α) / (1 - α)  (Gaussian closed-form)
    Historical ES:           ES_α  = E[R | R ≤ -VaR_α]  (mean of tail losses)
    Monte Carlo ES:          ES_α  = E[R_sim | R_sim ≤ -VaR_α]

Output Schema:
    Dict[str, float] for scalar risk metrics.
    pd.DataFrame for tabular results.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger: logging.Logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Portfolio risk quantification engine.

    Parameters
    ----------
    confidence_levels : List[float]
        VaR/CVaR confidence levels (e.g., [0.95, 0.99]).
    holding_period_days : int
        VaR holding period in trading days (default: 1).
    trading_days_per_year : int
        Annualization factor (default: 252).
    n_simulations : int
        Number of Monte Carlo simulations (default: 10000).
    random_seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        confidence_levels: Optional[List[float]] = None,
        holding_period_days: int = 1,
        trading_days_per_year: int = 252,
        n_simulations: int = 10000,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.confidence_levels: List[float] = confidence_levels or [0.95, 0.99]
        self.holding_period: int = holding_period_days
        self.trading_days: int = trading_days_per_year
        self.n_simulations: int = n_simulations
        self.random_seed: Optional[int] = random_seed

        # Validate confidence levels
        for cl in self.confidence_levels:
            if not 0 < cl < 1:
                raise ValueError(
                    f"Confidence level must be in (0, 1), got {cl}."
                )

    # ═══════════════════════════════════════════════════════════════════════
    # COVARIANCE & CORRELATION
    # ═══════════════════════════════════════════════════════════════════════

    def compute_covariance_matrix(
        self, returns: pd.DataFrame, annualize: bool = True
    ) -> pd.DataFrame:
        """
        Compute the variance-covariance matrix of asset returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series (observations × assets).
        annualize : bool
            If True, scale by the number of trading days per year.

        Returns
        -------
        pd.DataFrame
            Symmetric covariance matrix (assets × assets).
        """
        cov: pd.DataFrame = returns.cov()
        if annualize:
            cov = cov * self.trading_days
        logger.info(
            "Covariance matrix computed: %dx%d (annualized=%s)",
            cov.shape[0],
            cov.shape[1],
            annualize,
        )
        return cov

    def compute_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Pearson correlation matrix of asset returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        pd.DataFrame
            Symmetric correlation matrix (assets × assets), values in [-1, 1].
        """
        corr: pd.DataFrame = returns.corr()
        logger.info("Correlation matrix computed: %dx%d", corr.shape[0], corr.shape[1])
        return corr

    # ═══════════════════════════════════════════════════════════════════════
    # PORTFOLIO RETURNS
    # ═══════════════════════════════════════════════════════════════════════

    def compute_portfolio_returns(
        self, returns: pd.DataFrame, weights: pd.Series
    ) -> pd.Series:
        """
        Compute the weighted portfolio return series.

        Formula:
            R_p,t = Σ(w_i × r_i,t)

        Vectorized via matrix multiplication: returns @ weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series (observations × assets).
        weights : pd.Series
            Portfolio weights indexed by ticker. Must sum to ~1.0.

        Returns
        -------
        pd.Series
            Portfolio return series indexed by date.
        """
        aligned_weights: pd.Series = weights.reindex(returns.columns).fillna(0.0)
        port_returns: pd.Series = returns.dot(aligned_weights)
        port_returns.name = "PortfolioReturn"

        logger.info(
            "Portfolio returns computed: %d observations, effective weight sum=%.4f",
            len(port_returns),
            aligned_weights.sum(),
        )
        return port_returns

    # ═══════════════════════════════════════════════════════════════════════
    # VALUE-AT-RISK (VaR)
    # ═══════════════════════════════════════════════════════════════════════

    def parametric_var(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute Parametric (Variance-Covariance) Value-at-Risk.

        Assumption: Portfolio returns are normally distributed.

        Formula:
            σ_p = √(w' Σ w)
            VaR_α = z_α × σ_p × √t × V

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights (must sum to ~1.0).
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, float]
            Keys: 'Parametric_VaR_95', 'Parametric_VaR_99', etc.
        """
        cov_daily: pd.DataFrame = self.compute_covariance_matrix(
            returns, annualize=False
        )
        w: np.ndarray = weights.reindex(returns.columns).fillna(0.0).values

        port_variance: float = float(w @ cov_daily.values @ w)
        port_std_daily: float = np.sqrt(port_variance)
        port_std_horizon: float = port_std_daily * np.sqrt(self.holding_period)

        results: Dict[str, float] = {}
        for cl in self.confidence_levels:
            z_score: float = stats.norm.ppf(cl)
            var_dollar: float = z_score * port_std_horizon * portfolio_value
            key: str = f"Parametric_VaR_{int(cl * 100)}"
            results[key] = round(var_dollar, 2)
            logger.info(
                "%s: $%.2f (z=%.4f, σ_daily=%.6f)",
                key, var_dollar, z_score, port_std_daily,
            )

        return results

    def historical_var(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute Historical (non-parametric) Value-at-Risk.

        Uses the empirical distribution of past portfolio returns.

        Formula:
            VaR_α = -Percentile(R_p, (1 - α) × 100) × V

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights.
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, float]
            Keys: 'Historical_VaR_95', 'Historical_VaR_99', etc.
        """
        port_returns: pd.Series = self.compute_portfolio_returns(returns, weights)
        clean_returns = port_returns.dropna()

        results: Dict[str, float] = {}
        for cl in self.confidence_levels:
            key: str = f"Historical_VaR_{int(cl * 100)}"
            if clean_returns.empty:
                logger.warning("Portfolio returns series is empty after dropping NaNs in historical_var.")
                results[key] = 0.0
                continue
            percentile: float = (1 - cl) * 100
            var_pct: float = float(np.percentile(clean_returns, percentile))
            var_dollar: float = abs(var_pct) * portfolio_value
            results[key] = round(var_dollar, 2)
            logger.info(
                "%s: $%.2f (percentile=%.1f%%)", key, var_dollar, percentile,
            )

        return results

    def monte_carlo_var(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute Monte Carlo Value-at-Risk via multivariate normal simulation.

        Procedure:
            1. Estimate mean vector (μ) and covariance matrix (Σ) from historical returns.
            2. Simulate N portfolio return paths from N(μ, Σ).
            3. Compute portfolio-level returns as weighted sum.
            4. VaR = -Percentile(simulated_portfolio_returns, (1 - α) × 100) × V.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights.
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, float]
            Keys: 'MonteCarlo_VaR_95', 'MonteCarlo_VaR_99', etc.
        """
        clean_returns_df = returns.dropna(how='all', axis=1).dropna()
        results: Dict[str, float] = {}

        if clean_returns_df.empty:
            logger.warning("Returns DataFrame is empty in Monte Carlo VaR.")
            for cl in self.confidence_levels:
                results[f"MonteCarlo_VaR_{int(cl * 100)}"] = 0.0
            return results

        aligned_w: np.ndarray = weights.reindex(clean_returns_df.columns).fillna(0.0).values
        mean_returns: np.ndarray = clean_returns_df.mean().values
        cov_matrix: np.ndarray = clean_returns_df.cov().values

        # Vectorized simulation: generate all N paths at once
        rng: np.random.Generator = np.random.default_rng(self.random_seed)
        simulated_returns: np.ndarray = rng.multivariate_normal(
            mean=mean_returns, cov=cov_matrix, size=self.n_simulations
        )

        # Portfolio-level simulated returns: (N,) = (N, assets) @ (assets,)
        sim_port_returns: np.ndarray = simulated_returns @ aligned_w

        for cl in self.confidence_levels:
            percentile: float = (1 - cl) * 100
            var_pct: float = float(np.percentile(sim_port_returns, percentile))
            var_dollar: float = abs(var_pct) * portfolio_value
            key: str = f"MonteCarlo_VaR_{int(cl * 100)}"
            results[key] = round(var_dollar, 2)
            logger.info(
                "%s: $%.2f (N=%d simulations)", key, var_dollar, self.n_simulations,
            )

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # EXPECTED SHORTFALL (CVaR / ES)
    # ═══════════════════════════════════════════════════════════════════════

    def parametric_es(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute Parametric (Gaussian) Expected Shortfall.

        Closed-form under normality assumption.

        Formula:
            ES_α = σ_p × φ(z_α) / (1 - α) × V

        Where φ(z) is the standard normal PDF evaluated at z_α = Φ⁻¹(α).

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights.
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, float]
            Keys: 'Parametric_ES_95', 'Parametric_ES_99', etc.
        """
        cov_daily: pd.DataFrame = self.compute_covariance_matrix(
            returns, annualize=False
        )
        w: np.ndarray = weights.reindex(returns.columns).fillna(0.0).values

        port_variance: float = float(w @ cov_daily.values @ w)
        port_std_daily: float = np.sqrt(port_variance)
        port_std_horizon: float = port_std_daily * np.sqrt(self.holding_period)

        results: Dict[str, float] = {}
        for cl in self.confidence_levels:
            z_score: float = stats.norm.ppf(cl)
            phi_z: float = stats.norm.pdf(z_score)
            es_pct: float = port_std_horizon * phi_z / (1 - cl)
            es_dollar: float = es_pct * portfolio_value
            key: str = f"Parametric_ES_{int(cl * 100)}"
            results[key] = round(es_dollar, 2)
            logger.info(
                "%s: $%.2f (φ(z)=%.6f, σ_horizon=%.6f)",
                key, es_dollar, phi_z, port_std_horizon,
            )

        return results

    def historical_es(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute Historical Expected Shortfall (CVaR).

        The average loss in the tail beyond the VaR threshold.

        Formula:
            ES_α = E[R_p | R_p ≤ -VaR_α] × V

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights.
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, float]
            Keys: 'Historical_ES_95', 'Historical_ES_99', etc.
        """
        port_returns: pd.Series = self.compute_portfolio_returns(returns, weights)
        clean_returns: np.ndarray = port_returns.dropna().values

        results: Dict[str, float] = {}
        for cl in self.confidence_levels:
            key: str = f"Historical_ES_{int(cl * 100)}"
            if len(clean_returns) == 0:
                logger.warning("Portfolio returns series is empty after dropping NaNs in historical_es.")
                results[key] = 0.0
                continue
            percentile: float = (1 - cl) * 100
            var_threshold: float = float(np.percentile(clean_returns, percentile))
            tail_returns: np.ndarray = clean_returns[clean_returns <= var_threshold]

            if len(tail_returns) == 0:
                es_dollar: float = 0.0
            else:
                es_dollar = abs(float(np.mean(tail_returns))) * portfolio_value

            results[key] = round(es_dollar, 2)
            logger.info(
                "%s: $%.2f (tail observations=%d)", key, es_dollar, len(tail_returns),
            )

        return results

    def monte_carlo_es(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute Monte Carlo Expected Shortfall.

        Mean of simulated portfolio returns beyond the VaR threshold.

        Formula:
            ES_α = E[R_sim | R_sim ≤ -VaR_α] × V

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights.
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, float]
            Keys: 'MonteCarlo_ES_95', 'MonteCarlo_ES_99', etc.
        """
        clean_returns_df = returns.dropna(how='all', axis=1).dropna()
        results: Dict[str, float] = {}

        if clean_returns_df.empty:
            logger.warning("Returns DataFrame is empty in Monte Carlo ES.")
            for cl in self.confidence_levels:
                results[f"MonteCarlo_ES_{int(cl * 100)}"] = 0.0
            return results

        aligned_w: np.ndarray = weights.reindex(clean_returns_df.columns).fillna(0.0).values
        mean_returns: np.ndarray = clean_returns_df.mean().values
        cov_matrix: np.ndarray = clean_returns_df.cov().values

        rng: np.random.Generator = np.random.default_rng(self.random_seed)
        simulated_returns: np.ndarray = rng.multivariate_normal(
            mean=mean_returns, cov=cov_matrix, size=self.n_simulations
        )
        sim_port_returns: np.ndarray = simulated_returns @ aligned_w

        for cl in self.confidence_levels:
            percentile: float = (1 - cl) * 100
            var_threshold: float = float(np.percentile(sim_port_returns, percentile))
            tail_returns: np.ndarray = sim_port_returns[
                sim_port_returns <= var_threshold
            ]

            if len(tail_returns) == 0:
                es_dollar: float = 0.0
            else:
                es_dollar = abs(float(np.mean(tail_returns))) * portfolio_value

            key: str = f"MonteCarlo_ES_{int(cl * 100)}"
            results[key] = round(es_dollar, 2)
            logger.info(
                "%s: $%.2f (tail sims=%d / %d)",
                key, es_dollar, len(tail_returns), self.n_simulations,
            )

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # CONSOLIDATED REPORT
    # ═══════════════════════════════════════════════════════════════════════

    def full_risk_report(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, object]:
        """
        Generate a consolidated risk report combining all VaR and ES measures.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.
        weights : pd.Series
            Portfolio weights.
        portfolio_value : float
            Current total portfolio market value.

        Returns
        -------
        Dict[str, object]
            Keys: 'parametric_var', 'historical_var', 'monte_carlo_var',
            'parametric_es', 'historical_es', 'monte_carlo_es',
            'covariance_matrix', 'correlation_matrix'.
        """
        logger.info(
            "Generating full risk report (portfolio_value=$%.2f)...", portfolio_value
        )

        report: Dict[str, object] = {
            "parametric_var": self.parametric_var(returns, weights, portfolio_value),
            "historical_var": self.historical_var(returns, weights, portfolio_value),
            "monte_carlo_var": self.monte_carlo_var(returns, weights, portfolio_value),
            "parametric_es": self.parametric_es(returns, weights, portfolio_value),
            "historical_es": self.historical_es(returns, weights, portfolio_value),
            "monte_carlo_es": self.monte_carlo_es(returns, weights, portfolio_value),
            "covariance_matrix": self.compute_covariance_matrix(returns),
            "correlation_matrix": self.compute_correlation_matrix(returns),
        }

        logger.info("Full risk report generated successfully.")
        return report
