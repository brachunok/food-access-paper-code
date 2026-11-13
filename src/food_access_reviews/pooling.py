"""Bayesian partial pooling of store-level dimension scores."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def partial_pool_one_dimension(
    store_scores: pd.DataFrame,
    dimension: str,
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.95,
    seed: int = 123,
):
    """Fit the manuscript's normal–normal hierarchical model for one dimension."""
    try:
        import arviz as az
        import pymc as pm
    except ImportError as error:
        raise ImportError(
            "Partial pooling requires the 'modeling' extra: "
            "pip install -e '.[modeling]'"
        ) from error

    frame = (
        store_scores[
            [f"{dimension}_mean", f"{dimension}_se", f"{dimension}_n"]
        ]
        .rename(
            columns={
                f"{dimension}_mean": "ybar",
                f"{dimension}_se": "se",
                f"{dimension}_n": "n",
            }
        )
        .query("n > 0")
        .dropna(subset=["ybar", "se"])
        .copy()
    )
    if frame.empty:
        raise ValueError(f"No usable stores for {dimension}")

    started = time.time()
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 0.5)
        tau = pm.HalfNormal("tau", 0.5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(frame))
        pm.Normal("obs", mu=theta, sigma=frame["se"].to_numpy(), observed=frame["ybar"])
        inference = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
        )

    theta_draws = (
        inference.posterior["theta"].stack(sample=("chain", "draw")).values
    )
    pooled = pd.DataFrame(index=frame.index)
    pooled[f"{dimension}_pooled_mean"] = theta_draws.mean(axis=1)
    pooled[f"{dimension}_pooled_lo95"] = np.quantile(theta_draws, 0.025, axis=1)
    pooled[f"{dimension}_pooled_hi95"] = np.quantile(theta_draws, 0.975, axis=1)
    pooled[f"{dimension}_shrinkage"] = (
        pooled[f"{dimension}_pooled_mean"] - frame["ybar"]
    )

    diagnostics = az.summary(inference, var_names=["mu", "tau"], hdi_prob=0.94)
    summary = {
        "dimension": dimension,
        "n_stores": len(frame),
        "mu_mean": float(diagnostics.loc["mu", "mean"]),
        "tau_mean": float(diagnostics.loc["tau", "mean"]),
        "rhat_mu": float(diagnostics.loc["mu", "r_hat"]),
        "rhat_tau": float(diagnostics.loc["tau", "r_hat"]),
        "seconds": time.time() - started,
    }
    return store_scores.join(pooled, how="left"), inference, summary

