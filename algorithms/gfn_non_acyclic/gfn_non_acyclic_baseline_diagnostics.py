"""
Code for Trajectory Balance (TB) training.
For further details see: https://arxiv.org/abs/2301.12594 and https://arxiv.org/abs/2501.06148
"""

from functools import partial
import os

import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import wandb

from algorithms.common.diffusion_related.init_model import init_model_non_acyclic
from algorithms.gfn_non_acyclic.gfn_non_acyclic_rnd import rnd_mcmc
from eval import discrepancies
from eval.utils import extract_last_entry
from algorithms.gfn_non_acyclic.mcmc_diagnostics import (
    compute_geweke_z,
    rhat_status,
)
from utils.print_utils import print_results


_RHAT_CURVE_BASES = (
    "curves/rhat_max_rank_split",
    "curves/rhat_max_split",
    "curves/rhat_max_classical",
)
# Palette used to assign one consistent color per gamma in multi-gamma overlay
# plots. Indexed by the position of the gamma in `gammas`.
_GAMMA_PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf")


def _format_gamma(g):
    return f"{float(g):g}"


def gfn_non_acyclic_baseline_diagnostics(cfg, target, exp=None):
    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size
    n_eval_seeds = int(getattr(alg_cfg, "n_eval_seeds", 3))

    compute_mmd = bool(getattr(alg_cfg, "compute_mmd", True))
    metric_names = ("mmd", "sd") if compute_mmd else ("sd",)

    # Resolve list of gammas to compare
    raw_gammas = getattr(alg_cfg, "gammas", None)
    if raw_gammas is None:
        gammas = [float(alg_cfg.model.gamma)]
    else:
        gammas = [float(g) for g in raw_gammas]
    multi_gamma = len(gammas) > 1

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    initial_dist = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )

    def _make_rnd_eval(gamma):
        return partial(
            rnd_mcmc,
            aux_tuple=(gamma,),
            target=target,
            num_steps=alg_cfg.eval_max_steps,
            step_name=alg_cfg.step_name,
            initial_dist=initial_dist,
        )

    logger = {}
    # Curve keys treated as `exp.log_curve` curves rather than scalar metrics.
    curve_keys = set()

    def _build_t_grid(max_steps: int, min_steps: int, n_points: int):
        grid = np.linspace(min_steps, max_steps, num=n_points, dtype=int)
        grid = np.clip(grid, min_steps, max_steps)
        return np.unique(grid)

    def _samples_at_step(trajectories, t_idx):
        t_idx = int(np.clip(t_idx, 0, trajectories.shape[1] - 1))
        return trajectories[:, t_idx]

    def _get_reference(group: str, metric: str):
        """Look up `reference_metrics` in the target config, preferring a
        dim-specific block (``dim_<N>``) and falling back to ``default``
        when present.
        """
        ref_root = getattr(cfg.target, "reference_metrics", None)
        if ref_root is None:
            return None

        dim_block = getattr(ref_root, f"dim_{int(dim)}", None)
        default_block = getattr(ref_root, "default", None)

        for source in (dim_block, default_block):
            if source is None:
                continue
            group_cfg = getattr(source, group, None)
            if group_cfg is None:
                continue
            value = getattr(group_cfg, metric, None)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _xerr_yerr(xy, x_ci, y_ci):
        """Build (xerr, yerr) arrays for ax.errorbar from CI tuples."""
        x, y = xy

        def _err(value, ci):
            if (
                ci is None
                or value is None
                or not np.isfinite(value)
                or not np.all(np.isfinite(ci))
            ):
                return None
            lo, hi = ci
            return np.asarray([[max(value - lo, 0.0)], [max(hi - value, 0.0)]])

        return _err(x, x_ci), _err(y, y_ci)

    def _plot_metric_curves_overlay(metric, gammas_list, results):
        """Render one figure for ``metric`` overlaying every gamma in
        ``gammas_list`` with its corresponding entry in ``results``.

        - Single-gamma mode (``len(gammas_list) == 1``) keeps the legacy
          colors (gold curve, green R-hat dot, orange Geweke dot).
        - Multi-gamma mode assigns one palette color per gamma; R-hat is
          drawn as ``"v"`` and Geweke as ``"^"`` so dot type is
          encoded by marker shape rather than color.
        - Reference dots ("ULA + CLF" purple, "Full model" red) are drawn
          once.
        - The 95% CI band has no legend entry in either mode.
        """
        fig = plt.figure(figsize=(8, 5.5), dpi=200)
        ax = fig.add_subplot()

        single = len(gammas_list) == 1

        errorbar_kwargs_base = dict(
            markersize=7,
            capsize=2,
            elinewidth=1.0,
            markeredgewidth=1.0,
            zorder=5,
        )

        def _draw_dot(x, y, color, label, x_ci, y_ci, marker="X"):
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                return
            xerr, yerr = _xerr_yerr((x, y), x_ci, y_ci)
            ax.errorbar(
                [x],
                [y],
                xerr=xerr,
                yerr=yerr,
                color=color,
                ecolor=color,
                label=label,
                fmt=marker,
                **errorbar_kwargs_base,
            )

        for i, (g, data) in enumerate(zip(gammas_list, results)):
            if single:
                line_color = "gold"
                rhat_color = "#2ca02c"
                geweke_color = "#ff7f0e"
                line_label = "ULA"
                rhat_label = "ULA RG"
                geweke_label = "ULA Geweke"
                rhat_marker = "X"
                geweke_marker = "X"
            else:
                line_color = _GAMMA_PALETTE[i % len(_GAMMA_PALETTE)]
                rhat_color = line_color
                geweke_color = line_color
                line_label = f"ULA \u03b3={_format_gamma(g)}"
                rhat_label = None
                geweke_label = None
                rhat_marker = "v"
                geweke_marker = "^"

            x = np.asarray(data["t_grid"])
            ula = np.asarray(data["metric_curves"][metric], dtype=float)
            ax.plot(x, ula, label=line_label, color=line_color, linewidth=2.0)

            lo = data["metric_ci_low"].get(metric)
            hi = data["metric_ci_high"].get(metric)
            if lo is not None and hi is not None:
                lo = np.asarray(lo, dtype=float)
                hi = np.asarray(hi, dtype=float)
                mask = np.isfinite(lo) & np.isfinite(hi)
                if mask.any():
                    ax.fill_between(
                        x[mask],
                        lo[mask],
                        hi[mask],
                        color=line_color,
                        alpha=0.20,
                        linewidth=0,
                    )

            rhat_xy = (data["rhat_x_for_plot"], data["rhat_metric_values"].get(metric))
            _draw_dot(
                *rhat_xy,
                color=rhat_color,
                label=rhat_label,
                x_ci=data["rhat_x_ci"],
                y_ci=data["rhat_metric_ci"].get(metric),
                marker=rhat_marker,
            )

            gx = data["geweke_x_for_plot"]
            geweke_xy = (
                (gx, data["geweke_metric_values"].get(metric))
                if gx is not None
                else None
            )
            if geweke_xy is not None:
                _draw_dot(
                    *geweke_xy,
                    color=geweke_color,
                    label=geweke_label,
                    x_ci=data["geweke_x_ci"],
                    y_ci=data["geweke_metric_ci"].get(metric),
                    marker=geweke_marker,
                )

        # Multi-gamma legend: single shape-only entries for RG / Geweke so the
        # legend stays compact regardless of how many gammas are overlaid.
        if not single:
            from matplotlib.lines import Line2D

            shape_proxies = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="v",
                    color="gray",
                    markersize=8,
                    label="ULA RG",
                ),
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="^",
                    color="gray",
                    markersize=8,
                    label="ULA Geweke",
                ),
            ]
        else:
            shape_proxies = []

        ula_clf_mtl = _get_reference("ula_clf", "mtl")
        ula_clf_val = _get_reference("ula_clf", metric)
        if ula_clf_mtl is not None and ula_clf_val is not None:
            _draw_dot(
                ula_clf_mtl,
                ula_clf_val,
                color="#9467bd",
                label="ULA + CLF",
                x_ci=None,
                y_ci=None,
            )

        full_mtl = _get_reference("full_model", "mtl")
        full_val = _get_reference("full_model", metric)
        if full_mtl is not None and full_val is not None:
            _draw_dot(
                full_mtl,
                full_val,
                color="#d62728",
                label="Full model",
                x_ci=None,
                y_ci=None,
            )

        ax.set_xlabel("chain length")
        if metric == "sd":
            ax.set_ylabel("W2")
        else:
            ax.set_ylabel(metric.upper())
            ax.set_title(metric.upper())
        ax.grid(alpha=0.3)
        if shape_proxies:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles + shape_proxies, labels + [p.get_label() for p in shape_proxies]
            )
        else:
            ax.legend()
        fig.tight_layout()
        return fig

    def _compute_metric(name, samples, gt=None):
        ref = target_xs if gt is None else gt
        if ref is None:
            return jnp.asarray(jnp.inf)
        return getattr(discrepancies, f"compute_{name}")(ref, samples, cfg)

    _batches_cfg = getattr(getattr(alg_cfg, "early_stop", None), "batches", None)
    BATCH_N = (
        int(getattr(_batches_cfg, "n_batches", 4)) if _batches_cfg is not None else 4
    )
    BATCH_ALPHA = (
        float(getattr(_batches_cfg, "alpha", 0.05))
        if _batches_cfg is not None
        else 0.05
    )
    _batch_rng = np.random.default_rng(0)
    _NAN_JNP = jnp.asarray(jnp.nan)
    _Q_LO = 100.0 * (BATCH_ALPHA / 2.0)
    _Q_HI = 100.0 * (1.0 - BATCH_ALPHA / 2.0)

    def _percentile_ci(values, alpha=BATCH_ALPHA):
        if isinstance(values, list):
            if len(values) == 0:
                return _NAN_JNP, _NAN_JNP
            arr = jnp.stack([jnp.asarray(v) for v in values])
        else:
            arr = jnp.asarray(values)
            if arr.size == 0:
                return _NAN_JNP, _NAN_JNP
        if alpha == BATCH_ALPHA:
            q_lo, q_hi = _Q_LO, _Q_HI
        else:
            q_lo = 100.0 * (alpha / 2.0)
            q_hi = 100.0 * (1.0 - alpha / 2.0)
        return jnp.nanpercentile(arr, q_lo), jnp.nanpercentile(arr, q_hi)

    def _chain_batch_indices(n_items, n_batches=BATCH_N):
        if n_items <= 0 or n_batches <= 0:
            return []
        idx = _batch_rng.permutation(n_items)
        return [b for b in np.array_split(idx, n_batches) if b.size > 0]

    def _batch_metric_mean_ci(
        metric_name, samples, n_batches=BATCH_N, alpha=BATCH_ALPHA
    ):
        if target_xs is None or n_batches < 2:
            return _NAN_JNP, _NAN_JNP, _NAN_JNP
        n = int(samples.shape[0])
        n_t = int(target_xs.shape[0])
        if n < n_batches:
            return _NAN_JNP, _NAN_JNP, _NAN_JNP
        estimates = []
        for batch in _chain_batch_indices(n, n_batches):
            b_size = int(batch.size)
            if b_size <= n_t:
                t_idx = _batch_rng.choice(n_t, size=b_size, replace=False)
                target_sub = target_xs[t_idx]
            else:
                target_sub = target_xs
            estimates.append(
                _compute_metric(metric_name, samples[batch], gt=target_sub)
            )
        if not estimates:
            return _NAN_JNP, _NAN_JNP, _NAN_JNP
        arr = jnp.stack([jnp.asarray(v) for v in estimates])
        if alpha == BATCH_ALPHA:
            q_lo, q_hi = _Q_LO, _Q_HI
        else:
            q_lo = 100.0 * (alpha / 2.0)
            q_hi = 100.0 * (1.0 - alpha / 2.0)
        return (
            jnp.nanmean(arr),
            jnp.nanpercentile(arr, q_lo),
            jnp.nanpercentile(arr, q_hi),
        )

    def _batch_rhat_stop_ci(
        traj_np,
        t_grid,
        rhat_thr,
        geweke_valid_t_mask=None,
        n_batches=BATCH_N,
        alpha=BATCH_ALPHA,
    ):
        if n_batches < 2:
            return _NAN_JNP, _NAN_JNP
        n_c = traj_np.shape[0]
        if n_c < n_batches:
            return _NAN_JNP, _NAN_JNP
        stops = []
        for batch in _chain_batch_indices(n_c, n_batches):
            if batch.size < 2:
                continue
            sub = jnp.asarray(traj_np[batch])
            curve = []
            for i, t in enumerate(t_grid):
                if geweke_valid_t_mask is not None and not bool(geweke_valid_t_mask[i]):
                    curve.append(float("inf"))
                else:
                    curve.append(float(_rhat_classical_point_jax(sub, int(t))))
            triggered = next((i for i, v in enumerate(curve) if v < rhat_thr), None)
            stops.append(
                int(t_grid[triggered])
                if (triggered is not None and triggered < len(t_grid) - 1)
                else int(t_grid[-1])
            )
        if not stops:
            return _NAN_JNP, _NAN_JNP
        return _percentile_ci(jnp.asarray(stops, dtype=jnp.float32), alpha)

    _DIAG_EPS = 1e-12

    def _psrf_with_chain_masks_jax(x, chain_masks):
        # x: [chains, steps, params], chain_masks: [chains, steps] bool
        m = x.shape[0]
        mask_f = chain_masks.astype(jnp.float32)[:, :, None]
        n_eff = jnp.sum(mask_f, axis=1)  # [chains, 1]
        n_eff_safe = jnp.maximum(n_eff, 1.0)
        means = jnp.sum(x * mask_f, axis=1) / n_eff_safe

        centered = (x - means[:, None, :]) * mask_f
        vars_ = jnp.sum(centered * centered, axis=1) / jnp.maximum(n_eff - 1.0, 1.0)
        w = jnp.mean(vars_, axis=0)
        overall = jnp.mean(means, axis=0)
        n_bar = jnp.maximum(jnp.mean(n_eff), 1.0)
        b = n_bar * jnp.sum((means - overall[None, :]) ** 2, axis=0) / max(m - 1, 1)
        vhat = ((n_bar - 1.0) / n_bar) * w + b / n_bar
        rhat = jnp.sqrt(jnp.maximum(vhat, _DIAG_EPS) / jnp.maximum(w, _DIAG_EPS))

        valid_chain = jnp.all(n_eff[:, 0] >= 2.0)
        valid = jnp.logical_and(m >= 2, valid_chain)
        return jnp.where(valid, rhat, jnp.full_like(rhat, jnp.inf))

    @jax.jit
    def _rhat_classical_point_jax(traj, t):
        # traj: [chains, max_steps, params]
        c, n, _ = traj.shape
        step_idx = jnp.arange(n)
        active = step_idx < t
        active_cn = jnp.broadcast_to(active[None, :], (c, n))

        classical_vec = _psrf_with_chain_masks_jax(traj, active_cn)
        return jnp.max(classical_vec)

    @partial(jax.jit, static_argnames=("estimator",))
    def _geweke_max_abs_per_chain_jax(traj, t, frac1, frac2, estimator="ar"):
        # Returns per-chain max |z| over params; invalid t returns +inf.
        _, n, _ = traj.shape
        t_f = t.astype(jnp.float32)
        n_a = jnp.maximum((frac1 * t_f).astype(jnp.int32), 2)
        n_b = jnp.maximum((frac2 * t_f).astype(jnp.int32), 2)
        valid = (n_a + n_b) < t

        idx = jnp.arange(n)
        mask_a = idx < n_a
        mask_b = jnp.logical_and(idx >= (t - n_b), idx < t)
        ma = mask_a.astype(jnp.float32)[None, :, None]
        mb = mask_b.astype(jnp.float32)[None, :, None]

        def _masked_mean(x, m):
            denom = jnp.maximum(jnp.sum(m, axis=1, keepdims=True), 1.0)
            return jnp.sum(x * m, axis=1, keepdims=True) / denom

        def _spectrum_iid(x, m):
            n_eff = jnp.maximum(jnp.sum(m, axis=1), 2.0)
            mu = _masked_mean(x, m)
            xc = (x - mu) * m
            return jnp.sum(xc * xc, axis=1) / jnp.maximum(n_eff - 1.0, 1.0)

        def _spectrum_ar1(x, m):
            n_eff = jnp.maximum(jnp.sum(m, axis=1), 2.0)
            mu = _masked_mean(x, m)
            xc = (x - mu) * m
            gamma0 = jnp.sum(xc * xc, axis=1) / jnp.maximum(n_eff, 1.0)

            pair_mask = m[:, 1:, :] * m[:, :-1, :]
            gamma1_num = jnp.sum(xc[:, 1:, :] * xc[:, :-1, :] * pair_mask, axis=1)
            gamma1_den = jnp.maximum(jnp.sum(pair_mask, axis=1), 1.0)
            gamma1 = gamma1_num / gamma1_den

            phi = jnp.clip(gamma1 / jnp.maximum(gamma0, _DIAG_EPS), -0.99, 0.99)
            sigma2 = jnp.maximum(gamma0 * (1.0 - phi * phi), _DIAG_EPS)
            return sigma2 / jnp.maximum((1.0 - phi) ** 2, _DIAG_EPS)

        sa0 = jax.lax.cond(
            estimator == "iid",
            lambda _: _spectrum_iid(traj, ma),
            lambda _: _spectrum_ar1(traj, ma),
            operand=None,
        )
        sb0 = jax.lax.cond(
            estimator == "iid",
            lambda _: _spectrum_iid(traj, mb),
            lambda _: _spectrum_ar1(traj, mb),
            operand=None,
        )
        mean_a = jnp.squeeze(_masked_mean(traj, ma), axis=1)
        mean_b = jnp.squeeze(_masked_mean(traj, mb), axis=1)
        denom = jnp.sqrt(
            jnp.maximum(
                sa0 / jnp.maximum(n_a.astype(jnp.float32), 1.0)
                + sb0 / jnp.maximum(n_b.astype(jnp.float32), 1.0),
                _DIAG_EPS,
            )
        )
        z = (mean_a - mean_b) / denom
        max_abs_per_chain = jnp.max(jnp.abs(z), axis=1)
        return jnp.where(
            valid, max_abs_per_chain, jnp.full_like(max_abs_per_chain, jnp.inf)
        )

    def _compute_curves_and_diagnostics(trajectories):
        """
        Compute ULA curves, R-hat / Geweke diagnostics, and chain-batch CIs.
        """
        diag_cfg = getattr(cfg.algorithm, "early_stop", None)
        min_steps = int(getattr(diag_cfg, "min_chain_length", 20))
        n_grid = int(getattr(diag_cfg, "n_checkpoints", 5))
        max_steps = int(trajectories.shape[1])
        t_grid = _build_t_grid(max_steps, min_steps, n_grid)

        traj_jnp = jnp.asarray(trajectories)
        n_chains = int(trajectories.shape[0])

        # JAX-side accumulators (lists of `jnp` scalars)
        metric_curves_jnp = {m: [] for m in metric_names}
        metric_ci_low_jnp = {m: [] for m in metric_names}
        metric_ci_high_jnp = {m: [] for m in metric_names}
        # NumPy-side accumulators (R-hat curves use numpy returns).
        rhat_rank_split = []
        rhat_split = []
        rhat_classical = []
        # Per-trajectory Geweke stop tracking: -1 = not stopped yet.
        stop_t_per_traj = np.full(n_chains, -1, dtype=int)
        geweke_max_abs_curve = []
        geweke_frac_stopped_curve = []
        max_abs_z_threshold = float(
            getattr(diag_cfg.geweke, "max_abs_z_threshold", 4.0)
        )
        gw_frac1 = float(getattr(diag_cfg.geweke, "frac1", 0.1))
        gw_frac2 = float(getattr(diag_cfg.geweke, "frac2", 0.5))
        gw_estimator = str(getattr(diag_cfg.geweke, "spectral_estimator", "ar"))
        # Keep R-hat evaluation cadence aligned with Geweke validity.
        # Geweke is valid when n_a + n_b < t with n_a=max(int(frac1*t),2), n_b=max(int(frac2*t),2).
        geweke_valid_t_mask = []

        for t in t_grid:
            t_int = int(t)
            samples_t = trajectories[:, t_int - 1, :]
            n_a = max(int(gw_frac1 * t_int), 2)
            n_b = max(int(gw_frac2 * t_int), 2)
            geweke_valid_t = (n_a + n_b) < t_int
            geweke_valid_t_mask.append(geweke_valid_t)

            for m in metric_names:
                mean, lo, hi = _batch_metric_mean_ci(m, samples_t)
                metric_curves_jnp[m].append(mean)
                metric_ci_low_jnp[m].append(lo)
                metric_ci_high_jnp[m].append(hi)

            if geweke_valid_t:
                classical_t = _rhat_classical_point_jax(traj_jnp, t_int)
                # Keep legacy curve keys for compatibility with existing dashboards.
                rhat_rank_split.append(float(classical_t))
                rhat_split.append(float(classical_t))
                rhat_classical.append(float(classical_t))
            else:
                rhat_rank_split.append(float("inf"))
                rhat_split.append(float("inf"))
                rhat_classical.append(float("inf"))

            max_abs_per_chain = None
            # Use JAX Geweke kernel for ar/iid; keep batchmeans path unchanged.
            if gw_estimator in ("ar", "iid"):
                max_abs_per_chain = np.asarray(
                    _geweke_max_abs_per_chain_jax(
                        traj_jnp,
                        jnp.asarray(t_int, dtype=jnp.int32),
                        jnp.asarray(gw_frac1, dtype=jnp.float32),
                        jnp.asarray(gw_frac2, dtype=jnp.float32),
                        estimator=gw_estimator,
                    )
                )
                newly = (stop_t_per_traj == -1) & (
                    max_abs_per_chain <= max_abs_z_threshold
                )
                stop_t_per_traj[newly] = t_int
            else:
                try:
                    z = compute_geweke_z(
                        np.asarray(trajectories)[:, :t_int, :],
                        frac1=gw_frac1,
                        frac2=gw_frac2,
                        estimator=gw_estimator,
                    )
                    max_abs_per_chain = np.max(np.abs(z), axis=1)
                    newly = (stop_t_per_traj == -1) & (
                        max_abs_per_chain <= max_abs_z_threshold
                    )
                    stop_t_per_traj[newly] = t_int
                except ValueError:
                    # Chain too short for Geweke at this t; skip update.
                    pass

            if geweke_valid_t and max_abs_per_chain is not None:
                geweke_max_abs_curve.append(float(np.max(max_abs_per_chain)))
            else:
                geweke_max_abs_curve.append(float("nan"))
            geweke_frac_stopped_curve.append(float((stop_t_per_traj != -1).mean()))

        # Single sync per stacked array (collapses all queued metric kernels).
        metric_curves = {
            m: np.asarray(jnp.stack(v)) for m, v in metric_curves_jnp.items()
        }
        metric_ci_low = {
            m: np.asarray(jnp.stack(v)) for m, v in metric_ci_low_jnp.items()
        }
        metric_ci_high = {
            m: np.asarray(jnp.stack(v)) for m, v in metric_ci_high_jnp.items()
        }

        return {
            "t_grid": np.asarray(t_grid, dtype=int),
            "metric_curves": metric_curves,
            "metric_ci_low": metric_ci_low,
            "metric_ci_high": metric_ci_high,
            "rhat_rank_split": np.asarray(rhat_rank_split),
            "rhat_split": np.asarray(rhat_split),
            "rhat_classical": np.asarray(rhat_classical),
            "stop_t_per_traj": stop_t_per_traj,
            "geweke_valid_t_mask": np.asarray(geweke_valid_t_mask, dtype=bool),
            "geweke_max_abs_curve": np.asarray(geweke_max_abs_curve, dtype=float),
            "geweke_frac_stopped_curve": np.asarray(
                geweke_frac_stopped_curve, dtype=float
            ),
        }

    def _eval_compute(model_state, key, gamma):
        """
        Pure-compute eval for one gamma.
        """
        rnd_eval_partial = _make_rnd_eval(gamma)
        params = (model_state.params,)
        trajectories, _, _, _ = rnd_eval_partial(
            key,
            model_state,
            *params,
            batch_size=cfg.eval_samples,
        )
        samples_full = trajectories[:, -1]

        diag = _compute_curves_and_diagnostics(trajectories)
        t_grid = diag["t_grid"]

        # R-hat global early stopping.
        rhat_thr = float(getattr(alg_cfg.early_stop.rhat, "threshold", 1.01))
        rhat_curve = diag["rhat_classical"]
        triggered_idx = next(
            (i for i, v in enumerate(rhat_curve) if v < rhat_thr), None
        )
        if triggered_idx is not None and triggered_idx < len(t_grid) - 1:
            t_stop_rhat = int(t_grid[triggered_idx])
            rhat_triggered = True
            rhat_status_str = rhat_status(float(rhat_curve[triggered_idx]))
        else:
            t_stop_rhat = int(t_grid[-1])
            rhat_triggered = False
            rhat_status_str = rhat_status(float(rhat_curve[-1]))
        samples_rhat = _samples_at_step(trajectories, t_stop_rhat - 1)

        rhat_metric_jnp = {}
        rhat_metric_ci_jnp = {}
        for m in metric_names:
            mean, lo, hi = _batch_metric_mean_ci(m, samples_rhat)
            rhat_metric_jnp[m] = mean
            rhat_metric_ci_jnp[m] = (lo, hi)
        rhat_x_ci_jnp = _batch_rhat_stop_ci(
            np.asarray(trajectories),
            t_grid,
            rhat_thr,
            geweke_valid_t_mask=diag["geweke_valid_t_mask"],
        )
        rhat_visualise = None
        if rhat_triggered:
            rhat_visualise = target.visualise(samples=samples_rhat, prefix="rhat_stop")

        # Per-trajectory Geweke early stopping.
        stop_t_per_traj = diag["stop_t_per_traj"]
        stopped_mask = stop_t_per_traj != -1
        n_chains = stop_t_per_traj.shape[0]
        frac_stopped = float(stopped_mask.mean()) if n_chains > 0 else 0.0

        geweke_metric_jnp = {}
        geweke_metric_ci_jnp = {}
        geweke_visualise = None
        geweke_x_for_plot = None
        geweke_x_ci_jnp = (_NAN_JNP, _NAN_JNP)
        geweke_stop_mean = float("nan")
        geweke_stop_median = float("nan")
        if stopped_mask.any():
            stopped_t = stop_t_per_traj[stopped_mask]
            geweke_stop_mean = float(np.mean(stopped_t))
            geweke_stop_median = float(np.median(stopped_t))

            stopped_idx = np.where(stopped_mask)[0]
            partial_samples = trajectories[
                stopped_idx, stop_t_per_traj[stopped_mask] - 1
            ]
            for m in metric_names:
                mean, lo, hi = _batch_metric_mean_ci(m, partial_samples)
                geweke_metric_jnp[m] = mean
                geweke_metric_ci_jnp[m] = (lo, hi)
            geweke_visualise = target.visualise(
                samples=partial_samples, prefix="geweke_partial"
            )
            geweke_x_for_plot = geweke_stop_mean
            geweke_x_ci_jnp = _percentile_ci(jnp.asarray(stopped_t, dtype=jnp.float32))

        # Single host sync over every queued device computation.
        bundle = jax.device_get(
            {
                "rhat_metric": rhat_metric_jnp,
                "rhat_metric_ci": rhat_metric_ci_jnp,
                "rhat_x_ci": rhat_x_ci_jnp,
                "geweke_metric": geweke_metric_jnp,
                "geweke_metric_ci": geweke_metric_ci_jnp,
                "geweke_x_ci": geweke_x_ci_jnp,
            }
        )
        rhat_metric_values = {m: float(bundle["rhat_metric"][m]) for m in metric_names}
        rhat_metric_ci = {
            m: (
                float(bundle["rhat_metric_ci"][m][0]),
                float(bundle["rhat_metric_ci"][m][1]),
            )
            for m in metric_names
        }
        rhat_x_ci = (float(bundle["rhat_x_ci"][0]), float(bundle["rhat_x_ci"][1]))
        geweke_metric_values = {
            m: float(bundle["geweke_metric"][m]) for m in geweke_metric_jnp
        }
        geweke_metric_ci = {
            m: (
                float(bundle["geweke_metric_ci"][m][0]),
                float(bundle["geweke_metric_ci"][m][1]),
            )
            for m in geweke_metric_jnp
        }
        geweke_x_ci = (float(bundle["geweke_x_ci"][0]), float(bundle["geweke_x_ci"][1]))

        # When R-hat never triggered, anchor the dot to the curve's last point.
        if not rhat_triggered:
            for m in metric_names:
                rhat_metric_values[m] = float(diag["metric_curves"][m][-1])

        return {
            "gamma": float(gamma),
            "t_grid": t_grid,
            "metric_curves": diag["metric_curves"],
            "metric_ci_low": diag["metric_ci_low"],
            "metric_ci_high": diag["metric_ci_high"],
            "rhat_rank_split": diag["rhat_rank_split"],
            "rhat_split": diag["rhat_split"],
            "rhat_classical": diag["rhat_classical"],
            "geweke_max_abs_curve": diag["geweke_max_abs_curve"],
            "geweke_frac_stopped_curve": diag["geweke_frac_stopped_curve"],
            "rhat_triggered": rhat_triggered,
            "rhat_status_str": rhat_status_str,
            "t_stop_rhat": t_stop_rhat,
            "rhat_x_for_plot": t_stop_rhat,
            "rhat_metric_values": rhat_metric_values,
            "rhat_metric_ci": rhat_metric_ci,
            "rhat_x_ci": rhat_x_ci,
            "rhat_visualise": rhat_visualise,
            "geweke_frac_stopped": frac_stopped,
            "geweke_stop_mean": geweke_stop_mean,
            "geweke_stop_median": geweke_stop_median,
            "geweke_x_for_plot": geweke_x_for_plot,
            "geweke_x_ci": geweke_x_ci,
            "geweke_metric_values": geweke_metric_values,
            "geweke_metric_ci": geweke_metric_ci,
            "geweke_visualise": geweke_visualise,
            "samples_full": samples_full,
            "trajectories": np.asarray(trajectories),
        }

    def _prefix(base, gamma):
        """Return ``gamma_<g>/<base>`` in multi-gamma mode, else ``base``."""
        if not multi_gamma or gamma is None:
            return base
        return f"gamma_{_format_gamma(gamma)}/{base}"

    def _populate_logger(gamma, data):
        g = gamma if multi_gamma else None

        # Per-gamma curves (numerical curves over chain length).
        logger.setdefault(_prefix("curves/t_grid", g), []).append(data["t_grid"])
        for m in metric_names:
            curve_key = _prefix(f"curves/{m}", g)
            logger.setdefault(curve_key, []).append(data["metric_curves"][m])
            curve_keys.add(curve_key)
            logger.setdefault(_prefix(f"curves/{m}_ci_low", g), []).append(
                data["metric_ci_low"][m]
            )
            logger.setdefault(_prefix(f"curves/{m}_ci_high", g), []).append(
                data["metric_ci_high"][m]
            )
        rhat_curve_arrays = {
            "curves/rhat_max_rank_split": data["rhat_rank_split"],
            "curves/rhat_max_split": data["rhat_split"],
            "curves/rhat_max_classical": data["rhat_classical"],
        }
        for base, value in rhat_curve_arrays.items():
            key = _prefix(base, g)
            logger.setdefault(key, []).append(value)
            curve_keys.add(key)

        # R-hat scalars.
        rhat_step_value = (
            float(data["t_stop_rhat"]) if data["rhat_triggered"] else float("nan")
        )
        logger.setdefault(_prefix("diag/rhat_stop_step", g), []).append(rhat_step_value)
        logger.setdefault(_prefix("diag/rhat_status", g), []).append(
            data["rhat_status_str"]
        )
        logger.setdefault(_prefix("diag/rhat_stop_step_ci_low", g), []).append(
            data["rhat_x_ci"][0]
        )
        logger.setdefault(_prefix("diag/rhat_stop_step_ci_high", g), []).append(
            data["rhat_x_ci"][1]
        )
        for m in metric_names:
            v = (
                data["rhat_metric_values"][m]
                if data["rhat_triggered"]
                else float("nan")
            )
            logger.setdefault(_prefix(f"discrepancies_rhat/{m}", g), []).append(v)
        if data["rhat_visualise"] is not None:
            logger.update({_prefix(k, g): v for k, v in data["rhat_visualise"].items()})

        # Geweke scalars.
        logger.setdefault(_prefix("diag/geweke_frac_early_stopped", g), []).append(
            data["geweke_frac_stopped"]
        )
        logger.setdefault(_prefix("diag/geweke_stop_step_mean", g), []).append(
            data["geweke_stop_mean"]
        )
        logger.setdefault(_prefix("diag/geweke_stop_step_median", g), []).append(
            data["geweke_stop_median"]
        )
        logger.setdefault(_prefix("diag/geweke_stop_step_ci_low", g), []).append(
            data["geweke_x_ci"][0]
        )
        logger.setdefault(_prefix("diag/geweke_stop_step_ci_high", g), []).append(
            data["geweke_x_ci"][1]
        )
        for m in metric_names:
            v = data["geweke_metric_values"].get(m, float("nan"))
            logger.setdefault(
                _prefix(f"discrepancies_geweke_partial/{m}", g), []
            ).append(v)
        if data["geweke_visualise"] is not None:
            logger.update(
                {_prefix(k, g): v for k, v in data["geweke_visualise"].items()}
            )

        # Full-chain visualisation.
        logger.update(
            {
                _prefix(k, g): v
                for k, v in target.visualise(
                    samples=data["samples_full"], prefix="full"
                ).items()
            }
        )

    def _save_seed_diagnostics(seed_idx, seed_value, per_gamma_results):
        out_dir = os.path.join(os.getcwd(), "diagnostics_data")
        os.makedirs(out_dir, exist_ok=True)
        env_name = str(getattr(cfg.target, "name", "unknown_env"))
        dim_value = int(getattr(cfg.target, "dim", dim))
        out_path = os.path.join(
            out_dir,
            f"baseline_diagnostics_{env_name}_{dim_value}D_seed_{seed_value}.npz",
        )
        flat = {
            "seed": np.asarray(seed_value, dtype=int),
            "seed_index": np.asarray(seed_idx, dtype=int),
        }
        for result in per_gamma_results:
            g_prefix = f"gamma_{_format_gamma(result['gamma'])}"
            flat[f"{g_prefix}/t_grid"] = np.asarray(result["t_grid"], dtype=int)
            flat[f"{g_prefix}/chains"] = np.asarray(result["trajectories"])
            flat[f"{g_prefix}/rhat_classical"] = np.asarray(
                result["rhat_classical"], dtype=float
            )
            flat[f"{g_prefix}/geweke_max_abs"] = np.asarray(
                result["geweke_max_abs_curve"], dtype=float
            )
            flat[f"{g_prefix}/geweke_frac_early_stopped"] = np.asarray(
                result["geweke_frac_stopped_curve"], dtype=float
            )
        np.savez_compressed(out_path, **flat)

    for seed_idx in range(n_eval_seeds):
        seed_value = int(cfg.seed + seed_idx)
        key_gen = jax.random.PRNGKey(seed_value)

        # Initialize the model per-seed to keep runs independent.
        key, key_gen = jax.random.split(key_gen)
        model_state = init_model_non_acyclic(key, dim, alg_cfg)

        logger = {
            "stats/step": [],
            "stats/wallclock": [],
            "stats/nfe": [],
        }
        curve_keys = set()

        ### Multi-gamma orchestration: run eval once per gamma, aggregate, plot.
        key, key_gen = jax.random.split(key_gen)
        logger["stats/step"].append(0)
        logger["stats/nfe"].append(batch_size)
        per_gamma_results = []
        for gamma in gammas:
            k, key_gen = jax.random.split(key_gen)
            result = _eval_compute(model_state, k, gamma)
            per_gamma_results.append(result)
            _populate_logger(gamma, result)

        _save_seed_diagnostics(seed_idx, seed_value, per_gamma_results)

        # One overlay figure per metric, drawn from all gammas at once.
        for m in metric_names:
            fig_m = _plot_metric_curves_overlay(m, gammas, per_gamma_results)
            logger[f"figures/metric_{m}"] = [wandb.Image(fig_m)]
            plt.close(fig_m)
        plt.close("all")

        last_entry = extract_last_entry(logger)
        if not cfg.use_cometml:
            continue

        # Curves share the same x-grid per gamma; pick the right `t_grid`
        # by stripping the optional `gamma_<g>/` prefix and reusing the
        # corresponding entry from `last_entry`.
        def _t_grid_for(curve_key):
            if "/" in curve_key and curve_key.startswith("gamma_"):
                prefix = curve_key.split("/", 1)[0] + "/"
                t_key = prefix + "curves/t_grid"
            else:
                t_key = "curves/t_grid"
            return np.asarray(last_entry.get(t_key, [])).tolist()

        metrics = {}
        for k, value in last_entry.items():
            if isinstance(value, wandb.Image):
                exp.log_image(value.image, name=f"seed_{seed_idx}/{k}", step=seed_idx)
            elif k.endswith("curves/t_grid"):
                continue
            elif k in curve_keys:
                try:
                    exp.log_curve(
                        name=f"seed_{seed_idx}/{k}",
                        x=_t_grid_for(k),
                        y=np.asarray(value).tolist(),
                        step=seed_idx,
                    )
                except Exception:
                    pass
            else:
                metrics[k] = value
        exp.log_metrics(
            {f"seed_{seed_idx}/{k}": v for k, v in metrics.items()}, step=seed_idx
        )
