import pickle
from typing import List

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as pydist
from jax._src.flatten_util import ravel_pytree
import wandb

from targets.base_target import Target
from utils.path_utils import project_path


def pad_with_const(X):
    extra = np.ones((X.shape[0], 1))
    return np.hstack([extra, X])


def standardize_and_pad(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    return pad_with_const(X)


def load_model_sonar():
    def model(Y):
        w = numpyro.sample("weights", pydist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = jnp.dot(X, w)
        with numpyro.plate("J", n_data):
            y = numpyro.sample("y", pydist.BernoulliLogits(logits), obs=Y)

    with open(project_path("targets/data/sonar_full.pkl"), "rb") as f:
        X, Y = pickle.load(f)

    Y = (Y + 1) // 2
    X = standardize_and_pad(X)

    dim = X.shape[1]
    n_data = X.shape[0]
    model_args = (Y,)
    return model, model_args


class Sonar(Target):
    def __init__(
        self, dim=61, log_Z=None, can_sample=False, sample_bounds=None
    ) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.data_ndim = dim

        rng_key = jax.random.PRNGKey(1)
        model, model_args = load_model_sonar()
        model_param_info, potential_fn, constrain_fn, _ = (
            numpyro.infer.util.initialize_model(rng_key, model, model_args=model_args)
        )
        params_flat, unflattener = ravel_pytree(model_param_info[0])
        self.log_prob_model = lambda z: -1.0 * potential_fn(unflattener(z))
        self._plot_bound = 10.0

    def get_dim(self):
        return self.dim

    def log_prob(self, x: chex.Array):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        # log prob model can only handle unbatched input
        log_probs = jax.vmap(self.log_prob_model)(x)

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def visualise(
        self, samples: chex.Array = None, axes=None, show=False, prefix=""
    ) -> dict:
        projection_pairs = ((0, 1), (0, 3), (2, 1), (2, 3))
        plotting_bounds = (-float(self._plot_bound), float(self._plot_bound))
        grid_width_n_points = 100

        fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex="row", sharey="row")
        if samples is not None:
            samples = jnp.clip(samples, min=plotting_bounds[0], max=plotting_bounds[1])

        x_grid, y_grid = jnp.meshgrid(
            jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
            jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
        )
        x_points = jnp.column_stack([x_grid.ravel(), y_grid.ravel()])

        for ax, (x_dim, y_dim) in zip(axs.ravel(), projection_pairs):
            x = jnp.zeros((x_points.shape[0], self.dim))
            x = x.at[:, x_dim].set(x_points[:, 0])
            x = x.at[:, y_dim].set(x_points[:, 1])
            log_probs = self.log_prob(x)
            log_probs = jnp.clip(log_probs, min=-1000, max=None).reshape(
                (grid_width_n_points, grid_width_n_points)
            )

            contour = ax.contourf(x_grid, y_grid, log_probs, levels=20)
            fig.colorbar(contour, ax=ax)

            if samples is not None:
                ax.plot(samples[:, x_dim], samples[:, y_dim], "x", c="r", alpha=0.5)
            ax.set_xlabel(f"x{x_dim + 1}")
            ax.set_ylabel(f"x{y_dim + 1}")

        plt.tight_layout()
        wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None


if __name__ == "__main__":
    sonar = Sonar()
    sonar.visualise(None, show=True)
