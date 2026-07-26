import chex
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from matplotlib import pyplot as plt

from targets.base_target import Target
from utils.path_utils import project_path


class GermanCredit(Target):
    def __init__(self, log_Z=None, can_sample=False, sample_bounds=None):
        super().__init__(dim=25, log_Z=log_Z, can_sample=can_sample)
        data = np.loadtxt(project_path("targets/data/german.data-numeric"))
        X = data[:, :-1]
        X /= jnp.std(X, 0)[jnp.newaxis, :]
        X = jnp.hstack((jnp.ones((len(X), 1)), X))
        self.data = jnp.array(X, dtype=jnp.float32)
        self.labels = data[:, -1] - 1
        self.num_dimensions = self.data.shape[1]
        self._prior_std_const = jnp.array(10.0, dtype=jnp.float32)
        self.prior_mean_const = jnp.array(0.0, dtype=jnp.float32)
        self.labels = jnp.array(jnp.expand_dims(self.labels.astype(jnp.float32), 1))
        self.const_term = jnp.array(0.5 * jnp.log(2.0 * jnp.pi), dtype=jnp.float32)
        self._plot_bound = 10.0
        samples = np.load(project_path("targets/data/german_credit10k.npy")).astype(
            np.float32
        )
        self.normalization_shift = 0.0
        self.normalization_shift = self.log_prob(samples.mean(axis=0))

    def log_prob(self, x: chex.Array) -> chex.Array:
        def _log_prob(x: chex.Array):
            features = -jnp.matmul(self.data, x.transpose())
            log_likelihood = jnp.sum(
                jnp.where(
                    self.labels == 1,
                    jax.nn.log_sigmoid(features),
                    jax.nn.log_sigmoid(features) - features,
                ),
                axis=0,
            )
            log_posterior = log_likelihood - self.normalization_shift
            return log_posterior

        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        log_probs = _log_prob(x)

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
                ax.plot(samples[:, x_dim], samples[:, y_dim], "o", alpha=0.5)
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
        ground_truth_samples = np.load(
            project_path("targets/data/german_credit10k.npy")
        )

        indices = jax.random.choice(
            seed, ground_truth_samples.shape[0], shape=sample_shape, replace=False
        )
        # Use the generated indices to select the subset
        return jnp.array(ground_truth_samples[indices])


if __name__ == "__main__":
    germanCredit = GermanCredit()

    key = jax.random.PRNGKey(42)

    num_samples = 100_000
    dim = 25
    samples = np.load(project_path("targets/data/german_credit10k.npy")).astype(
        np.float32
    )
    print("shape:", samples.shape)
    print(samples.min(), samples.mean(), samples.std(), samples.max())
    log_probs = germanCredit.log_prob(samples)

    # Compute statistics: min, mean, max
    min_log_prob = jnp.min(log_probs)
    mean_log_prob = jnp.mean(log_probs)
    max_log_prob = jnp.max(log_probs)
    std_log_prob = jnp.std(log_probs)

    print(f"min:  {float(min_log_prob):.4f}")
    print(f"mean: {float(mean_log_prob):.4f}")
    print(f"std: {float(std_log_prob):.4f}")
    print(f"max:  {float(max_log_prob):.4f}")
    print(germanCredit.normalization_shift)

    germanCredit.visualise(None, show=True)
