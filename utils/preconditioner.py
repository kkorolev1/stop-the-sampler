import chex
import jax
import jax.numpy as jnp


def logistic_log_posterior_grad(
    w: chex.Array, X: chex.Array, Y: chex.Array
) -> chex.Array:
    """Gradient of log p(w, Y | X) for a standard Gaussian prior."""
    probs = jax.nn.sigmoid(X @ w)
    return -w + X.T @ (Y - probs)


def negative_log_posterior_hessian(
    w: chex.Array,
    X: chex.Array,
) -> chex.Array:
    """
    Negative Hessian of the log posterior:
        H(w) = I + X^T diag(p_i(1-p_i)) X.
    """
    probs = jax.nn.sigmoid(X @ w)
    weights = probs * (1.0 - probs)

    identity = jnp.eye(X.shape[1], dtype=X.dtype)
    return identity + X.T @ (weights[:, None] * X)


def compute_laplace_preconditioner(
    X: chex.Array,
    Y: chex.Array,
    lamda: float = 1e-4,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Compute the MAP point and fixed Laplace preconditioner
        M = (H_MAP + lambda * I)^{-1}.
    """
    dim = X.shape[1]
    identity = jnp.eye(dim, dtype=X.dtype)
    w = jnp.zeros(dim, dtype=X.dtype)

    converged = False

    for _ in range(max_iters):
        gradient = logistic_log_posterior_grad(w, X, Y)
        precision = negative_log_posterior_hessian(w, X)

        # Newton step for maximizing the concave log posterior.
        step = jnp.linalg.solve(precision + lamda * identity, gradient)
        new_w = w + step

        relative_step = jnp.linalg.norm(step) / (1.0 + jnp.linalg.norm(w))
        w = new_w

        if float(relative_step) < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "MAP optimization did not converge. Increase map_max_iters " "or lamda."
        )

    precision = negative_log_posterior_hessian(w, X)
    regularized_precision = precision + lamda * identity

    preconditioner = jnp.linalg.solve(regularized_precision, identity)

    # Remove small numerical asymmetry.
    preconditioner = 0.5 * (preconditioner + preconditioner.T)

    preconditioner_cholesky = jnp.linalg.cholesky(preconditioner)

    return preconditioner, preconditioner_cholesky
