# Title

This repository is based on the [code](https://github.com/DenisBless/variational_sampling_methods) of Blessing et al. (2024) from their paper "[Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423)".

## Installation
- python 3.10.14
- jax 0.6.2

We recommend using the conda (or mamba) environment to install the dependencies.
```bash
conda create -n gfn-smc-jax python=3.10.14
conda activate gfn-smc-jax
```

Install the jax and jaxlib with the appropriate CUDA version or TPU support, e.g., cuda12
```bash
pip install -U "jax[cuda12]==0.6.2"
```

Install the other dependencies.
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python run.py algorithm=<algorithm_name> target=<target_name>
```

`<algorithm_name>` can be one of the following:
- `gfn_non_acyclic_baseline` (for ULA)
- `gfn_non_acyclic` (for the main algorithm (section 3.3))
- `gfn_non_acyclic_ml` (for the multilevel scheme (section 3.5))

`target_name` can be one of the following:
- `gaussian_mixture9`
- `funnel`
- `many_well`
