## WebShop - WebArena (RAGEN)

Professional README for the WebShop-WebArena RAGEN project.

---

## Overview

This repository contains research and experimental code for training and evaluating agents that operate in two related environments: a WebShop e-commerce simulation and a WebArena navigation/interaction benchmark. The project includes training scripts, environments, agent implementations, evaluation tools, pretrained checkpoints, and experiment configuration files.

Key highlights:
- Implementations of environment wrappers and agent interfaces for WebShop and WebArena.
- Training scripts and example experiment configs to reproduce reported results.
- Evaluation and failure-analysis utilities.
- Pretrained model checkpoints (PyTorch `.pt`) for quick evaluation.

This README gives a concise starting guide for developers and researchers who want to run, train, or evaluate agents in this codebase.

## Repository layout

Top-level layout (important files/folders):

- `main.py` — project entry / example runner (check for local CLI usage).
- `train_complete.py`, `train_webarena.py`, `train_webshop.py` — training scripts.
- `agents/` — agent implementations (e.g., `webshop_agent.py`, baseline agents).
- `environments/` — environment wrappers for `webarena` and `webshop` (`webarena_env.py`, `webshop_env.py`).
- `models/` — model definitions (e.g., `transformer.py`).
- `data/` — sample data for `webarena` and `webshop` (e.g., tasks and product samples).
- `evaluation/` — evaluation tools and metrics (`evaluate_webarena.py`, `metrics.py`, `failure_analysis.py`).
- `training/` — training configs and helper scripts (`config.yaml`, `train_webshop.py`).
- `checkpoints/` — saved model checkpoints (`*.pt`) used for evaluation and fine-tuning.
- `experiments/` — experiment configuration files and outputs.
- `outputs/`, `logs/` — experiment output and logs.
- `tests/` — basic test suite (`test_basic.py`).

## Requirements

- Python 3.8+ recommended.
- The repository includes a `setup.py`; preferred way to install dependencies is editable install (it will install required packages listed there).

Create and activate a virtualenv, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# If you prefer a requirements file, create one from setup metadata or pip freeze from a working env.
```

If your workloads use GPU (PyTorch), ensure you have a compatible CUDA toolkit and install the appropriate `torch` build (the `setup.py` may already pin a suitable `torch` version).

## Quick start

Run the main script or a training script. Exact CLI options may vary — inspect the top of the target script to see available flags.

Example: run a default training or demo (replace with appropriate CLI flags as needed):

```bash
# Run an example / entrypoint (check script for flags)
python main.py

# Run WebArena-specific training (example script shipped in repo root)
python train_webarena.py

# Run WebShop training
python training/train_webshop.py
```

Run evaluation for WebArena:

```bash
python evaluation/evaluate_webarena.py
```

If you want to run tests:

```bash
pytest -q
```

## Checkpoints and model files

Pretrained model files are stored under `checkpoints/`. Examples:

- `ragen_ep150_success20.pt`
- `webarena_best.pt` (may vary)

To evaluate with a checkpoint, pass its path to the appropriate evaluation or inference script. The scripts typically accept a `--checkpoint` or similar flag — check the top of the script for concrete CLI flags.

## Experiments

Experiment configs live under `experiments/configs/` (for example `experiment_config.json`). The `experiments/outputs/` folder contains sample outputs and results from completed runs.

To reproduce an experiment you should:

1. Review and adjust the experiment configuration JSON/YAML.
2. Ensure the `checkpoints/` and `data/` paths referenced in the config file exist.
3. Run the corresponding training script or experiment launcher.

## Evaluation and metrics

Evaluation code is in `evaluation/`.
- `evaluate_webarena.py` — runs evaluation for the WebArena tasks and writes results to `outputs/` or `experiments/results/`.
- `metrics.py` — contains metric computations used during evaluation.

Use these scripts to compute success rates and to generate failure analysis reports.

## Development notes

- Code is organized to separate environment code (`environments/`), agents (`agents/`), models (`models/`), and training/evaluation (`training/`, `evaluation/`).
- If you move files (for example into a single `webarena/` folder), you'll likely need to update `import` statements and config file paths.
- Tests live under `tests/`; add unit tests for any new behavior.

### Recommended workflow

1. Create a new branch for your feature/experiment.
2. Create or update experiment config files under `experiments/configs/`.
3. Run training locally on a small sample or with reduced iterations for quick iteration.
4. Run full-scale training on a machine with GPU resources.
5. Save checkpoints to `checkpoints/` and results to `experiments/outputs/`.

## Contributing

Contributions are welcome. Please follow standard GitHub etiquette:

1. Fork the repository.
2. Create a feature branch.
3. Add tests where appropriate.
4. Open a pull request describing the change and why it's needed.

If you add new dependencies, update `setup.py` and document the change in this README.

## Troubleshooting

- If imports fail after moving files, run a quick search for the old package path and update `from ... import ...` lines.
- If a training script cannot find data, verify `data/` paths in the config file and script parameters.
- For CUDA / device errors, ensure your local `torch` install matches the CUDA version on your machine.

## License & Authors

Add your preferred license text or a `LICENSE` file at the repo root. If this is a private/research repo, state that clearly.

Maintainers: see the project owner / repository owner for contact details.

---

If you'd like, I can:

- add badges (build/status, license) to the top of this README.
- generate a succinct `requirements.txt` from `setup.py` for reproducible installs.
- create a short `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

Tell me which of the above you'd like next and I will implement it.
