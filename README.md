# rl-insulin-pump

Creating RL Agents for simglucose/PadovaT1D simulator.

## Getting Started

### Setup

1. Set up virtual environment

	``` bash
	python -m venv env
	source env/bin/activate  # On Windows: env\Scripts\activate
	```

2. Install dependencies

	``` bash
	pip install -r requirements.txt
	```

3. Install simglucose in editable mode:

	``` bash
	cd simglucose
	pip install -e . --config-settings editable_mode=compat
	cd ..
	```

	Installing simglucose in editable mode within the project directory allows for direct imports and potential modifications to the source code. This gives us much greater control for customization and enables effective debugging.

4. Run scripts from root dir
	``` bash
	python -m src.main
	```

### Run Tests

We use `pytest` for unit testing. Tests should be stored in the `tests` directory and should test basic functionality of any new helper functions added to the codebase. To run all tests:

```bash
cd tests
pytest -v --color=yes
```

### Lint and format

We use `pre-commit` with `ruff` to check formatting and linting before commits are pushed. You can check this before making commits to the repository with the following:

```bash
pre-commit run --all-files
```

## Project Directories

```bash
.
├── .github/                          # GitHub workflows for CI/CD tests
├── configs/                          # Training and evaluation configuration YAMLs
├── experiments/                      # Scripts and setups for running controlled experiments
├── results/                          # Output from experiments: plots, logs, metrics, etc.
├── scripts/                          # Shell scripts for launching jobs (e.g. on WatGPU)
├── simglucose/                       # Custom fork of simglucose package
├── src/                              # Main RL framework package
│   ├── agents/                       # Code for creating, loading, and managing agents
│   ├── environments/				  # Custom Gym environments and reward functions
│   ├── training/                     # Orchestration of training, prediction, and evaluation
│   ├── utils/                        # Generic helper utilities
│   ├── __init__.py
│   └── main.py                       # Entry point for training/evaluation
├── tests/                            # Unit and integration tests
├── .gitignore
├── .pre-commit-config.yaml           # Pre-commit checks configuration
├── README.md
├── requirements.txt                  # Python package dependencies
└── setup.py                          # Install script to register `src` as a module
```

## Managing the `simglucose` Fork

This repository includes `simglucose` as a git subtree. If you need to update or push changes to the [simglucose fork](https://github.com/Blood-Glucose-Control/simglucose.git), use the following commands **in the root directory**.

### Pulling Updates from the Subtree
If the upstream `simglucose` repository has updates, you can pull them into the subtree:
```bash
git subtree pull --prefix=simglucose https://github.com/Blood-Glucose-Control/simglucose.git master --squash
```

### Pushing Changes to the Subtree

To push changes made to `simglucose` back to the forked repository:

```bash
git subtree push --prefix=simglucose https://github.com/Blood-Glucose-Control/simglucose.git master
```
