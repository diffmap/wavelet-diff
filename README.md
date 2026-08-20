# Wavelet Diff

This project tests a wavelet-conditioned score transformer for probabilistic time-series forecasting.

## Project layout

- `src/model/` contains the model, SDE, training, sampling, and data-loading code.
- `src/benchmarks/` contains exploratory baseline scripts.
- `data/` contains local raw, processed, and wavelet data.
- `outputs/` contains local checkpoints, plots, and prediction runs.
- `Documentation/` contains the current model notes.
- `tests/` contains fast shape and data-contract tests.

Large data and model outputs are local artifacts. The repository ignores them.

## Environment

Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate wavelet-diff
```

Install the package in editable mode. This step lets `src.model` imports resolve
without path hacks, and it lets you run modules with `python -m`.

```bash
pip install -e .
```

Run the checks with:

```bash
ruff check src data tests
pytest -q
```

## Data workflow

Run commands from the repository root. Run `data/data_cleaner.py` to create the train and test CSV files.

Run the wavelet conversion notebook to create the `.pt` tensors. The notebook still needs conversion to a script.

## Model workflow

Run all model modules with `python -m`, from the repository root, so package-relative
imports resolve correctly.

Run `python -m src.model.trainer` to train the score transformer. Run
`python -m src.model.predictandsave` to sample forecasts.

The current pipeline reconstructs returns from the approximation band only. This is a known limitation.

## Current limitations

- The project has no validated inverse SWT reconstruction yet.
- Training and sampling use configuration values in `src/model/config.py`.
- Checkpoints and prediction runs require local data that is not stored in Git.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
