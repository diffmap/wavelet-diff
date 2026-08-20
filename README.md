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
`python -m src.model.predictandsave --checkpoint outputs/checkpoints/model.pth`
to sample forecasts. The sampler accepts sample counts, window counts, diffusion
steps, and a seed from the command line.

Run the bounded CPU pilot with:

```bash
python -m src.model.trainer --config configs/current_cpu_pilot.json
```

The evaluation module uses rolling origins. Each forecast function receives only
the `history_len` observations before an origin and returns samples with shape
`[n_samples, predict_len]`. The evaluator reports MAE, RMSE, CRPS, and 50% and
90% interval coverage.

Run the baseline benchmark with:

```bash
python -m src.benchmarks.rolling_baselines \
  --input data/Testing\ Data/bitcoin_raw_prices_test.csv \
  --output outputs/baseline_metrics.csv
```

Run a compatible model checkpoint with the same rolling-origin metrics:

```bash
python -m src.benchmarks.model_rolling_evaluation \
  --checkpoint outputs/checkpoints/current_model.pth \
  --output outputs/model_metrics.csv
```

The command also writes a per-origin metrics file beside the aggregate file.

The pipeline reconstructs each sampled future from all SWT bands with the inverse SWT.

### Baseline result

The following result uses the local test prices, 50 history observations, a
20-observation horizon, stride 20, 100 bootstrap samples, and seed 42.

| Baseline | MAE | RMSE | CRPS | 50% coverage | 90% coverage | Origins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero return | 0.01819 | 0.02508 | 0.01819 | 0.000 | 0.000 | 51 |
| Last return | 0.02473 | 0.03113 | 0.02473 | 0.000 | 0.000 | 51 |
| Historical bootstrap | 0.01841 | 0.02530 | 0.01413 | 0.485 | 0.864 | 51 |

These are baseline measurements. They are not model measurements.

The current-code CPU pilot is also measured across the same 51 origins.
It uses a 64-dimensional model, two Transformer layers, five epochs, 20 samples,
and 10 reverse-SDE steps.

| Experiment | MAE | RMSE | CRPS | 50% coverage | 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current CPU pilot | 0.26317 | 0.33103 | 0.30904 | 0.943 | 1.000 |

The pilot is a negative result. Its CRPS is worse than the historical bootstrap baseline.
See [`results/benchmark_metrics.csv`](results/benchmark_metrics.csv).

Run the guidance ablation with:

```bash
python -m src.benchmarks.model_guidance_ablation \
  --checkpoint outputs/checkpoints/current_cpu_pilot/score_transformer_pilot_ep5.pth \
  --output outputs/current_cpu_pilot_guidance.csv
```

The pilot produced identical metrics for guidance weights 0.00, 0.25, and 1.00.
This negative ablation result indicates that five training epochs do not learn useful guidance.

## Current limitations

- Training and sampling use configuration values in `src/model/config.py`.
- Checkpoints and prediction runs require local data that is not stored in Git.
- The baseline and model rolling-origin commands write separate metric files.
- Checkpoints from the historical model implementation are incompatible with the current model class. Retrain them with the current source code.
- A full model quality table requires a current-code checkpoint. The repository has only historical checkpoints, so the model command currently serves as a reproducible evaluation path.

Run the protocol ablation with:

```bash
python -m src.benchmarks.baseline_ablation \
  --input data/Testing\ Data/bitcoin_raw_prices_test.csv \
  --output outputs/baseline_ablation.csv
```

This ablation tests history lengths of 20, 50, and 100 observations.
It tests forecast horizons of 5 and 20 observations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
