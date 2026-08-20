# Benchmark results

The CSV file stores measurements from the local Bitcoin test data.

The baseline rows use 50 history observations, a 20-observation horizon, stride 20, and seed 42.

The pilot row uses a 64-dimensional model with two Transformer layers.
It trains for five epochs with 128 samples per epoch.
It uses 20 forecast samples and 10 reverse-SDE steps.

The pilot result is a negative result. Its CRPS is worse than every baseline.
The pilot has excessive interval coverage and requires more training and ablation work.

The guidance ablation tests weights 0.00, 0.25, and 1.00.
All three settings produce the same pilot metrics at the recorded precision.
This result indicates that the short pilot does not learn useful conditional guidance.
