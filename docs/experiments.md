# Formal experiment protocol

The `formal_causal_v1` matrix is fixed before the formal test evaluation.
Its machine-readable definition is `configs/experiments/formal_v1.json`.

## Dataset

| Setting | Value |
|---|---|
| Observation policy | `history_only_v1` |
| Training sequences | 8,000 |
| Validation sequences | 1,000 |
| Test sequences | 500 at each of 30, 50, 70, 90 and 120 km/h |
| Training velocity | Uniform over 10–120 km/h, sampled per generation batch |
| Training observation SNR | Uniform over 5–25 dB, sampled per generation batch |
| Test observation SNR | 20 dB |
| History / forecast frames | 10 / 5 |
| Physical BS antennas / RF chains / UE antennas | 64 / 8 / 4 |

Delay spread is configured over 25–250 ns in training and 50 ns for testing.
The one-subcarrier setup does not establish delay-domain or wideband robustness.
Training samples in the same generation batch share their velocity, delay-spread
and SNR settings. Manifest checks and full-dataset validation run before training.

## Matrix

Each variant uses seeds **42, 123 and 2024**, for 15 runs in total.

This initial matrix was followed by 12 runs for LSTM, GRU, RNN and CNN under
the same data/recipe. The complete study has 27 training runs. See the
[model-family benchmark](benchmark.md) and [complete model comparison](benchmark-results/README.md).

| Variant | FiLM encoder | Classical AR prior | Mixture |
|---|---|---|---|
| DAST | Enabled | Enabled | FFT-feature-dependent gate |
| Transformer | Not used | Not used | Neural prediction only |
| Fixed gate | Enabled | Enabled | Learned per-horizon blend independent of the input |
| No PAD | Enabled | Disabled | Neural prediction only |
| No FiLM | Disabled | Enabled | FFT-feature-dependent gate retained |

The **no FiLM** variant removes only feature modulation. It still uses
FFT-derived gate features and must not be described as removing all Doppler
information. The historical no-Doppler experiment changed more than one
component and is not reused for this matrix.

## Training

- AdamW, learning rate 0.001, weight decay 0.0001, batch size 64.
- MSE loss for 10 warmup epochs, then NMSE; gradient clipping at norm 1.
- Two-stage schedule, with a 0.1 learning-rate factor after approximately
  two thirds of the maximum training budget; initial step warmup is retained.
- Maximum 300 epochs and validation-NMSE early-stopping patience of 120 epochs.
- Same recipe for every variant. Best checkpoints are selected using validation
  NMSE, including during MSE warmup.
- Training data can be cached in RAM; this does not change sample values.

The suite stops on nonfinite metrics or gradients instead of silently omitting
an unsuccessful seed. All declared seeds must exist for aggregation.

## Resume and provenance

Checkpoints contain model, optimizer, scheduler, AMP state when used,
early-stopping state and random-generator states. Writes replace the previous
checkpoint atomically after the new file has been flushed. The last checkpoint
is saved even on an early-stopping epoch.

Resume requires the same dataset fingerprint, source identity, model configuration,
training recipe and seed. An epoch-boundary interruption/resume test matches
uninterrupted training on CPU. Bitwise equality is not promised across different
GPU hardware, drivers or library versions.

## Evaluation and reporting

After all training runs finish, evaluate the selected checkpoint from each run
on all five test scenarios. Do not change the recipe based on these test scores.
Save per-frame linear NMSE arrays, individual-seed scores and aggregate dB
mean/std. Missing seeds are errors, not an invitation to report a smaller group
under a three-seed heading.

All 15 runs and the 75 declared test groups have completed. See
[the full results](results/README.md) for scores, seed variability and component
ablations. Every selected checkpoint was checked against its validation history,
and all 187,500 saved per-sequence/per-horizon NMSE values were recomputed from
the selected weights. The recomputation matched the saved arrays exactly in the
recorded CUDA environment.

The smaller paired-policy pilot used a different data size and training budget;
it is not included in these tables. Historical thesis figures are not labelled
as results of the corrected observation policy.
