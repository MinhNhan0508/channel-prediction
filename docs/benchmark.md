# Model-family comparison and component ablations

These answer different questions. The model-family comparison evaluates DAST,
Transformer, LSTM, GRU, RNN and CNN on identical held-out observations. The
component study compares DAST with fixed-gate, no-AR and no-FiLM variants.
Component ablations are not substitutes for independent model families.

## Learned baselines

The baseline extension adds four existing model configurations, each with seeds
42, 123 and 2024. It reuses the sealed history-only dataset and the original
training recipe: 300-epoch cap, validation-NMSE early stopping with patience 120,
batch size 64 and the same optimizer/loss schedule. No DAST/Transformer run is
retrained or selected again based on the extra baseline test scores.

This is an extension of the initial formal study. It is not described as a fully
preregistered matrix preceding every inspection of test results.

```sh
python tools/train_suite.py run --plan configs/experiments/baseline_v1.json
```

The suite automatically evaluates every completed run. All declared seeds are
required. The combined report checks that dataset identity, training recipe,
test scenarios and core model/data/trainer sources agree between suites.

## Classical references

All reference predictors see only the ten historical noisy effective-channel
matrices. They do not receive future targets, ground-truth velocity or true
observation SNR. Targets are used only to compute evaluation metrics.

| Reference | Definition |
|---|---|
| Hold last / No-Prediction | Repeat the last noisy observation for five forecast frames |
| AR(4), unbounded | Coefficient-wise complex ridge least-squares, ridge coefficient 0.001 |
| AR(4), bounded | Same fit with recursive magnitude capped at three times each coefficient's historical maximum |
| Complex Kalman | AR(4) transition, complex Hermitian covariance, Joseph covariance update and the same forecast magnitude cap |

The AR prior is not a full angular-delay-domain PAD reproduction. Including
both capped and uncapped AR separates the effect of the cap from the contribution
of the learned hybrid. The standalone AR reference uses double-precision solves;
the neural prior uses its original PyTorch complex arithmetic. Equivalence tests
allow the corresponding numerical tolerance.

Kalman transition coefficients are estimated from the historical window. Process
and observation noise powers are each half of the fitted AR residual power plus
1e-6 before splitting. These are declared heuristics, not known physical noise
parameters or settings optimized on the test set. Initial covariance is residual
power times the identity.

The benchmark calls `src/references.py`. The historical
`src/models/kalman.py` uses a real-covariance approximation; its old scores are
not attributed to the complex-covariance reference. The original thesis and
archived code are preserved separately.

Deterministic references run once on each test set. They are not copied into
three artificial seed runs and do not receive training-seed error bars.

```sh
python tools/evaluate_references.py --data data/sionna_causal_v1 --experiment runs/formal-v1/experiment.json --output runs/references
python tools/report_benchmark.py --formal runs/formal-v1/evaluation.json --baselines runs/baseline-v1/evaluation.json --references runs/references/manifest.json --output runs/model-comparison
```

The combined report requires all six learned models, all three seeds and all
five velocities. It recomputes reference scores from saved per-sequence/per-frame
ratios and rejects missing groups, duplicates or incompatible experiments.

## Reading the plots

Lower NMSE is better. Neural curves show mean and population standard deviation
of three seed-level dB scores; reference curves represent one deterministic
evaluation. Each score averages linear per-sample/per-frame NMSE over 500 test
sequences and five forecast frames before conversion to dB. Error bars are not
confidence intervals. Zero dB means error power equals target power and does not
define the No-Prediction baseline.

## Interrupted-run provenance

The RNN seed-123 run was resumed after epoch 212 when Windows refused an atomic
replacement of the status JSON file. The completed epoch checkpoint retained
optimizer, scheduler and RNG state. One logged but uncommitted epoch was archived
and replayed from that checkpoint, leaving a contiguous committed metric history.
A local adapter added bounded retries for transient file-sharing errors. The
sealed model/data/trainer source and experiment identity remained unchanged.
