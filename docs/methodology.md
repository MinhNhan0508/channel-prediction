# Methodology

## Prediction task

The input is a sequence of 10 observed effective-channel matrices. Each matrix
contains 8 RF chains × 4 receive antennas, represented by real and imaginary
components. The model predicts 5 future matrices in one forward pass.

The simulation has one UE with four receive antennas, not four independently
simulated users. Channels are generated with Sionna CDL-A at 30 GHz, with a
0.125 ms sampling interval and one subcarrier. These are synthetic channels,
not measurements from a deployed radio system.

## Observations and targets

For each sequence, a DFT-codebook combiner is selected using the mean physical
channel over the **historical frames only**. It remains fixed over the observed
and predicted window. Beam selection assumes clean historical physical-channel
pilots; this is an idealization, not an implemented noisy-pilot estimator.

AWGN is added to the effective-channel history. Its power is calibrated using
only that history. Targets are the clean future effective channels under the
same combiner. Changing future channel values must not change the combiner or
the noisy input when the random seed is held fixed; a regression test checks
this property.

Train, validation and test splits use separate generation-seed offsets. A
single maximum-absolute-component normalization scale is fitted on training
samples only and reused for validation, test and inference.

The manifest records the observation policy, configuration, source fingerprints,
library versions and a fingerprint of the generated NPZ files. Data from a
different policy must not be silently reused by the training suite.

## DAST

The neural branch projects 64 real-valued features per frame into a 96-dimensional
representation. An FFT-derived temporal profile controls a feature-wise affine
modulation (FiLM). A two-layer Transformer encoder and two-layer decoder predict
all five future frames using learned query tokens.

A parallel classical branch fits a regularized order-4 autoregression to each
complex effective-channel coefficient and extrapolates it. A learned gate uses
FFT-derived features to blend the two predictions for each future frame:

`prediction = gate × AR_prediction + (1 − gate) × Transformer_prediction`

The gate is a soft mixture, not a hard switch selecting one model. Its inputs
are learned temporal features; no explicit ground-truth velocity is supplied
to the model.

The thesis calls the classical branch a PAD prior, and internal identifiers
retain that name for compatibility. This implementation is a coefficient-wise
AR/least-squares predictor, inspired by classical channel prediction. It is
not a full reproduction of angle-delay-domain PAD. See the
[original PAD paper](https://arxiv.org/abs/1912.11330) for that formulation.

The complete maintained DAST configuration has 406,677 trainable parameters.
The classical fit is computed from each input sequence and is not trained by
backpropagation; gradients update the neural branch and blending gate.

## Metric

For each sample and future frame, NMSE is the squared prediction-error norm
divided by the squared target norm. These linear ratios are averaged over
samples and future frames, then converted with `10 × log10(mean NMSE)`.
Lower dB values are better. Zero dB means a ratio of one; it does not define
the score of a hold-last-observation baseline.

Across training seeds, report the mean and population standard deviation of
each seed's final dB score. Three seeds characterize initialization variability
for this experiment, not universal statistical significance.

## Limits

- Single-subcarrier CDL-A simulation does not validate wideband deployment,
  delay-domain prediction or performance on measured channels.
- The combiner uses ideal historical physical-channel information.
- The experiments study one dataset generation configuration; results do not
  establish robustness to every channel distribution or receiver design.
- A vanilla Transformer baseline is implemented; it is not a reproduction of
  every hybrid-attention mechanism in the reference literature.
- Downstream BER or spectral-efficiency results require their own receiver,
  modulation and equalization assumptions. NMSE alone is not a deployment claim.

## Related work and software

- [Channel Prediction via Hybrid-Attention Parallel Synchronous Transformer](https://globals.ieice.org/en_transactions/communications/10.23919/transcom.2025EBP3052/_f),
  IEICE Transactions on Communications, E109-B(3), 466–478, 2026.
- [Addressing the Curse of Mobility in Massive MIMO with Prony-Based Angular-Delay Domain Channel Predictions](https://arxiv.org/abs/1912.11330).
- [Sionna](https://github.com/NVlabs/sionna) supplies the channel simulator;
  [PyTorch](https://pytorch.org/) supplies the neural-network framework.

These sources motivate components and simulation choices. This project does
not claim their reported benchmark numbers as its own results.
