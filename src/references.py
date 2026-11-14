"""History-only classical reference predictors with explicit numerical policies."""
import numpy as np


def _series(history):
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 5 or history.shape[1:] != (10, 8, 4, 2) or not len(history):
        raise ValueError('Expected (batch, 10, 8, 4, 2) history.')
    if not np.isfinite(history).all():
        raise ValueError('History must be finite.')
    z = history[..., 0].astype(np.float64) + 1j * history[..., 1].astype(np.float64)
    return z.transpose(0, 2, 3, 1).reshape(-1, 10), len(history)


def _output(prediction, count):
    z = prediction.reshape(count, 8, 4, 5).transpose(0, 3, 1, 2)
    result = np.stack([z.real, z.imag], axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite classical prediction.')
    return result


def _fit_ar(series):
    order, samples = 4, series.shape[1] - 4
    design = np.stack([series[:, 3-k:3-k+samples] for k in range(order)], axis=-1)
    target = series[:, order:]
    adjoint = design.conj().transpose(0, 2, 1)
    gram = adjoint @ design
    regularization = 1e-3 * np.maximum(np.abs(gram).mean((1, 2)), 1e-12)
    coefficients = np.linalg.solve(gram + regularization[:, None, None] * np.eye(order),
                                   (adjoint @ target[..., None]))[..., 0]
    return coefficients, target - np.einsum('smp,sp->sm', design, coefficients)


def hold_last(history):
    _series(history)
    return np.repeat(np.asarray(history, dtype=np.float32)[:, -1:], 5, axis=1)


def ar_predict(history, bounded=False):
    """Coefficient-wise AR(4), ridge 1e-3; optional recursive 3x-history cap."""
    series, count = _series(history)
    coefficients, _ = _fit_ar(series)
    buffer = series[:, -4:].copy()
    limit = 3 * np.abs(series).max(axis=1)
    predictions = []
    for _ in range(5):
        value = np.sum(coefficients * buffer[:, ::-1], axis=1)
        if bounded:
            value *= np.minimum(1., limit / np.maximum(np.abs(value), 1e-12))
        predictions.append(value)
        buffer = np.concatenate([buffer[:, 1:], value[:, None]], axis=1)
    return _output(np.stack(predictions, axis=1), count)


def kalman_predict(history):
    """Complex AR(4) Kalman filter with Hermitian covariance and Joseph update.

    Coefficients and innovation power are estimated from historical observations.
    Q=R=half innovation power, with a 1e-6 floor before splitting. This is a
    declared heuristic, not a filter calibrated with true channel/noise parameters.
    Forecast state heads are capped at 3x the historical maximum magnitude.
    """
    series, count = _series(history)
    coefficients, residual = _fit_ar(series)
    n, order = len(series), 4
    innovation = (np.abs(residual) ** 2).mean(axis=1) + 1e-6
    process_power = observation_power = .5 * innovation
    transition = np.zeros((n, order, order), dtype=np.complex128)
    transition[:, 0] = coefficients
    transition[:, 1:, :-1] = np.eye(order - 1)
    adjoint = transition.conj().transpose(0, 2, 1)
    eye = np.broadcast_to(np.eye(order), (n, order, order))
    process_noise = np.zeros_like(transition)
    process_noise[:, 0, 0] = process_power
    state = series[:, 3::-1].copy()
    covariance = innovation[:, None, None] * eye
    for frame in range(4, 10):
        state = (transition @ state[..., None])[..., 0]
        predicted = transition @ covariance @ adjoint + process_noise
        denominator = predicted[:, 0, 0].real + observation_power
        gain = predicted[:, :, 0] / denominator[:, None]
        state += gain * (series[:, frame] - state[:, 0])[:, None]
        correction = eye.astype(np.complex128).copy()
        correction[:, :, 0] -= gain
        covariance = (correction @ predicted @ correction.conj().transpose(0, 2, 1)
                      + observation_power[:, None, None] * gain[:, :, None] * gain[:, None, :].conj())
        covariance = .5 * (covariance + covariance.conj().transpose(0, 2, 1))
    limit = 3 * np.abs(series).max(axis=1)
    predictions = []
    for _ in range(5):
        state = (transition @ state[..., None])[..., 0]
        state[:, 0] *= np.minimum(1., limit / np.maximum(np.abs(state[:, 0]), 1e-12))
        predictions.append(state[:, 0].copy())
    return _output(np.stack(predictions, axis=1), count)
