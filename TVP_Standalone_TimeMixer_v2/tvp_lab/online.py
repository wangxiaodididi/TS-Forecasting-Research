"""Lightweight Kalman / recursive ridge on frozen-backbone *component contributions*.

Shapes: baseline and y [N,H,C], components [N,H,C,S] in target units.
Prediction at forecast origin i receives updates ONLY from origins j with j+H <= i.
A single state vector is shared over channels and horizon positions to limit variance.
This is a post-hoc TVP *forecast mapping*, not neural attention and not causal inference.
"""
from __future__ import annotations
import numpy as np


def check_shapes(base, parts, y):
    base, parts, y = map(lambda x: np.asarray(x, dtype=np.float64), (base, parts, y))
    if base.ndim != 3 or y.shape != base.shape or parts.ndim != 4 or parts.shape[:3] != base.shape:
        raise ValueError('Need base/y [N,H,C] and parts [N,H,C,S].')
    if not all(np.isfinite(x).all() for x in (base, parts, y)):
        raise ValueError('Data contains non-finite observations.')
    return base, parts, y


def fit_static(base, parts, y, ridge=5.):
    base, parts, y = check_shapes(base, parts, y)
    x = parts.reshape(-1, parts.shape[-1]); r = (y-base).reshape(-1)
    # Mean squared loss + ridge ||delta||^2. delta=0 means EXACT original backbone.
    gram = (x.T @ x) / len(x)
    rhs = (x.T @ r) / len(x)
    return np.linalg.solve(gram + ridge * np.eye(x.shape[-1]), rhs)


def gate_features(parts):
    """At-origin, label-free uncertainty proxy: disagreement among scale contributions."""
    p = np.asarray(parts)
    return np.log1p(np.std(p, axis=-1).mean(axis=(1,2)))


def fit_gate(base, parts, y, ridge=5.):
    base, parts, y = check_shapes(base, parts, y)
    gate = gate_features(parts)
    mu, sigma = float(gate.mean()), float(gate.std()+1.e-8)
    x = np.concatenate((parts, parts*((gate-mu)/sigma)[:,None,None,None]), axis=-1)
    coef = fit_static(base, x, y, ridge=ridge)
    return coef, mu, sigma


def predict_gate(base, parts, fit):
    coef, mu, sigma = fit
    q = ((gate_features(parts)-mu)/sigma)[:,None,None,None]
    feats = np.concatenate((parts, parts*q), axis=-1)
    return base + np.einsum('nhcs,s->nhc', feats, coef)


class RecursiveTVP:
    """Random walk state delta_i = delta_(i-1) + eta, eta~N(0,qI).

    Each matured *whole forecast origin* gives one batch observation, averaged
    in the information update. Forecasts overlap; covariance is algorithmic
    regularization, NOT a valid inferential parameter covariance estimator.
    """
    def __init__(self, n_sources, initial=None, q=1e-4, measurement=1., p0=.1, clip=.8):
        self.d = np.zeros(n_sources, dtype=float) if initial is None else np.array(initial, dtype=float).copy()
        self.P = np.eye(n_sources) * float(p0)
        self.q = float(q); self.measurement = float(measurement); self.clip = float(clip)
        if self.q < 0 or measurement <= 0 or p0 <= 0: raise ValueError('Invalid TVP hyperparameters')
        self.num_updates = 0
    def forecast(self, b, parts):
        return np.asarray(b) + np.einsum('...s,s->...', parts, self.d)
    def update(self, b, parts, y):
        x = np.asarray(parts, float).reshape(-1, len(self.d))
        residual = (np.asarray(y, float)-np.asarray(b, float)).reshape(-1)
        prior_inv = np.linalg.inv(self.P + self.q * np.eye(len(self.d)))
        # Observation averaging prevents H*C from creating artificial certainty.
        scale = self.measurement * len(residual)
        posterior = np.linalg.inv(prior_inv + x.T@x/scale)
        self.d = posterior @ (prior_inv @ self.d + x.T @ residual / scale)
        self.d = np.clip(self.d, -self.clip, self.clip)
        self.P = (posterior+posterior.T)/2
        self.num_updates += 1


def replay(base, parts, y, initial, horizon, q=1e-4, measurement=1., p0=.1):
    """Returns online forecasts, state paths and update indices for one contiguous split.

    If labels are fully available before next split, use warm_start() on that
    prior split, not replay() on concatenated val+test with arbitrary boundaries.
    """
    base, parts, y = check_shapes(base, parts, y)
    model = RecursiveTVP(parts.shape[-1], initial, q, measurement, p0)
    result=np.empty_like(base); states=np.empty((len(base), parts.shape[-1])); updates=[]
    for i in range(len(base)):
        matured = i-int(horizon)
        if matured >= 0:
            model.update(base[matured], parts[matured], y[matured]); updates.append((i,matured))
        result[i]=model.forecast(base[i],parts[i]); states[i]=model.d
    return result, states, updates


def warm_start(base, parts, y, initial, q=1e-4, measurement=1., p0=.1):
    """All calibration labels are available by the start of the test split."""
    base, parts, y = check_shapes(base, parts, y)
    m=RecursiveTVP(parts.shape[-1], initial, q, measurement, p0)
    for i in range(len(base)): m.update(base[i], parts[i], y[i])
    return m


def test_online(base, parts, y, fitted_state, horizon):
    """The fitted_state object must have been initialized from pre-test data ONLY."""
    base, parts, y=check_shapes(base,parts,y)
    preds=np.empty_like(base); states=np.empty((len(base),parts.shape[-1])); updated=[]
    for i in range(len(base)):
        j=i-int(horizon)
        if j>=0:
            fitted_state.update(base[j],parts[j],y[j]); updated.append(j)
        preds[i]=fitted_state.forecast(base[i],parts[i]); states[i]=fitted_state.d
    return preds, states, updated
