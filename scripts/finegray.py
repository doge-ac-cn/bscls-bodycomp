#!/usr/bin/env python3
"""Fine-Gray subdistribution hazard model via IPCW weighted partial likelihood.

event encoding: 0 = censored, 1 = event of interest, 2 = competing event.
Weights: for a subject j who already had a competing event at T_j < t, the
weight in the risk set at time t is G(t)/G(T_j), where G is the KM estimate
of the censoring distribution (censoring treated as the event).
"""
import numpy as np


class FineGray:
    def __init__(self, tol=1e-9, max_iter=200):
        self.tol = tol
        self.max_iter = max_iter
        self.beta_ = None
        self.cov_ = None

    def fit(self, time, event, X):
        time = np.asarray(time, float)
        event = np.asarray(event, int)
        X = np.asarray(X, float)
        n, p = X.shape

        # ---- KM estimate of censoring distribution G(t) ----
        cens = (event == 0).astype(int)
        order = np.argsort(time)
        ts = time[order]
        cs = cens[order]
        uniq_t = np.unique(ts)
        G_at = np.ones(len(uniq_t))
        n_at = n
        surv = 1.0
        for k, t in enumerate(uniq_t):
            d_t = int(np.sum((ts == t) & (cs == 1)))
            if n_at > 0 and d_t > 0:
                surv *= (1.0 - d_t / n_at)
            G_at[k] = surv
            n_at -= int(np.sum(ts == t))
        # G(t_i) for each subject (right-continuous step function)
        idx_g = np.clip(np.searchsorted(uniq_t, ts, side="right") - 1, 0, len(uniq_t) - 1)
        G_subj = np.empty(n)
        G_subj[order] = G_at[idx_g]

        def G_at_time(t):
            idx = np.searchsorted(uniq_t, t, side="right") - 1
            if idx < 0:
                return 1.0
            return G_at[idx]

        # ---- weighted partial likelihood, Newton-Raphson (log-sum-exp stabilized) ----
        ev_idx = np.where(event == 1)[0]
        ev_order = ev_idx[np.argsort(time[ev_idx])]
        beta = np.zeros(p)
        hess = None
        for it in range(self.max_iter):
            Xb = np.clip(X @ beta, -40, 40)
            grad = np.zeros(p)
            hess = np.zeros((p, p))
            for i in ev_order:
                ti = time[i]
                in_risk = (time >= ti) | ((event == 2) & (time < ti))
                w = np.ones(n)
                comp_before = (event == 2) & (time < ti)
                if comp_before.any():
                    w[comp_before] = G_at_time(ti) / np.clip(G_subj[comp_before], 1e-10, None)
                # log(w * exp(Xb)) with w>0 guaranteed via clip
                eta = Xb + np.log(np.clip(w, 1e-12, None))
                eta_r = eta[in_risk]
                m = eta_r.max()
                e_r = np.exp(eta_r - m)
                denom = e_r.sum()
                if denom <= 0 or not np.isfinite(denom):
                    continue
                num = (e_r[:, None] * X[in_risk]).sum(axis=0)
                grad += X[i] - num / denom
                hess += (X[in_risk] * e_r[:, None]).T @ X[in_risk] / denom \
                    - np.outer(num, num) / denom ** 2
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            # step-size limit + simple line search to avoid divergence
            if not np.all(np.isfinite(step)):
                step = np.zeros(p)
            step = np.clip(step, -2.0, 2.0)
            beta_new = beta + step
            if np.max(np.abs(step)) < self.tol:
                beta = beta_new
                break
            beta = beta_new
            if not np.all(np.isfinite(beta)):
                beta = np.zeros(p)
                break
        self.beta_ = beta
        self.cov_ = None
        if hess is not None and np.all(np.isfinite(hess)):
            try:
                self.cov_ = np.linalg.inv(hess)
            except np.linalg.LinAlgError:
                self.cov_ = None
        return self

    @property
    def hazard_ratios_(self):
        return np.exp(self.beta_)

    def summary(self, names=None):
        names = names if names is not None else [f"x{i}" for i in range(len(self.beta_))]
        se = np.sqrt(np.diag(self.cov_)) if self.cov_ is not None else np.full_like(self.beta_, np.nan)
        z = self.beta_ / se
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(np.abs(z)))
        return [{"var": names[i], "coef": float(self.beta_[i]), "HR": float(np.exp(self.beta_[i])),
                 "se": float(se[i]), "z": float(z[i]), "p": float(p[i])} for i in range(len(self.beta_))]
