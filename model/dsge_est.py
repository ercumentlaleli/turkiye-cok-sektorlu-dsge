"""
Genel Bayesçi DSGE tahmin motoru (v2 modeli için)
 - Gözlenen tanımları esnek, eksik veri (NaN) destekli Kalman filtresi
 - Önseller: beta / gamma / invg (Dynare tip-1) / normal / trunc-normal
 - Sabit parametreler + tahmin edilen alt küme (dönemsel yeniden tahmin için)
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import sys, json, time
import numpy as np, pandas as pd
import scipy.linalg as la
from scipy import stats, special, optimize
sys.path.insert(0, str(MODEL_DIR))
import tr_dsge3 as M   # v3: yatırım + finansal hızlandırıcı

VARS, SHOCKS, IX, build = M.VARS, M.SHOCKS, M.IX, M.build
SX = {s: k for k, s in enumerate(SHOCKS)}
N = len(VARS)

# ------------------------------------------------------------------ veri
OBS = [  # (ad, tür, model değişkeni)
    ("dy", "diff", "y"), ("dc", "diff", "c"), ("pi", "level", "pi"), ("i", "level", "i"),
    ("dq", "diff", "q"), ("doil", "diff", "oil"), ("ystar", "level", "ystar"),
    ("dinv", "diff", "inv"), ("spread", "spread", None),   # spread = spr + zeta*prem (firma kredi faizi - politika faizi)
    ("dcred", "credit", None),   # finans dışı özel sektör kredisi/GSYH büyümesi = Δ(b - y), b = girişimci borcu (BIS)
    ("exp4", "expect", None)]    # 12 ay sonrası TÜFE beklentisi (PKA, 2013 sonrası) = E_t Σ_{j=1..4} π_{t+j}
OBSN = [o[0] for o in OBS]
ME_SHARE = {"dy": 0.10, "dc": 0.10, "ystar": 0.10, "dinv": 0.10, "spread": 0.10,
            "dcred": 0.20,   # döviz cinsi kredilerin kur kaynaklı yeniden değerlemesi modelde yok -> daha büyük ölçüm hatası
            "exp4": 0.10}   # sabit ölçüm hatası (veri varyansı payı)

def load_obs(start, end):
    d = pd.read_csv(f"{MODEL_DIR}/data/evds_raw.csv", index_col=0)
    f = pd.read_csv(f"{MODEL_DIR}/data/foreign.csv", index_col=0)
    d = d.join(f, how="left").join(pd.read_csv(f"{MODEL_DIR}/data/financial.csv", index_col=0), how="left")
    rate = d["aofm"].fillna(d["polrate_bis"])
    obs = pd.DataFrame({
        "dy": 100 * np.log(d["gdp"]).diff(), "dc": 100 * np.log(d["cons"]).diff(),
        "pi": 100 * np.log(d["cpi_sa"]).diff(), "i": rate / 4,
        "dq": -100 * np.log(d["reer"]).diff(), "doil": 100 * np.log(d["brent"]).diff(),
        "ystar": d["cli_eu4"] - 100, "dinv": 100 * np.log(d["inv"]).diff(),
        "spread": (d["loan_rate"] - rate) / 4, "dcred": 100 * np.log(d["credit_gdp"]).diff(),
        "exp4": 100 * np.log(1 + d["exp12_cpi"] / 100)}, index=d.index).loc[start:end]
    means = obs.mean()
    return obs - means, means

# ------------------------------------------------------------------ önseller
def logprior_one(x, typ, m, s):
    if typ == "beta":
        if not 0 < x < 1: return -np.inf
        a = m * (m * (1 - m) / s**2 - 1); b = a * (1 - m) / m
        return stats.beta.logpdf(x, a, b)
    if typ == "gamma":
        if x <= 0: return -np.inf
        return stats.gamma.logpdf(x, (m / s) ** 2, scale=s**2 / m)
    if typ == "normal":
        return stats.norm.logpdf(x, m, s)
    if typ == "invg":
        if x <= 0: return -np.inf
        nu = s
        s0 = m / (np.sqrt(nu / 2) * np.exp(special.gammaln((nu - 1) / 2) - special.gammaln(nu / 2)))
        return (np.log(2) - special.gammaln(nu / 2) + (nu / 2) * np.log(nu * s0**2 / 2)
                - (nu + 1) * np.log(x) - nu * s0**2 / (2 * x**2))
    raise ValueError(typ)

def default_priors():
    return [
        ("sigma", 1.8, "gamma", 2.0, 0.5), ("h", 0.53, "beta", 0.7, 0.1), ("eta", 0.5, "gamma", 0.8, 0.2),
        ("thH", 0.63, "beta", 0.6, 0.1), ("thF", 0.6, "beta", 0.5, 0.1), ("dH", 0.3, "beta", 0.5, 0.15),
        ("dF", 0.5, "beta", 0.5, 0.15), ("rho_w", 0.75, "beta", 0.7, 0.1), ("zeta", 0.8, "gamma", 1.0, 0.5),
        ("om_e", 0.2, "beta", 0.2, 0.05), ("kap_oil", 0.05, "gamma", 0.05, 0.025),
        ("rho_i", 0.85, "beta", 0.75, 0.1), ("phi_pi", 1.85, "gamma", 1.5, 0.25),
        ("phi_dy", 0.25, "gamma", 0.25, 0.1), ("phi_e", 0.11, "gamma", 0.1, 0.05),
        ("rho_a", 0.7, "beta", 0.7, 0.1), ("rho_d", 0.6, "beta", 0.7, 0.1),
        ("rho_prem", 0.7, "beta", 0.7, 0.1), ("rho_oil", 0.3, "beta", 0.5, 0.2),
        ("rho_ystar", 0.9, "beta", 0.8, 0.1), ("rho_rstar", 0.8, "beta", 0.7, 0.1),
        ("rho_m", 0.28, "beta", 0.5, 0.2), ("rho_cp", 0.3, "beta", 0.5, 0.2),
        ("sd_a", 3.0, "invg", 0.5, 4.0), ("sd_d", 0.9, "invg", 0.5, 4.0), ("sd_prem", 0.5, "invg", 0.5, 4.0),
        ("sd_oil", 15.0, "invg", 5.0, 4.0), ("sd_ystar", 0.4, "invg", 0.5, 4.0),
        ("sd_rstar", 0.9, "invg", 0.5, 4.0), ("sd_m", 0.4, "invg", 0.5, 4.0), ("sd_cp", 0.5, "invg", 0.5, 4.0),
        # v3: yatırım ve finansal hızlandırıcı
        ("Sk", 4.0, "gamma", 4.0, 1.5), ("nu", 0.05, "gamma", 0.05, 0.02), ("rho_mu", 0.7, "beta", 0.7, 0.1),
        ("sd_mu", 0.5, "invg", 0.5, 4.0), ("sd_nw", 0.5, "invg", 0.5, 4.0),
    ]

# ------------------------------------------------------------------ çözüm
class Solver:
    def __init__(self): self.P = None
    def solve(self, p, tol=1e-10, maxit=3000):
        A0, A1, A2, B = build(p)
        P = self.P if self.P is not None else np.zeros((N, N))
        for it in range(maxit):
            try: Pn = -np.linalg.solve(A0 @ P + A1, A2)
            except np.linalg.LinAlgError: return None
            if not np.all(np.isfinite(Pn)): return None
            if np.max(np.abs(Pn - P)) < tol: P = Pn; break
            P = Pn
        else: return None
        if np.max(np.abs(np.linalg.eigvals(P))) >= 1 - 1e-8: return None
        I, Z = np.eye(N), np.zeros((N, N))
        lam = la.eig(np.block([[Z, I], [-A2, -A1]]), np.block([[I, Z], [Z, A0]]), right=False)
        if int(np.sum(np.abs(lam[np.isfinite(lam)]) < 1 - 1e-9)) != N: return None
        Q = -np.linalg.solve(A0 @ P + A1, B)
        self.P = P
        return P, Q

# ------------------------------------------------------------------ tahminci
class Estimator:
    def __init__(self, start, end, priors=None, fixed=None, base=None):
        self.Ydf, self.means = load_obs(start, end)
        self.Y = self.Ydf[OBSN].values
        self.priors = priors or default_priors()
        self.names = [p[0] for p in self.priors]
        self.base = dict(base or M.BASE); self.base.update(fixed or {})
        self.solver = Solver()
        v = np.nanvar(self.Y, axis=0)
        self.H = np.diag([ME_SHARE.get(o, 0) * v[k] for k, o in enumerate(OBSN)]) + 1e-8 * np.eye(len(OBSN))

    def params(self, theta):
        p = dict(self.base); p.update(dict(zip(self.names, theta))); return p

    def logprior(self, theta):
        lp = 0.0
        for x, (_, _, t, m, s) in zip(theta, self.priors):
            lp += logprior_one(x, t, m, s)
            if not np.isfinite(lp): return -np.inf
        return lp

    def state_space(self, theta):
        p = self.params(theta); sol = self.solver.solve(p)
        if sol is None: return None
        P, Q = sol
        sds = np.array([p["sd_" + s[2:]] for s in SHOCKS])
        T = np.zeros((2 * N, 2 * N)); T[:N, :N] = P; T[N:, :N] = np.eye(N)
        R = np.zeros((2 * N, len(SHOCKS))); R[:N] = Q * sds
        Z = np.zeros((len(OBS), 2 * N))
        KN = p["KN"]; Pj = np.eye(N); Psum = np.zeros((N, N))
        for _ in range(4): Pj = P @ Pj; Psum += Pj
        for k, (_, kind, v) in enumerate(OBS):
            if kind == "spread":
                Z[k, IX["spr"]] = 1; Z[k, IX["prem"]] = p["zeta"]; continue
            if kind == "credit":   # b = KN/(KN-1)*(qk+k) - 1/(KN-1)*n ;  gözlenen = Δ(b - y)
                for off, sg in ((0, 1), (N, -1)):
                    Z[k, off + IX["qk"]] += sg * KN / (KN - 1); Z[k, off + IX["k"]] += sg * KN / (KN - 1)
                    Z[k, off + IX["n"]] -= sg / (KN - 1); Z[k, off + IX["y"]] -= sg
                continue
            if kind == "expect":   # E_t Σ_{j=1..4} π_{t+j} = (P + P² + P³ + P⁴)[π, :] z_t
                Z[k, :N] = Psum[IX["pi"]]; continue
            Z[k, IX[v]] = 1
            if kind == "diff": Z[k, N + IX[v]] = -1
        return T, R, Z

    def kalman(self, T, R, Z, smooth=False, Y=None):
        Y = self.Y if Y is None else Y
        n = T.shape[0]; RR = R @ R.T
        a = np.zeros(n); Pm = la.solve_discrete_lyapunov(T, RR)
        ll = 0.0; store = []
        for t in range(len(Y)):
            ok = ~np.isnan(Y[t]); Zt = Z[ok]; Ht = self.H[np.ix_(ok, ok)]
            v = Y[t, ok] - Zt @ a
            F = Zt @ Pm @ Zt.T + Ht
            try: cF = la.cho_factor(F)
            except la.LinAlgError: return -np.inf, None
            Finv = la.cho_solve(cF, np.eye(len(v)))
            ll += -0.5 * (len(v) * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(cF[0]))) + v @ Finv @ v)
            K = Pm @ Zt.T @ Finv
            if smooth: store.append((a.copy(), Pm.copy(), v, Finv, Zt))
            a = T @ (a + K @ v); Pm = T @ (Pm - K @ Zt @ Pm) @ T.T + RR
        if not smooth: return ll, None
        r = np.zeros(n); xs = np.zeros((len(Y), n)); eps = np.zeros((len(Y), R.shape[1]))
        for t in range(len(Y) - 1, -1, -1):
            a_t, P_t, v, Finv, Zt = store[t]
            L = T - T @ P_t @ Zt.T @ Finv @ Zt
            r_prev = Zt.T @ Finv @ v + L.T @ r
            xs[t] = a_t + P_t @ r_prev; eps[t] = R.T @ r_prev; r = r_prev
        return ll, (xs, eps)

    def logpost(self, theta):
        lp = self.logprior(theta)
        if not np.isfinite(lp): return -np.inf
        ss = self.state_space(theta)
        if ss is None: return -np.inf
        ll, _ = self.kalman(*ss)
        return ll + lp if np.isfinite(ll) else -np.inf

    def find_mode(self, th0, rounds=3, maxfev=8000, verbose=True):
        f = lambda th: (lambda v: -v if np.isfinite(v) else 1e10)(self.logpost(th))
        best, bv = np.array(th0, float), f(th0); t0 = time.time()
        for r in range(rounds):
            r1 = optimize.minimize(f, best, method="Nelder-Mead",
                                   options=dict(maxiter=maxfev, maxfev=maxfev, xatol=1e-5, fatol=1e-4, adaptive=True))
            r2 = optimize.minimize(f, r1.x, method="Powell", options=dict(maxiter=20000, xtol=1e-5, ftol=1e-7))
            if verbose: print(f"tur {r}: NM {r1.fun:.3f}  Powell {r2.fun:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            if r2.fun < bv - 1e-3: best, bv = r2.x, r2.fun
            else: break
        return best, -bv

    def hessian_cov(self, x, rel=1e-3):
        f = self.logpost; k = len(x); h = np.maximum(np.abs(x) * rel, 1e-4); Hm = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
                Hm[i, j] = Hm[j, i] = -(f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*h[i]*h[j])
        w, V = np.linalg.eigh(Hm); w = np.maximum(w, w.max() * 1e-6)
        return V @ np.diag(1 / w) @ V.T

    def rwmh(self, x0, cov, ndraw, scale, seed, out, log_every=2000):
        rng = np.random.default_rng(seed); Lc = np.linalg.cholesky(cov)
        x = np.array(x0, float); lpx = self.logpost(x)
        draws = np.zeros((ndraw, len(x))); acc = 0; t0 = time.time()
        for it in range(ndraw):
            y = x + scale * Lc @ rng.standard_normal(len(x)); lpy = self.logpost(y)
            if np.log(rng.uniform()) < lpy - lpx: x, lpx = y, lpy; acc += 1
            draws[it] = x
            if (it + 1) % log_every == 0:
                print(f"{out} {it+1} acc={acc/(it+1):.3f} lp={lpx:.1f} t={time.time()-t0:.0f}s", flush=True)
                np.save(out, draws[:it + 1])
        np.save(out, draws)
        return draws

def rhat(chs):
    n = min(len(c) for c in chs); chs = [c[:n] for c in chs]
    m = np.array([c.mean(0) for c in chs]); v = np.array([c.var(0, ddof=1) for c in chs])
    W = v.mean(0); B = n * m.var(0, ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)
