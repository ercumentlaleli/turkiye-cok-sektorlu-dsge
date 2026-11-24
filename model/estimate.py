"""
Türkiye küçük açık ekonomi DSGE — Bayesçi tahmin
Gözlenenler (çeyreklik, örneklem ortalamasından arındırılmış):
  dy  : 100*Δlog reel GSYH (MTA, zincirlenmiş)            -> y_t - y_{t-1}
  dc  : 100*Δlog hanehalkı tüketimi (MTA)                 -> c_t - c_{t-1}
  pi  : 100*Δlog TÜFE (MA)                                 -> pi_t
  i   : politika faizi/4 (2006-10 politika faizi, 2011+ AOFM) -> i_t
  dq  : -100*Δlog TÜFE bazlı REK (+ = TL reel değer kaybı)  -> q_t - q_{t-1}
dy ve dc için ölçüm hatası (veri varyansının %10'u, sabit).
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import sys, time, json
import numpy as np, pandas as pd
from scipy import optimize, stats, special
sys.path.insert(0, str(MODEL_DIR))
from tr_dsge import VARS, SHOCKS, IX, BASE, build
import scipy.linalg as la

# ------------------------------------------------------------------ veri
def load_data(start="2006Q1", end="2019Q4"):
    d = pd.read_csv(f"{MODEL_DIR}/data/evds_raw.csv", index_col=0)
    rate = d["aofm"].fillna(d["polrate_bis"])
    obs = pd.DataFrame({
        "dy": 100 * np.log(d["gdp"]).diff(),
        "dc": 100 * np.log(d["cons"]).diff(),
        "pi": 100 * np.log(d["cpi_sa"]).diff(),
        "i": rate / 4,
        "dq": -100 * np.log(d["reer"]).diff(),
    }, index=d.index)
    obs = obs.loc[start:end]
    means = obs.mean()
    return obs - means, means

OBS = ["dy", "dc", "pi", "i", "dq"]

# ------------------------------------------------------------------ parametreler ve önseller
# (isim, başlangıç, önsel tipi, önsel ort., önsel std/df)
PRIORS = [
    ("sigma", 2.0, "gamma", 2.0, 0.5),
    ("h", 0.7, "beta", 0.7, 0.1),
    ("eta", 0.8, "gamma", 0.8, 0.2),
    ("thH", 0.6, "beta", 0.6, 0.1),
    ("thF", 0.5, "beta", 0.5, 0.1),
    ("dH", 0.5, "beta", 0.5, 0.15),
    ("dF", 0.5, "beta", 0.5, 0.15),
    ("rho_w", 0.7, "beta", 0.7, 0.1),
    ("zeta", 1.0, "gamma", 1.0, 0.5),
    ("rho_i", 0.75, "beta", 0.75, 0.1),
    ("phi_pi", 1.5, "gamma", 1.5, 0.25),
    ("phi_y", 0.25, "gamma", 0.25, 0.1),
    ("phi_e", 0.1, "gamma", 0.1, 0.05),
    ("rho_a", 0.8, "beta", 0.7, 0.1),
    ("rho_d", 0.7, "beta", 0.7, 0.1),
    ("rho_prem", 0.8, "beta", 0.7, 0.1),
    ("rho_oil", 0.7, "beta", 0.7, 0.1),
    ("rho_ystar", 0.8, "beta", 0.7, 0.1),
    ("rho_rstar", 0.8, "beta", 0.7, 0.1),
    ("rho_m", 0.5, "beta", 0.5, 0.2),
    ("sd_a", 1.0, "invg", 0.5, 4.0),
    ("sd_d", 1.0, "invg", 0.5, 4.0),
    ("sd_prem", 1.0, "invg", 0.5, 4.0),
    ("sd_oil", 3.0, "invg", 2.0, 4.0),
    ("sd_ystar", 1.0, "invg", 0.5, 4.0),
    ("sd_rstar", 0.3, "invg", 0.5, 4.0),
    ("sd_m", 0.3, "invg", 0.5, 4.0),
]
NAMES = [p[0] for p in PRIORS]
K = len(NAMES)

def logprior(theta):
    lp = 0.0
    for x, (nm, _, typ, m, s) in zip(theta, PRIORS):
        if typ == "beta":
            if not 0 < x < 1: return -np.inf
            a = m * (m * (1 - m) / s**2 - 1); b = a * (1 - m) / m
            lp += stats.beta.logpdf(x, a, b)
        elif typ == "gamma":
            if x <= 0: return -np.inf
            k = (m / s) ** 2; th = s**2 / m
            lp += stats.gamma.logpdf(x, k, scale=th)
        elif typ == "invg":   # Dynare tip-1 ters gama (std için): ort.=m, df=s
            if x <= 0: return -np.inf
            nu = s
            # E[sigma] = s0*sqrt(nu/2)*Gamma((nu-1)/2)/Gamma(nu/2)
            s0 = m / (np.sqrt(nu / 2) * np.exp(special.gammaln((nu - 1) / 2) - special.gammaln(nu / 2)))
            lp += np.log(2) - special.gammaln(nu / 2) + (nu / 2) * np.log(nu * s0**2 / 2) \
                  - (nu + 1) * np.log(x) - nu * s0**2 / (2 * x**2)
    return lp

# ------------------------------------------------------------------ çözüm (sıcak başlangıçlı)
_P_cache = {"P": None}
N = len(VARS)

def solve_fast(p, tol=1e-10, maxit=3000):
    A0, A1, A2, B = build(p)
    P = _P_cache["P"] if _P_cache["P"] is not None else np.zeros((N, N))
    for it in range(maxit):
        try:
            Pn = -np.linalg.solve(A0 @ P + A1, A2)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(Pn)): return None
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn; break
        P = Pn
    else:
        return None
    if np.max(np.abs(np.linalg.eigvals(P))) >= 1 - 1e-8: return None
    # tekillik kontrolü: kararlı genelleştirilmiş özdeğer sayısı = N
    I, Z = np.eye(N), np.zeros((N, N))
    lam = la.eig(np.block([[Z, I], [-A2, -A1]]), np.block([[I, Z], [Z, A0]]), right=False)
    mod = np.abs(lam[np.isfinite(lam)])
    if int(np.sum(mod < 1 - 1e-9)) != N: return None
    Q = -np.linalg.solve(A0 @ P + A1, B)
    _P_cache["P"] = P
    return P, Q

def theta_to_p(theta):
    p = dict(BASE)
    p.update(dict(zip(NAMES, theta)))
    return p

# ------------------------------------------------------------------ durum-uzay ve Kalman
def state_space(theta):
    p = theta_to_p(theta)
    sol = solve_fast(p)
    if sol is None: return None
    P, Q = sol
    sds = np.array([p["sd_" + s[2:]] for s in SHOCKS])
    # genişletilmiş durum: [z_t ; z_{t-1}]
    T = np.zeros((2 * N, 2 * N)); T[:N, :N] = P; T[N:, :N] = np.eye(N)
    R = np.zeros((2 * N, len(SHOCKS))); R[:N] = Q * sds
    Z = np.zeros((5, 2 * N))
    Z[0, IX["y"]] = 1; Z[0, N + IX["y"]] = -1
    Z[1, IX["c"]] = 1; Z[1, N + IX["c"]] = -1
    Z[2, IX["pi"]] = 1
    Z[3, IX["i"]] = 1
    Z[4, IX["q"]] = 1; Z[4, N + IX["q"]] = -1
    return T, R, Z

def kalman(Y, T, R, Z, H, smooth=False):
    n = T.shape[0]; RR = R @ R.T
    a = np.zeros(n)
    Pm = la.solve_discrete_lyapunov(T, RR)
    ll = 0.0; Tn = Y.shape[0]
    store = []
    for t in range(Tn):
        v = Y[t] - Z @ a
        F = Z @ Pm @ Z.T + H
        try:
            cF = la.cho_factor(F)
        except la.LinAlgError:
            return -np.inf, None
        Finv_v = la.cho_solve(cF, v)
        ll += -0.5 * (len(v) * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(cF[0]))) + v @ Finv_v)
        K_ = Pm @ Z.T @ la.cho_solve(cF, np.eye(len(v)))
        if smooth: store.append((a.copy(), Pm.copy(), v, la.cho_solve(cF, np.eye(len(v)))))
        a_upd = a + K_ @ v
        P_upd = Pm - K_ @ Z @ Pm
        a = T @ a_upd
        Pm = T @ P_upd @ T.T + RR
    if not smooth: return ll, None
    # Durbin-Koopman düzleştirici (durum ve şok)
    r = np.zeros(n); xs = np.zeros((Tn, n)); eps = np.zeros((Tn, R.shape[1]))
    for t in range(Tn - 1, -1, -1):
        a_t, P_t, v, Finv = store[t]
        L = T - T @ P_t @ Z.T @ Finv @ Z
        r_prev = Z.T @ Finv @ v + L.T @ r
        xs[t] = a_t + P_t @ r_prev
        eps[t] = R.T @ r_prev      # standartlaştırılmış düzleştirilmiş şok (t dönemi)
        r = r_prev
    return ll, (xs, eps)

# ------------------------------------------------------------------ posterior
Ydf, MEANS = load_data()
Y = Ydf[OBS].values
H = np.diag([0.10 * np.var(Y[:, 0]), 0.10 * np.var(Y[:, 1]), 0, 0, 0]) + 1e-8 * np.eye(5)

def logpost(theta):
    lp = logprior(theta)
    if not np.isfinite(lp): return -np.inf
    ss = state_space(theta)
    if ss is None: return -np.inf
    ll, _ = kalman(Y, *ss, H)
    if not np.isfinite(ll): return -np.inf
    return ll + lp

if __name__ == "__main__":
    th0 = np.array([p[1] for p in PRIORS])
    t0 = time.time(); v0 = logpost(th0); print("logpost(th0)", v0, "süre", time.time() - t0)
    t0 = time.time()
    for _ in range(20): logpost(th0 * (1 + 0.01 * np.random.randn(K)))
    print("ortalama değerlendirme süresi", (time.time() - t0) / 20)
