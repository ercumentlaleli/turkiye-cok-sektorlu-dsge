"""
Rejim değişkenli (Markov-switching) DSGE: 2006–2026 birlikte tahmin.

Rejimler iki bağımsız Markov zincirinden oluşur:
  A: para politikası rejimi    (1 = aktif, 2 = pasif)        -> phi_pi, phi_dy, rho_i, phi_e, sd_m
  B: fiyatlama ortamı           (1 = normal, 2 = yüksek enfl.) -> thH, dH, sd_cp ve nominal gözlenenlerin ortalaması
Birleşik durum s = (a, b), S = 4; geçiş matrisi Pi = kron(PA, PB).

Çözüm: Farmer, Waggoner ve Zha (2011) minimum durum değişkenli çözümü; rejime bağlı
  A0(s)·Σ_j Pi[s,j]·P_j·P_s + A1(s)·P_s + A2(s) = 0  ->  P_s  (fonksiyonel iterasyon)
  Q_s = -(A0(s)·Σ_j Pi[s,j]·P_j + A1(s))^{-1} B(s)
Ajanlar rejim değişimini beklentilerine yansıttığından pasif rejimde phi_pi < 1 olsa da çözüm tekil olabilir.
Olabilirlik: Kim (1994) filtresi (rejim olasılıkları Hamilton filtresiyle; durumlar her adımda birleştirilir).
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent
import sys, json, time
import numpy as np, pandas as pd, scipy.linalg as la
sys.path.insert(0, str(MODEL_DIR))
import dsge_est as DE
from dsge_est import logprior_one

M = DE.M; VARS, SHOCKS, IX, N = DE.VARS, DE.SHOCKS, DE.IX, DE.N
SX = {s: k for k, s in enumerate(SHOCKS)}
REG_A = ["phi_pi", "phi_dy", "rho_i", "phi_e", "sd_m"]
REG_B = ["thH", "dH", "sd_cp"]
NOMINAL = ["pi", "i", "exp4", "spread"]      # rejim B'ye göre ortalaması değişen gözlenenler
STATES = [(a, b) for a in (0, 1) for b in (0, 1)]
S = len(STATES)

def solve_ms(plist, Pi, tol=1e-9, maxit=4000, P0=None):
    """plist: S rejimin parametre sözlükleri; Pi: SxS geçiş matrisi. Döner: [(P_s, Q_s)] veya None."""
    mats = [M.build(p) for p in plist]
    P = [np.zeros((N, N)) if P0 is None else P0[s].copy() for s in range(S)]
    for it in range(maxit):
        Pn = []
        for s in range(S):
            A0, A1, A2, _ = mats[s]
            EP = sum(Pi[s, j] * P[j] for j in range(S))
            try: Pn.append(-np.linalg.solve(A0 @ EP + A1, A2))
            except np.linalg.LinAlgError: return None
        diff = max(np.max(np.abs(Pn[s] - P[s])) for s in range(S)); P = Pn
        if not np.isfinite(diff) or diff > 1e6: return None
        if diff < tol: break
    else:
        return None
    out = []
    for s in range(S):
        A0, A1, A2, B = mats[s]
        EP = sum(Pi[s, j] * P[j] for j in range(S))
        out.append((P[s], -np.linalg.solve(A0 @ EP + A1, B)))
    if max(np.max(np.abs(np.linalg.eigvals(o[0]))) for o in out) > 1.5: return None   # patlayan çözüm
    return out

def kim_filter(Ts, Rs, Zs, cs, H, Pi, Y, p0=None):
    """Ts, Rs, Zs, cs: rejime göre durum geçiş/şok/gözlem matrisleri ve gözlem sabitleri. Döner: log-olabilirlik, düzleştirilmemiş rejim olasılıkları."""
    n = Ts[0].shape[0]; T_ = len(Y)
    # başlangıç: rejim 1'in koşulsuz dağılımı
    pr = np.full(S, 1.0 / S) if p0 is None else p0.copy()
    a = [np.zeros(n) for _ in range(S)]
    Pm = [la.solve_discrete_lyapunov(Ts[0], Rs[0] @ Rs[0].T) for _ in range(S)]
    ll = 0.0; filt = np.zeros((T_, S))
    for t in range(T_):
        ok = ~np.isnan(Y[t])
        a_new = np.zeros((S, S, n)); P_new = np.zeros((S, S, n, n)); lik = np.zeros((S, S))
        for i in range(S):            # önceki rejim
            for j in range(S):        # şimdiki rejim
                ap = Ts[j] @ a[i]; Pp = Ts[j] @ Pm[i] @ Ts[j].T + Rs[j] @ Rs[j].T
                if ok.any():
                    Zt = Zs[j][ok]; v = Y[t, ok] - cs[j][ok] - Zt @ ap
                    F = Zt @ Pp @ Zt.T + H[np.ix_(ok, ok)]
                    try: cF = la.cho_factor(F)
                    except la.LinAlgError: return -np.inf, None
                    Fi_v = la.cho_solve(cF, v)
                    K = Pp @ Zt.T @ la.cho_solve(cF, np.eye(len(v)))
                    a_new[i, j] = ap + K @ v; P_new[i, j] = Pp - K @ Zt @ Pp
                    lik[i, j] = np.exp(-0.5 * (len(v) * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(cF[0]))) + v @ Fi_v))
                else:
                    a_new[i, j], P_new[i, j], lik[i, j] = ap, Pp, 1.0
        joint = (pr[:, None] * Pi) * lik
        tot = joint.sum()
        if not np.isfinite(tot) or tot <= 0: return -np.inf, None
        ll += np.log(tot); joint /= tot
        pr = joint.sum(0); filt[t] = pr
        for j in range(S):            # Kim birleştirmesi
            w = joint[:, j] / max(pr[j], 1e-300)
            a[j] = (w[:, None] * a_new[:, j]).sum(0)
            Pm[j] = sum(w[i] * (P_new[i, j] + np.outer(a_new[i, j] - a[j], a_new[i, j] - a[j])) for i in range(S))
    return ll, filt


# ------------------------------------------------------------------ tahmin
BASE_FIX = None
def load_data(start="2006Q1", end="2026Q2"):
    obs, _ = DE.load_obs(start, end)          # tüm örneklem ortalamasından arındırılmış
    raw = obs + _                              # ham seriler
    base = raw.loc["2006Q1":"2019Q4"].mean()   # normal rejim ortalamaları (2006–19)
    Y = raw - base                             # nominal gözlenenlerin rejim B=2 kayması ayrıca tahmin edilir
    Y.loc["2020Q2":"2020Q4"] = np.nan          # salgın çeyrekleri
    return Y, base

def default_ms_priors():
    return [  # (isim, başlangıç, tip, ort, sd)
        ("A2_phi_pi", 0.8, "gamma", 1.0, 0.4), ("A2_phi_dy", 0.2, "gamma", 0.25, 0.1), ("A2_rho_i", 0.7, "beta", 0.75, 0.1),
        ("A2_phi_e", 0.08, "gamma", 0.1, 0.05), ("A2_sd_m", 1.0, "invg", 0.5, 4.0),
        ("B2_thH", 0.5, "beta", 0.6, 0.15), ("B2_dH", 0.4, "beta", 0.5, 0.15), ("B2_sd_cp", 2.0, "invg", 1.0, 4.0),
        ("off_pi", 6.0, "normal", 5.0, 5.0), ("off_i", 5.0, "normal", 5.0, 5.0), ("off_exp4", 20.0, "normal", 20.0, 15.0),
        ("off_spread", 2.0, "normal", 2.0, 3.0),
        ("pA11", 0.95, "beta", 0.95, 0.03), ("pA22", 0.85, "beta", 0.85, 0.1),
        ("pB11", 0.95, "beta", 0.95, 0.03), ("pB22", 0.85, "beta", 0.85, 0.1),
        ("sd_a", 1.7, "invg", 0.5, 4.0), ("sd_d", 0.45, "invg", 0.5, 4.0), ("sd_prem", 0.35, "invg", 0.5, 4.0),
        ("sd_oil", 15.0, "invg", 5.0, 4.0), ("sd_ystar", 0.6, "invg", 0.5, 4.0), ("sd_rstar", 0.5, "invg", 0.5, 4.0),
        ("sd_mu", 1.4, "invg", 0.5, 4.0), ("sd_nw", 3.5, "invg", 0.5, 4.0)]

class MSEstimator:
    def __init__(self, start="2006Q1", end="2026Q2", priors=None):
        self.Ydf, self.base = load_data(start, end); self.Y = self.Ydf[DE.OBSN].values
        self.priors = priors or default_ms_priors(); self.names = [q[0] for q in self.priors]
        pm = pd.read_csv(MODEL_DIR / "s1_posterior_table.csv", index_col=0).post_mean.to_dict()
        self.p1 = dict(M.BASE); self.p1.update(pm)          # rejim (1,1): 2006–19 posterior ortalaması
        v = np.nanvar(self.Y, axis=0)
        self.H = np.diag([DE.ME_SHARE.get(o, 0) * v[k] for k, o in enumerate(DE.OBSN)]) + 1e-8 * np.eye(len(DE.OBSN))
        self.P0 = None

    def regime_params(self, th):
        d = dict(zip(self.names, th)); plist = []
        for (a, b) in STATES:
            p = dict(self.p1)
            for k in ("sd_a", "sd_d", "sd_prem", "sd_oil", "sd_ystar", "sd_rstar", "sd_mu", "sd_nw"): p[k] = d[k]
            if a == 1:
                for k in REG_A: p[k] = d["A2_" + k]
            if b == 1:
                for k in REG_B: p[k] = d["B2_" + k]
            plist.append(p)
        PA = np.array([[d["pA11"], 1 - d["pA11"]], [1 - d["pA22"], d["pA22"]]])
        PB = np.array([[d["pB11"], 1 - d["pB11"]], [1 - d["pB22"], d["pB22"]]])
        return plist, np.kron(PA, PB), d

    def logprior(self, th):
        lp = 0.0
        for x, (_, _, t, m, s) in zip(th, self.priors):
            lp += logprior_one(x, t, m, s)
            if not np.isfinite(lp): return -np.inf
        return lp

    def state_space(self, th):
        plist, Pi, d = self.regime_params(th)
        sol = solve_ms(plist, Pi, P0=self.P0)
        if sol is None: return None
        self.P0 = [s_[0] for s_ in sol]
        Ts, Rs, Zs, cs = [], [], [], []
        E = DE.Estimator.__new__(DE.Estimator)          # gözlem matrisini dsge_est ile aynı kurala göre kur
        for s, (a, b) in enumerate(STATES):
            P, Q = sol[s]; p = plist[s]
            sds = np.array([p["sd_" + x[2:]] for x in SHOCKS])
            T = np.zeros((2 * N, 2 * N)); T[:N, :N] = P; T[N:, :N] = np.eye(N)
            R = np.zeros((2 * N, len(SHOCKS))); R[:N] = Q * sds
            Z = _obs_matrix(P, p)
            c = np.zeros(len(DE.OBSN))
            if b == 1:
                for k_, o in enumerate(DE.OBSN):
                    if o in NOMINAL: c[k_] = d["off_" + o]
            Ts.append(T); Rs.append(R); Zs.append(Z); cs.append(c)
        return Ts, Rs, Zs, cs, Pi

    def logpost(self, th):
        lp = self.logprior(th)
        if not np.isfinite(lp): return -np.inf
        ss = self.state_space(th)
        if ss is None: return -np.inf
        Ts, Rs, Zs, cs, Pi = ss
        ll, _ = kim_filter(Ts, Rs, Zs, cs, self.H, Pi, self.Y)
        return ll + lp if np.isfinite(ll) else -np.inf

def _obs_matrix(P, p):
    """dsge_est.Estimator.state_space ile aynı gözlem kuralları."""
    Z = np.zeros((len(DE.OBS), 2 * N)); KN = p["KN"]; Pj = np.eye(N); Psum = np.zeros((N, N))
    for _ in range(4): Pj = P @ Pj; Psum += Pj
    for k, (_, kind, v) in enumerate(DE.OBS):
        if kind == "spread": Z[k, IX["spr"]] = 1; Z[k, IX["prem"]] = p["zeta"]; continue
        if kind == "credit":
            for off, sg in ((0, 1), (N, -1)):
                Z[k, off + IX["qk"]] += sg * KN / (KN - 1); Z[k, off + IX["k"]] += sg * KN / (KN - 1)
                Z[k, off + IX["n"]] -= sg / (KN - 1); Z[k, off + IX["y"]] -= sg
            continue
        if kind == "prod":
            Z[k, IX["y"]] += 1; Z[k, IX["l"]] -= 1; Z[k, N + IX["y"]] -= 1; Z[k, N + IX["l"]] += 1; continue
        if kind == "expect": Z[k, :N] = Psum[IX["pi"]]; continue
        Z[k, IX[v]] = 1
        if kind == "diff": Z[k, N + IX[v]] = -1
    return Z
