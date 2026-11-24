"""
Schorfheide-Sill-Kryshko (2010) uydu denklemleri — NACE-2 sektörleri
  z_{s,t} = λ0 + Λ_s' x_t + u_t,   u_t = ρ_s u_{t-1} + η_t
x_t: DSGE düzleştirilmiş durumlarından türetilen çekirdek değişkenler (her posterior çekilişi için)
  Fiyat (Yİ-ÜFE büyümesi):  pi, de (nominal kur), doil, mc, y   (+ mevsim kuklaları)
  Üretim (SÜE büyümesi):    dy, dq, dystar, rr, doil
Sektörel IRF = Λ_s' × IRF(x); seviye tepkisi = kümülatif toplam.
Belirsizlik: DSGE posterior çekilişleri × Λ'nin GLS örnekleme dağılımından çekiliş.
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import sys, json, numpy as np, pandas as pd
from dsge_est import *

SECT = ["B"] + [f"C{k}" for k in range(10, 33)] + ["D35"]
PX = ["pi", "de", "doil", "y"]
QX = ["dy", "dq", "dystar", "doil"]
SHOCK_SPEC = {"e_m": "Para politikası (+1 puan faiz)", "e_oil": "Brent petrol (+%10)",
              "e_rstar": "Kur primi (TL reel %1 değer kaybı)", "e_ystar": "Dış talep (AB-4 öncü gös.) +%1",
              "e_cp": "Yerli maliyet (yerli fiyatlar) +%1", "e_d": "Yurt içi talep (tüketim) +%1"}
# Etki anı normalizasyonu: şok -> (değişken, hedef). Hepsi yüzde/puan, çeyreklik birimlerde.
NORM = {"e_m": ("i", 0.25), "e_oil": ("doil", 10.0), "e_rstar": ("q", 1.0), "e_ystar": ("ystar", 1.0),
        "e_d": ("c", 1.0), "e_cp": ("piH", 1.0)}
HZ = 12

def sector_growth(start="2006Q1", end="2019Q4"):
    d = pd.read_csv(f"{MODEL_DIR}/data/sectors.csv", index_col=0)
    g = 100 * np.log(d).diff().loc[start:end]
    return g

def core_vars(xs):
    """xs: (T, 2N) düzleştirilmiş durumlar -> dict of core series"""
    cur, lag = xs[:, :N], xs[:, N:]
    v = lambda k: cur[:, IX[k]]; l = lambda k: lag[:, IX[k]]
    return {"pi": v("pi"), "de": v("de"), "doil": v("oil") - l("oil"), "mc": v("mc"), "y": v("y"),
            "dy": v("y") - l("y"), "dq": v("q") - l("q"), "dystar": v("ystar") - l("ystar"), "rr": v("rr")}

def core_irf(P, Q, shock_vec, H=HZ):
    z = np.zeros((H + 1, N)); z[1] = Q @ shock_vec          # z[0] = şok öncesi (0)
    for t in range(2, H + 1): z[t] = P @ z[t - 1]
    cur, lag = z[1:], z[:-1]
    v = lambda k: cur[:, IX[k]]; l = lambda k: lag[:, IX[k]]
    return {"pi": v("pi"), "de": v("de"), "doil": v("oil") - l("oil"), "mc": v("mc"), "y": v("y"),
            "dy": v("y") - l("y"), "dq": v("q") - l("q"), "dystar": v("ystar") - l("ystar"), "rr": v("rr"),
            "i": v("i"), "q": v("q"), "ystar": v("ystar"), "c": v("c"), "piH": v("piH")}

def gls_ar1(y, X, iters=20):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; rho = 0.0
    for _ in range(iters):
        u = y - X @ b
        rho = np.clip(np.dot(u[1:], u[:-1]) / np.dot(u[:-1], u[:-1]), -0.95, 0.95)
        ys = y[1:] - rho * y[:-1]; Xs = X[1:] - rho * X[:-1]
        b_new = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-8: b = b_new; break
        b = b_new
    ys = y[1:] - rho * y[:-1]; Xs = X[1:] - rho * X[:-1]
    e = ys - Xs @ b; s2 = e @ e / (len(ys) - Xs.shape[1])
    V = s2 * np.linalg.inv(Xs.T @ Xs)
    r2 = 1 - np.var(y - X @ b) / np.var(y)
    return b, V, rho, r2

def run(draws, E, seed=0):
    rng = np.random.default_rng(seed)
    G = sector_growth(); Tn = len(G)
    q = np.array([int(s[-1]) for s in G.index])
    seas = np.column_stack([(q == k).astype(float) for k in (2, 3, 4)])
    out = {sh: {"PP": [], "IP": []} for sh in SHOCK_SPEC}
    fit = {"PP": [], "IP": []}; LAM = {"PP": [], "IP": []}
    for th in draws:
        ss = E.state_space(th)
        if ss is None: continue
        T_, R_, Z_ = ss
        _, (xs, _) = E.kalman(T_, R_, Z_, smooth=True)
        cv = core_vars(xs)
        Xp = np.column_stack([np.ones(Tn)] + [cv[k] for k in PX] + [seas])
        Xq = np.column_stack([np.ones(Tn)] + [cv[k] for k in QX])
        p = E.params(th); P, Q = E.solver.P, None
        sol = E.solver.solve(p); P, Q = sol
        irfs = {}
        for sh in SHOCK_SPEC:
            e = np.zeros(len(SHOCKS)); e[SX[sh]] = p["sd_" + sh[2:]]
            ci = core_irf(P, Q, e)
            nv, tgt = NORM[sh]   # ör. faiz +1 puan (yıllık) = 0,25 çeyreklik; Brent +%10; diğerleri %1
            scale = tgt / ci[nv][0]; ci = {k: v * scale for k, v in ci.items()}
            irfs[sh] = ci
        lamP, lamQ, r2P, r2Q = [], [], [], []
        for kind, X, names in (("PP", Xp, PX), ("IP", Xq, QX)):
            for s in SECT:
                y = G[f"{kind}_{s}"].values
                b, V, rho, r2 = gls_ar1(y, X)
                bd = rng.multivariate_normal(b, V)          # parametre belirsizliği
                lam = bd[1:1 + len(names)]
                (lamP if kind == "PP" else lamQ).append(lam)
                (r2P if kind == "PP" else r2Q).append(r2)
                for sh in SHOCK_SPEC:
                    resp = sum(lam[j] * irfs[sh][nm] for j, nm in enumerate(names))
                    out[sh][kind].append(np.cumsum(resp))
        fit["PP"].append(r2P); fit["IP"].append(r2Q)
        LAM["PP"].append(lamP); LAM["IP"].append(lamQ)
    nd = len(fit["PP"]); S = len(SECT)
    res = {sh: {k: np.array(v[k]).reshape(nd, S, HZ) for k in ("PP", "IP")} for sh, v in out.items()}
    return res, {k: np.array(v) for k, v in fit.items()}, {k: np.array(v) for k, v in LAM.items()}

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "full":
    E = Estimator("2006Q1", "2019Q4")
    D = np.load("s1_draws.npy"); rng = np.random.default_rng(7)
    sel = D[rng.choice(len(D), 300, replace=False)]
    res, fit, lam = run(sel, E)
    np.savez("ssk_results.npz", **{f"{sh}_{k}": res[sh][k] for sh in res for k in res[sh]},
             fitPP=fit["PP"], fitIP=fit["IP"], lamPP=lam["PP"], lamIP=lam["IP"])
    print("bitti", res["e_m"]["PP"].shape)
elif __name__ == "__main__":
    E = Estimator("2006Q1", "2019Q4")
    m = json.load(open("s1_mode.json"))
    res, fit, lam = run(np.array([m["theta"]]), E)
    print("R2 fiyat:", dict(zip(SECT, fit["PP"][0].round(2))))
    print("R2 üretim:", dict(zip(SECT, fit["IP"][0].round(2))))
    for sh in ["e_m", "e_oil"]:
        print(sh, "PP h=4:", dict(zip(SECT, res[sh]["PP"][0, :, 4].round(2))))
        print(sh, "IP h=4:", dict(zip(SECT, res[sh]["IP"][0, :, 4].round(2))))
