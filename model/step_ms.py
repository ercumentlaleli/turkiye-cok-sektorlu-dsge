"""Rejim değişkenli model (msdsge.py) tahmini: python step_ms.py mode | chain <tohum> <çekiliş>"""
import sys, json, time
import numpy as np
from scipy import optimize
import msdsge as MS
E = MS.MSEstimator()
stage = sys.argv[1]

def find_mode(th0, rounds=3, maxfev=6000):
    f = lambda th: (lambda v: -v if np.isfinite(v) else 1e10)(E.logpost(th))
    best, bv = np.array(th0, float), f(th0); t0 = time.time()
    for r in range(rounds):
        r1 = optimize.minimize(f, best, method="Nelder-Mead", options=dict(maxiter=maxfev, maxfev=maxfev, xatol=1e-5, fatol=1e-4, adaptive=True))
        r2 = optimize.minimize(f, r1.x, method="Powell", options=dict(maxiter=20000, xtol=1e-5, ftol=1e-7))
        print(f"tur {r}: NM {r1.fun:.3f}  Powell {r2.fun:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if r2.fun < bv - 1e-3: best, bv = r2.x, r2.fun
        else: break
    return best, -bv

def hessian_cov(x, rel=1e-3):
    f = E.logpost; k = len(x); h = np.maximum(np.abs(x) * rel, 1e-4); Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
            Hm[i, j] = Hm[j, i] = -(f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    w, V = np.linalg.eigh(Hm); w = np.maximum(w, w.max() * 1e-6)
    return V @ np.diag(1 / w) @ V.T

if stage == "mode":
    th0 = np.array([q[1] for q in E.priors])
    print("lp0", E.logpost(th0), flush=True)
    best, lp = find_mode(th0)
    json.dump(dict(names=E.names, theta=list(best), lp=lp), open("ms_mode.json", "w"))
    for n_, v in zip(E.names, best): print(f"{n_:12s} {v:9.3f}")
    np.save("ms_cov.npy", hessian_cov(best))
elif stage == "chain":
    seed, nd = int(sys.argv[2]), int(sys.argv[3])
    m = json.load(open("ms_mode.json")); cov = np.load("ms_cov.npy")
    rng = np.random.default_rng(seed); Lc = np.linalg.cholesky(cov)
    x = np.array(m["theta"]) + 0.2 * Lc @ rng.standard_normal(len(m["theta"]))
    lpx = E.logpost(x)
    if not np.isfinite(lpx): x = np.array(m["theta"]); lpx = E.logpost(x)
    draws = np.zeros((nd, len(x))); acc = 0; t0 = time.time(); scale = 0.3
    for it in range(nd):
        y = x + scale * Lc @ rng.standard_normal(len(x)); lpy = E.logpost(y)
        if np.log(rng.uniform()) < lpy - lpx: x, lpx = y, lpy; acc += 1
        draws[it] = x
        if (it + 1) % 1000 == 0:
            print(f"ms_chain{seed} {it+1} acc={acc/(it+1):.3f} lp={lpx:.1f} t={time.time()-t0:.0f}s", flush=True)
            np.save(f"ms_chain{seed}.npy", draws[:it + 1])
    np.save(f"ms_chain{seed}.npy", draws)
