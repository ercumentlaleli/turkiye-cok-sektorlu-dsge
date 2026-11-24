import numpy as np, json, sys, time
from estimate import *
mode = np.array(json.load(open("mode.json"))["theta"])
def hessian(f, x, rel=1e-3):
    k = len(x); h = np.maximum(np.abs(x) * rel, 1e-4); Hm = np.zeros((k, k)); f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
            v = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*h[i]*h[j])
            Hm[i, j] = Hm[j, i] = v
    return Hm
if sys.argv[1] == "hess":
    Hm = -hessian(logpost, mode)
    w, V = np.linalg.eigh(Hm); print("min/max eig", w.min(), w.max())
    w = np.maximum(w, 1e-2 * w.max() / 1e4)
    Sigma = V @ np.diag(1 / w) @ V.T
    np.save("prop_cov.npy", Sigma)
    for n_, s in zip(NAMES, np.sqrt(np.diag(Sigma))): print(f"{n_:10s} {s:.4f}")
else:
    seed, ndraw, scale = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    rng = np.random.default_rng(seed)
    Sigma = np.load("prop_cov.npy"); Lc = np.linalg.cholesky(Sigma)
    import os
    prev = np.load(f"chain{seed}.npy") if (len(sys.argv) > 5 and sys.argv[5] == "cont") else None
    x = prev[-1].copy() if prev is not None else mode + 0.5 * Lc @ rng.standard_normal(K)
    lpx = logpost(x)
    while not np.isfinite(lpx):
        x = mode + 0.2 * Lc @ rng.standard_normal(K); lpx = logpost(x)
    draws = np.zeros((ndraw, K)); lps = np.zeros(ndraw); acc = 0; t0 = time.time()
    for it in range(ndraw):
        y = x + scale * Lc @ rng.standard_normal(K)
        lpy = logpost(y)
        if np.log(rng.uniform()) < lpy - lpx:
            x, lpx = y, lpy; acc += 1
        draws[it] = x; lps[it] = lpx
        if (it + 1) % 2000 == 0:
            print(f"chain{seed} {it+1} acc={acc/(it+1):.3f} lp={lpx:.1f} t={time.time()-t0:.0f}s", flush=True)
            np.save(f"chain{seed}b.npy", draws[:it+1])
    np.save(f"chain{seed}b.npy", draws)
