import numpy as np, json, time
from estimate import *
th0 = np.array([p[1] for p in PRIORS])
neg = lambda th: -logpost(th) if np.isfinite(logpost(th)) else 1e10
best = th0; bv = neg(th0)
t0=time.time()
for rnd in range(3):
    r = optimize.minimize(neg, best, method="Nelder-Mead", options=dict(maxiter=6000, maxfev=6000, xatol=1e-5, fatol=1e-4, adaptive=True))
    print(rnd, r.fun, time.time()-t0)
    r2 = optimize.minimize(neg, r.x, method="Powell", options=dict(maxiter=20000, xtol=1e-5, ftol=1e-7))
    print(rnd, 'powell', r2.fun, time.time()-t0)
    if r2.fun < bv - 1e-3: best, bv = r2.x, r2.fun
    else: break
json.dump(dict(theta=list(best), logpost=-bv), open("mode.json","w"))
for n_, v in zip(NAMES, best): print(f"{n_:10s} {v:8.3f}")
