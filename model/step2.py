"""Dönemsel karşılaştırma: 2023Ç3–2026Ç2 (ortodoks para politikası dönemi, 12 gözlem).
Politika kuralı, fiyat katılıkları/endeksleme ve şok varyansları yeniden tahmin edilir;
diğer yapısal parametreler 2006–2019 posterior ortalamasında sabit tutulur. Önseller 1. dönemle aynı."""
import sys, json, numpy as np, pandas as pd
from dsge_est import *
s1 = pd.read_csv("s1_posterior_table.csv", index_col=0)
FREE = ["thH", "thF", "dH", "dF", "rho_i", "phi_pi", "phi_dy", "phi_e",
        "sd_a", "sd_d", "sd_prem", "sd_oil", "sd_ystar", "sd_rstar", "sd_m", "sd_cp"]
allp = {p[0]: p for p in default_priors()}
pri = [allp[k] for k in FREE]
fixed = {k: float(s1.post_mean[k]) for k in s1.index if k not in FREE}
E = Estimator(sys.argv[2] if len(sys.argv) > 2 else "2023Q3", "2026Q2", priors=pri, fixed=fixed)
if sys.argv[1] == "mode":
    th0 = np.array([s1.post_mean[k] for k in FREE])
    print("lp0", E.logpost(th0), "T=", len(E.Y)); print(E.means.round(2).to_dict())
    best, lp = E.find_mode(th0, rounds=4)
    json.dump(dict(names=FREE, theta=list(best), lp=lp, fixed=fixed), open("s2_mode.json", "w"))
    for n_, v in zip(FREE, best): print(f"{n_:8s} {v:8.3f}  (2006-19: {s1.post_mean[n_]:.3f})")
    np.save("s2_cov.npy", E.hessian_cov(best))
else:
    seed = int(sys.argv[3]); m = json.load(open("s2_mode.json")); cov = np.load("s2_cov.npy")
    E.rwmh(np.array(m["theta"]), cov, int(sys.argv[4]) if len(sys.argv) > 4 else 60000, 0.35, seed, f"s2_chain{seed}.npy")
