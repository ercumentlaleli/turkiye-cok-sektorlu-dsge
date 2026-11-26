import sys, json, numpy as np
from dsge_est import *
E = Estimator("2006Q1", "2019Q4")
stage = sys.argv[1]
if stage == "mode":
    th0 = np.array([p[1] for p in E.priors])
    start = sys.argv[2] if len(sys.argv) > 2 else "prior"
    if start == "post":   # önceki posterior ortalamalarından başla (yeni parametreler önsel ortalamasında)
        import pandas as pd
        pm = pd.read_csv("s1_posterior_table.csv", index_col=0).post_mean
        th0 = np.array([pm.get(n_, p_[1]) for n_, p_ in zip(E.names, E.priors)])
    print("lp0", E.logpost(th0))
    best, lp = E.find_mode(th0)
    out = "s1_mode.json" if start == "prior" else f"s1_mode_{start}.json"
    json.dump(dict(names=E.names, theta=list(best), lp=lp), open(out, "w"))
    for n_, v in zip(E.names, best): print(f"{n_:10s} {v:9.3f}")
    cov = E.hessian_cov(best); np.save("s1_cov.npy" if start == "prior" else f"s1_cov_{start}.npy", cov)
    print("sd:", dict(zip(E.names, np.sqrt(np.diag(cov)).round(3))))
elif stage == "chain":
    seed, nd = int(sys.argv[2]), int(sys.argv[3])
    m = json.load(open("s1_mode.json")); cov = np.load("s1_cov.npy")
    x0 = np.array(m["theta"]) + 0.3 * np.linalg.cholesky(cov) @ np.random.default_rng(seed + 10).standard_normal(len(m["theta"]))
    while not np.isfinite(E.logpost(x0)): x0 = np.array(m["theta"])
    E.rwmh(x0, cov, nd, 0.28, seed, f"s1_chain{seed}.npy")
