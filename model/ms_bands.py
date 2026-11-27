"""Çok sektörlü model için posterior belirsizlik bantları.
2006–2019 posterior çekilişlerinden (s1_draws.npy) rastgele seçilen parametre vektörleriyle model yeniden çözülür;
her şok için sektörel fiyat düzeyi, sektörel üretim ve makro tepkilerin %68 ve %90 bantları explorer_data.json'a eklenir.
Merkez değer, posterior ortalama parametrelerle elde edilen çözümdür (ms_experiments.py)."""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, str(MODEL_DIR))
import ms_model as M

NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 200
H = 17
NORM = {"e_m": "i1", "e_rstar": "q1", "e_oil": "oil10", "e_ystar": "ystar1", "e_d": "c1", "e_cp": "piH1", "e_mu": "inv1", "e_nw": "spr1"}
MAC = ["cpi_level", "pi", "y", "inv", "i", "q"]
MKEY = {"cpi_level": "cpi"}

names = list(pd.read_csv(MODEL_DIR / "s1_posterior_table.csv", index_col=0).index)
D = np.load(MODEL_DIR / "s1_draws.npy")
rng = np.random.default_rng(11)
sel = D[rng.choice(len(D), NDRAW, replace=False)]
ex = json.load(open(MODEL_DIR / "explorer_data.json", encoding="utf-8"))
ids = [s["id"] for s in ex["shocks"]]
store = {i: {"P": [], "Y": [], **{m: [] for m in MAC}} for i in ids}
P0 = None; ok = 0; t0 = time.time()
for th in sel:
    p = M.base_params(); p.update(dict(zip(names, th)))
    try:
        sol = M.solve_qz(p)
    except RuntimeError:
        continue
    if sol["n_stable"] != M.N: continue
    P0 = sol["P"]; ok += 1
    for i in ids:
        z = M.summarize(M.irf(sol, p, i, H, NORM.get(i)))
        store[i]["P"].append(z["P_level"].T); store[i]["Y"].append(z["y_k"].T)
        for m in MAC: store[i][m].append(z[m])
print(f"{ok}/{NDRAW} çekiliş çözüldü ({time.time() - t0:.0f} sn)")
Q = [5, 16, 84, 95]
for s in ex["shocks"]:
    st = store[s["id"]]; nd = 4 if s["kind"] == "Sektörel şok" else 3
    s["P_band"] = np.round(np.percentile(np.array(st["P"]), Q, axis=0), nd).tolist()     # [q][sektör][ufuk]
    s["Y_band"] = np.round(np.percentile(np.array(st["Y"]), Q, axis=0), nd).tolist()
    s["macro_band"] = {MKEY.get(m, m): np.round(np.percentile(np.array(st[m]), Q, axis=0), nd).tolist() for m in MAC}
ex["band_info"] = dict(n_draws=ok, quantiles=Q, center="posterior ortalama parametrelerle çözüm")
json.dump(ex, open(MODEL_DIR / "explorer_data.json", "w", encoding="utf-8"), ensure_ascii=False)
print("explorer_data.json güncellendi")
