from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, json, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dsge_est import *
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
m2 = json.load(open("s2_mode.json")); FREE = m2["names"]; fixed = m2["fixed"]
import glob
ch = [np.load(f)[10000:] for f in sorted(glob.glob("s2_chain*.npy"))]; D2 = np.vstack(ch); R2 = rhat(ch)
D1 = np.load("s1_draws.npy")
E1 = Estimator("2006Q1", "2019Q4")
allp = {p[0]: p for p in default_priors()}
idx1 = [E1.names.index(k) for k in FREE]
tab = pd.DataFrame({"2006-19 ort": D1[:, idx1].mean(0), "2006-19 %5": np.percentile(D1[:, idx1], 5, 0),
                    "2006-19 %95": np.percentile(D1[:, idx1], 95, 0), "2023-26 ort": D2.mean(0),
                    "2023-26 %5": np.percentile(D2, 5, 0), "2023-26 %95": np.percentile(D2, 95, 0),
                    "Rhat": R2, "P(2023-26 > 2006-19)": [np.mean(D2[np.random.default_rng(k).integers(0, len(D2), 20000), k] > D1[np.random.default_rng(k+99).integers(0, len(D1), 20000), idx1[k]]) for k in range(len(FREE))]},
                   index=FREE)
tab.to_csv("s2_compare_table.csv", float_format="%.3f"); print(tab.round(3).to_string())
# IRF karşılaştırması: aynı yapısal parametreler, iki dönemin posterioru
Hh = 12; rng = np.random.default_rng(3)
def irfs_for(draw_fn, n=300):
    out = {s: [] for s in ("e_m", "e_rstar", "e_oil")}
    sol_ = Solver()
    for _ in range(n):
        p = draw_fn(); sol = sol_.solve(p)
        if sol is None: continue
        P, Q = sol
        for s in out:
            z = np.zeros((Hh, N)); e = np.zeros(len(SHOCKS)); e[SX[s]] = 1.0; z[0] = Q @ e
            for t in range(1, Hh): z[t] = P @ z[t - 1]
            if s == "e_m": z *= 0.25 / z[0, IX["i"]]                # +1 puan faiz
            if s == "e_rstar": z *= 1.0 / z[0, IX["q"]]              # reel kur %1 değer kaybı
            if s == "e_oil": z *= 10.0 / z[0, IX["oil"]]             # Brent +%10
            out[s].append(z)
    return {s: np.array(v) for s, v in out.items()}
def d1():
    return E1.params(D1[rng.integers(len(D1))])
def d2():
    p = dict(BASE_); p.update(fixed); p.update(dict(zip(FREE, D2[rng.integers(len(D2))]))); return p
import tr_dsge2; BASE_ = dict(tr_dsge2.BASE)
I1, I2 = irfs_for(d1), irfs_for(d2)
SH = [("e_m", "Faiz +1 puan"), ("e_rstar", "Kur primi: TL reel %1 değer kaybı"), ("e_oil", "Brent +%10")]
VS = [("y", 1, "GSYH (%)"), ("pi", 4, "TÜFE enfl. (yıllık puan)"), ("i", 4, "Politika faizi (yıllık puan)")]
fig, axes = plt.subplots(3, 3, figsize=(11, 8.2)); x = np.arange(Hh)
for r, (s, sl) in enumerate(SH):
    for c, (v, mlt, vl) in enumerate(VS):
        ax = axes[r, c]; ax.axhline(0, color=MUTED, lw=0.8)
        for I, col, lab in ((I1, BLUE, "2006–2019"), (I2, ORANGE, "2023Ç3–2026Ç2")):
            b = mlt * I[s][:, :, IX[v]]
            ax.fill_between(x, *np.percentile(b, [16, 84], 0), color=col, alpha=0.18, lw=0)
            ax.plot(x, np.median(b, 0), color=col, lw=2, label=lab)
        if r == 0: ax.set_title(vl, loc="left", color=INK, fontsize=10)
        if c == 0: ax.set_ylabel(sl, color=INK, fontsize=9.5)
        ax.set_xlim(0, Hh - 1); ax.set_xticks([0, 4, 8])
        if r == 2: ax.set_xlabel("Çeyrek")
axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
fig.suptitle("Dönemsel karşılaştırma: aynı şoka iki dönemde verilen tepki (medyan, %68 bant)", x=0.01, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("s2_1_donem_irf.png", dpi=150); plt.close(fig)
summ = {}
for s, _ in SH:
    summ[s] = {}
    for v, mlt, _ in VS:
        a1 = np.median(mlt * I1[s][:, :, IX[v]], 0); a2 = np.median(mlt * I2[s][:, :, IX[v]], 0)
        summ[s][v] = {"2006-19_h0": round(a1[0], 2), "2023-26_h0": round(a2[0], 2),
                      "2006-19_sum8": round(a1[:8].sum() / 4 if v != "y" else a1[:8].sum(), 2),
                      "2023-26_sum8": round(a2[:8].sum() / 4 if v != "y" else a2[:8].sum(), 2)}
json.dump(summ, open("s2_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
