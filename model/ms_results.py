"""Rejim değişkenli model sonuçları: posterior tablo, rejim olasılıkları, rejime göre etki-tepkiler."""
import glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import msdsge as MS
from dsge_est import rhat
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

E = MS.MSEstimator()
BURN = 4000
files = sorted(glob.glob("ms_chain*.npy"))
if files:
    np.save("ms_draws.npy", np.stack([np.load(f)[BURN::5] for f in files]))   # inceltilmiş çekilişler (zincir x çekiliş x param)
ch = list(np.load("ms_draws.npy")); D = np.vstack(ch); R = rhat(ch)
tab = pd.DataFrame({"prior": [q[2] for q in E.priors], "prior_mean": [q[3] for q in E.priors], "post_mean": D.mean(0),
                    "p05": np.percentile(D, 5, 0), "p95": np.percentile(D, 95, 0), "rhat": R}, index=E.names)
tab.to_csv("ms_posterior_table.csv", float_format="%.3f"); print(tab.round(3).to_string())

# rejim olasılıkları (posterior çekilişleri üzerinden ortalama filtrelenmiş olasılık)
rng = np.random.default_rng(5); sel = D[rng.choice(len(D), 150, replace=False)]
PA, PB = [], []
for th in sel:
    ss = E.state_space(th)
    if ss is None: continue
    Ts, Rs, Zs, cs, Pi = ss
    _, filt = MS.kim_filter(Ts, Rs, Zs, cs, E.H, Pi, E.Y)
    if filt is None: continue
    PA.append(filt[:, 2] + filt[:, 3]); PB.append(filt[:, 1] + filt[:, 3])
PA, PB = np.array(PA), np.array(PB); dates = list(E.Ydf.index); x = np.arange(len(dates))
fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
for ax, arr, col, lab in ((axes[0], PA, BLUE, "Oynak (takdire dayalı) para politikası rejimi"), (axes[1], PB, ORANGE, "Yüksek enflasyon fiyatlama rejimi")):
    ax.fill_between(x, *np.percentile(arr, [16, 84], 0), color=col, alpha=0.2, lw=0)
    ax.plot(x, arr.mean(0), color=col, lw=2); ax.set_ylim(-0.02, 1.02); ax.set_ylabel("Olasılık")
    ax.set_title(lab, loc="left", color=INK, fontsize=10.5)
    ax.axvspan(dates.index("2020Q2") - 0.5, dates.index("2020Q4") + 0.5, color=GRAY, alpha=0.12, lw=0)
tk = [i for i, d_ in enumerate(dates) if d_.endswith("Q1") and int(d_[:4]) % 2 == 0]
axes[1].set_xticks(tk); axes[1].set_xticklabels([dates[i][:4] for i in tk])
fig.suptitle("Rejim değişkenli model: filtrelenmiş rejim olasılıkları (posterior ortalaması, %68 bant; gri: salgın çeyrekleri eksik veri)",
             x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("ms_1_rejim_olasiliklari.png", dpi=150); plt.close(fig)

# rejime göre etki-tepkiler (rejimde kalındığı varsayımıyla; beklentiler rejim değişimini içerir)
H = 13; LABS = ["Sistematik politika, normal fiyatlama", "Sistematik politika, yüksek enflasyon", "Oynak politika, normal fiyatlama", "Oynak politika, yüksek enflasyon"]
COLS = [BLUE, ORANGE, AQUA, GRAY]
SH = [("e_m", "Faiz +1 puan"), ("e_rstar", "Kur primi: TL reel %1 değer kaybı"), ("e_oil", "Brent +%10")]
VS = [("pi", 4, "TÜFE enfl. (yıllık puan)"), ("y", 1, "GSYH (%)"), ("i", 4, "Politika faizi (yıllık puan)")]
IX = MS.IX
store = {s: {r: [] for r in range(MS.S)} for s, _ in SH}
for th in sel[:100]:
    plist, Pi, d = E.regime_params(th); sol = MS.solve_ms(plist, Pi)
    if sol is None: continue
    for r in range(MS.S):
        P, Q = sol[r]
        for sh, _ in SH:
            e = np.zeros(len(MS.SHOCKS)); e[MS.SX[sh]] = 1.0
            z = np.zeros((H, MS.N)); z[0] = Q @ e
            for t in range(1, H): z[t] = P @ z[t - 1]
            nv, tgt = {"e_m": ("i", 0.25), "e_rstar": ("q", 1.0), "e_oil": ("oil", 10.0)}[sh]
            if abs(z[0, IX[nv]]) < 1e-10: continue
            store[sh][r].append(z * tgt / z[0, IX[nv]])
fig, axes = plt.subplots(3, 3, figsize=(11, 8.4)); xx = np.arange(H); summ_irf = {}
for i, (sh, sl) in enumerate(SH):
    summ_irf[sh] = {}
    for j, (v, m, vl) in enumerate(VS):
        ax = axes[i, j]; ax.axhline(0, color=MUTED, lw=0.8)
        for r in range(MS.S):
            if not store[sh][r]: continue
            b = m * np.array(store[sh][r])[:, :, IX[v]]
            ax.plot(xx, np.median(b, 0), color=COLS[r], lw=2, label=LABS[r])
            summ_irf[sh].setdefault(LABS[r], {})[v] = {"h0": round(float(np.median(b[:, 0])), 2), "sum8": round(float(np.median(b[:, :8].sum(1))) / (4 if v != "y" else 1), 2)}
        if i == 0: ax.set_title(vl, loc="left", color=INK, fontsize=10)
        if j == 0: ax.set_ylabel(sl, color=INK, fontsize=9)
        ax.set_xlim(0, H - 1); ax.set_xticks([0, 4, 8, 12])
        if i == 2: ax.set_xlabel("Çeyrek")
axes[0, 0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
fig.suptitle("Rejime göre etki-tepkiler (posterior medyan; ajanlar rejim değişimini beklentilerine yansıtır)", x=0.01, ha="left",
             fontsize=11.5, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("ms_2_rejim_irf.png", dpi=150); plt.close(fig)

per = {"2006Q1-2019Q4": slice("2006Q1", "2019Q4"), "2021Q1-2023Q2": slice("2021Q1", "2023Q2"), "2023Q3-2026Q2": slice("2023Q3", "2026Q2")}
idx = pd.Index(dates)
regp = {k: {"oynak_politika": round(float(PA.mean(0)[idx.slice_indexer(v.start, v.stop)].mean()), 2),
            "yuksek_enflasyon": round(float(PB.mean(0)[idx.slice_indexer(v.start, v.stop)].mean()), 2)} for k, v in per.items()}
summ = {"rejim_olasiligi_donem_ort": regp, "irf": summ_irf, "rhat_max": round(float(R.max()), 3),
        "pA": dict(zip(dates, PA.mean(0).round(3).tolist())), "pB": dict(zip(dates, PB.mean(0).round(3).tolist()))}
json.dump(summ, open("ms_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: summ[k] for k in ("rejim_olasiligi_donem_ort", "irf", "rhat_max")}, indent=1, ensure_ascii=False))
