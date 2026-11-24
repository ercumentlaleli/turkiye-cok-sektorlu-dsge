import numpy as np, json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dsge_est import *

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6})

E = Estimator("2006Q1", "2019Q4")
BURN = 10000
chains = [np.load(f"s1_chain{s}.npy")[BURN:] for s in (1, 2)]
draws = np.vstack(chains); R = rhat(chains)
old = pd.read_csv("posterior_table.csv", index_col=0)
rows = []
for k, (nm, _, typ, pm, ps) in enumerate(E.priors):
    d = draws[:, k]
    rows.append(dict(param=nm, prior=typ, prior_mean=pm, post_mean=d.mean(), p05=np.percentile(d, 5),
                     p95=np.percentile(d, 95), rhat=R[k], onceki_model=old.post_mean.get(nm, np.nan)))
tab = pd.DataFrame(rows).set_index("param"); tab.to_csv("s1_posterior_table.csv", float_format="%.3f")
print(tab.round(3).to_string())
np.save("s1_draws.npy", draws)
theta_mean = draws.mean(0)

GROUPS = [("Arz (verimlilik)", ["e_a"]), ("Yurt içi talep", ["e_d"]), ("Kur primi", ["e_prem", "e_rstar"]),
          ("Dış talep (AB-4 öncü gös.)", ["e_ystar"]), ("Para politikası", ["e_m"]), ("Brent petrol", ["e_oil"]),
          ("Yerli maliyet", ["e_cp"])]
GCOL = [CAT[0], CAT[1], CAT[2], CAT[3], CAT[6], CAT[7], CAT[4]]
gi = lambda l: [SHOCKS.index(s) for s in l]

# IRF bantları
SHS = [("e_m", "Para politikası"), ("e_rstar", "Kur primi"), ("e_oil", "Brent petrol"), ("e_cp", "Yerli maliyet")]
VS = [("y", 1, "GSYH (%)"), ("pi", 4, "TÜFE enfl. (yıllık puan)"), ("i", 4, "Politika faizi (yıllık puan)"),
      ("q", 1, "Reel kur (%, + = TL kaybı)")]
Hh = 16; rng = np.random.default_rng(1); sel = draws[rng.choice(len(draws), 400, replace=False)]
irfs = {s: [] for s, _ in SHS}
for th in sel:
    p = E.params(th); sol = E.solver.solve(p)
    if sol is None: continue
    P, Q = sol
    for s, _ in SHS:
        z = np.zeros((Hh, N)); e = np.zeros(len(SHOCKS)); e[SX[s]] = p["sd_" + s[2:]]; z[0] = Q @ e
        for t in range(1, Hh): z[t] = P @ z[t - 1]
        irfs[s].append(z)
fig, axes = plt.subplots(4, 4, figsize=(11, 10)); x = np.arange(Hh)
for r, (s, sl) in enumerate(SHS):
    a = np.array(irfs[s])
    for c, (v, mlt, vl) in enumerate(VS):
        ax = axes[r, c]; b = mlt * a[:, :, IX[v]]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.fill_between(x, *np.percentile(b, [5, 95], 0), color="#b7d3f6", lw=0)
        ax.fill_between(x, *np.percentile(b, [16, 84], 0), color="#86b6ef", lw=0)
        ax.plot(x, np.median(b, 0), color="#1c5cab", lw=2)
        if r == 0: ax.set_title(vl, loc="left", color=INK, fontsize=10)
        if c == 0: ax.set_ylabel(sl + "\n(1 std. şok)", color=INK, fontsize=10)
        ax.set_xlim(0, Hh - 1); ax.set_xticks([0, 4, 8, 12])
        if r == 3: ax.set_xlabel("Çeyrek")
fig.suptitle("Genişletilmiş model (Brent + AB öncü göstergesi + yerli maliyet şoku): etki-tepkiler, %68/%90 bantlar",
             x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig("s1_1_irf.png", dpi=150); plt.close(fig)

# FEVD
p = E.params(theta_mean); P, Q = E.solver.solve(p)
Qs = Q * np.array([p["sd_" + s[2:]] for s in SHOCKS])
def fevd(h):
    acc = np.zeros((N, len(SHOCKS))); Pk = np.eye(N)
    for _ in range(h): acc += (Pk @ Qs) ** 2; Pk = P @ Pk
    return acc / acc.sum(1, keepdims=True)
FV = {h: fevd(h) for h in (1, 4, 8, 20)}
fvv = [("y", "GSYH"), ("pi", "TÜFE enflasyonu"), ("i", "Politika faizi"), ("q", "Reel kur")]
fig, axes = plt.subplots(1, 4, figsize=(11, 4.3), sharey=True); hs = [1, 4, 8, 20]
for ax, (v, vl) in zip(axes, fvv):
    bot = np.zeros(4)
    for (gl, gs), col in zip(GROUPS, GCOL):
        vals = np.array([100 * FV[h][IX[v], gi(gs)].sum() for h in hs])
        ax.bar(range(4), vals, bottom=bot, color=col, width=0.7, edgecolor=SURF, linewidth=1.5, label=gl); bot += vals
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h}Ç" for h in hs]); ax.set_title(vl, loc="left", color=INK)
    ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Tahmin hatası varyansı payı (%)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=7, frameon=False, fontsize=8.3, labelcolor=INK2)
fig.suptitle("Varyans ayrıştırması — genişletilmiş model", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0.08, 1, 0.93)); fig.savefig("s1_2_fevd.png", dpi=150); plt.close(fig)

# Tarihsel ayrıştırma (enflasyon)
T_, R_, Z_ = E.state_space(theta_mean); _, (xs, eps) = E.kalman(T_, R_, Z_, smooth=True)
Tn = len(E.Y); contrib = np.zeros((len(SHOCKS), Tn, len(OBS)))
for j in range(len(SHOCKS)):
    xj = np.zeros(T_.shape[0])
    for t in range(Tn): xj = T_ @ xj + R_[:, j] * eps[t, j]; contrib[j, t] = Z_ @ xj
resid = E.Y - contrib.sum(0); dates = list(E.Ydf.index)
k = OBSN.index("pi")
fig, ax = plt.subplots(figsize=(11, 4.8)); xx = np.arange(Tn); pos = np.zeros(Tn); neg = np.zeros(Tn)
for (gl, gs), col in zip(GROUPS, GCOL):
    v = 4 * contrib[gi(gs), :, k].sum(0)
    ax.bar(xx, np.where(v > 0, v, 0), bottom=pos, color=col, width=0.85, label=gl, edgecolor=SURF, linewidth=0.4)
    ax.bar(xx, np.where(v < 0, v, 0), bottom=neg, color=col, width=0.85, edgecolor=SURF, linewidth=0.4)
    pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
rv = 4 * resid[:, k]
ax.bar(xx, np.where(rv > 0, rv, 0), bottom=pos, color="#c3c2b7", width=0.85, label="Başlangıç koş.")
ax.bar(xx, np.where(rv < 0, rv, 0), bottom=neg, color="#c3c2b7", width=0.85)
ax.plot(xx, 4 * E.Y[:, k], color=INK, lw=1.8, label="Gözlenen")
ax.axhline(0, color=MUTED, lw=0.8)
tk = [i for i, d in enumerate(dates) if d.endswith("Q1") and int(d[:4]) % 2 == 0]
ax.set_xticks(tk); ax.set_xticklabels([dates[i][:4] for i in tk]); ax.grid(axis="x", visible=False)
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.35); ax.set_ylabel("Puan (yıllık, ortalamadan sapma)")
ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left", labelcolor=INK2)
ax.set_title("Enflasyonun tarihsel şok ayrıştırması — genişletilmiş model", x=0, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(); fig.savefig("s1_3_tarihsel_enflasyon.png", dpi=150); plt.close(fig)

summ = {"fevd4": {vl: {gl: round(100 * FV[4][IX[v], gi(gs)].sum(), 1) for gl, gs in GROUPS} for v, vl in fvv},
        "fevd20": {vl: {gl: round(100 * FV[20][IX[v], gi(gs)].sum(), 1) for gl, gs in GROUPS} for v, vl in fvv},
        "irf_peak": {s: {v: float(np.median(m * np.array(irfs[s])[:, :, IX[v]], 0)[np.argmax(np.abs(np.median(m * np.array(irfs[s])[:, :, IX[v]], 0)))]) for v, m, _ in VS} for s, _ in SHS},
        "sm_shock_std": dict(zip(SHOCKS, eps.std(0).round(2).tolist())),
        "hd_pi": {d: {gl: round(4 * contrib[gi(gs), dates.index(d), k].sum(), 2) for gl, gs in GROUPS}
                  for d in ["2008Q2", "2008Q4", "2011Q4", "2014Q4", "2018Q3"]}}
json.dump(summ, open("s1_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
