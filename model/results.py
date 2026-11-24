from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from estimate import *
from estimate import _P_cache
from tr_dsge import SX

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

BURN = 15000
chains = [np.vstack([np.load(f"chain{s}.npy"), np.load(f"chain{s}b.npy")])[BURN:] for s in (1, 2)]
draws = np.vstack(chains)
print("çekiliş sayısı", draws.shape)

# ---------------- yakınsama: Gelman-Rubin R-hat
def rhat(chs):
    n = min(len(c) for c in chs); chs = [c[:n] for c in chs]
    m = np.array([c.mean(0) for c in chs]); v = np.array([c.var(0, ddof=1) for c in chs])
    W = v.mean(0); B = n * m.var(0, ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)
R = rhat(chains)

rows = []
for k, (nm, _, typ, pm, ps) in enumerate(PRIORS):
    d = draws[:, k]
    rows.append(dict(param=nm, prior=typ, prior_mean=pm, prior_sd=ps if typ != "invg" else np.inf,
                     calib=BASE.get(nm, np.nan), mode=json.load(open("mode.json"))["theta"][k],
                     post_mean=d.mean(), p05=np.percentile(d, 5), p95=np.percentile(d, 95), rhat=R[k]))
tab = pd.DataFrame(rows).set_index("param")
tab.to_csv("posterior_table.csv", float_format="%.3f")
print(tab.round(3).to_string())

theta_mean = draws.mean(0)

# ---------------- IRF bantları
SHOCK_SET = [("e_m", "Para politikası"), ("e_rstar", "Kur primi (UIP)"), ("e_d", "Yurt içi talep"), ("e_a", "Arz (verimlilik/maliyet)")]
VAR_SET = [("y", 1, "GSYH (%)"), ("pi", 4, "TÜFE enfl. (yıllık puan)"), ("i", 4, "Politika faizi (yıllık puan)"),
           ("q", 1, "Reel kur (%, + = TL kaybı)")]
Hh = 16
rng = np.random.default_rng(0)
sel = draws[rng.choice(len(draws), 500, replace=False)]
irfs = {s: [] for s, _ in SHOCK_SET}
_P_cache["P"] = None
for th in sel:
    p = theta_to_p(th); sol = solve_fast(p)
    if sol is None: continue
    P, Q = sol
    for s, _ in SHOCK_SET:
        z = np.zeros((Hh, N)); e = np.zeros(len(SHOCKS)); e[SX[s]] = p["sd_" + s[2:]]
        z[0] = Q @ e
        for t in range(1, Hh): z[t] = P @ z[t - 1]
        irfs[s].append(z)
fig, axes = plt.subplots(len(SHOCK_SET), len(VAR_SET), figsize=(11, 10))
x = np.arange(Hh)
for r, (s, slab) in enumerate(SHOCK_SET):
    arr = np.array(irfs[s])
    for c, (v, m, vlab) in enumerate(VAR_SET):
        ax = axes[r, c]; a = m * arr[:, :, IX[v]]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.fill_between(x, np.percentile(a, 5, 0), np.percentile(a, 95, 0), color="#b7d3f6", lw=0)
        ax.fill_between(x, np.percentile(a, 16, 0), np.percentile(a, 84, 0), color="#86b6ef", lw=0)
        ax.plot(x, np.median(a, 0), color="#1c5cab", lw=2)
        if r == 0: ax.set_title(vlab, fontsize=10, color=INK, loc="left")
        if c == 0: ax.set_ylabel(slab + "\n(1 std. şok)", color=INK, fontsize=10)
        ax.set_xlim(0, Hh - 1); ax.set_xticks([0, 4, 8, 12])
        if r == len(SHOCK_SET) - 1: ax.set_xlabel("Çeyrek")
fig.suptitle("Tahmin edilen etki-tepkiler: medyan, %68 ve %90 posterior bantları (2006Ç1–2019Ç4)",
             x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig("est_1_irf_bantlari.png", dpi=150); plt.close(fig)

# ---------------- Varyans ayrıştırması (posterior ortalamada, 1/4/8/20 çeyrek)
_P_cache["P"] = None
p = theta_to_p(theta_mean); P, Q = solve_fast(p)
sds = np.array([p["sd_" + s[2:]] for s in SHOCKS]); Qs = Q * sds
SH_LAB = {"e_a": "Verimlilik", "e_d": "Yurt içi talep", "e_prem": "Risk primi", "e_oil": "Enerji",
          "e_ystar": "Dış talep", "e_rstar": "Dış faiz", "e_m": "Para politikası"}
GROUPS = [("Arz (verimlilik/maliyet)", ["e_a"]), ("Yurt içi talep", ["e_d"]),
          ("Kur primi (risk primi + dış faiz)", ["e_prem", "e_rstar"]), ("Dış talep", ["e_ystar"]),
          ("Para politikası", ["e_m"]), ("Enerji", ["e_oil"])]
GCOL = [CAT[0], CAT[1], CAT[2], CAT[3], CAT[6], CAT[4]]
def gidx(lst): return [SHOCKS.index(s) for s in lst]
def fevd(h):
    out = {}
    acc = np.zeros((N, len(SHOCKS))); Pk = np.eye(N)
    for k in range(h):
        acc += (Pk @ Qs) ** 2; Pk = P @ Pk
    return acc / acc.sum(1, keepdims=True)
FV = {h: fevd(h) for h in (1, 4, 8, 20)}
fv_vars = [("y", "GSYH"), ("pi", "TÜFE enflasyonu"), ("i", "Politika faizi"), ("q", "Reel kur")]
fig, axes = plt.subplots(1, 4, figsize=(11, 4.2), sharey=True)
hs = [1, 4, 8, 20]
for ax, (v, vl) in zip(axes, fv_vars):
    bottom = np.zeros(len(hs))
    for g, ((gl, gs), col) in enumerate(zip(GROUPS[:5], GCOL)):
        vals = np.array([100 * FV[h][IX[v], gidx(gs)].sum() for h in hs])
        ax.bar(range(len(hs)), vals, bottom=bottom, color=col, width=0.7, edgecolor=SURF, linewidth=1.5, label=gl)
        bottom += vals
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"{h}Ç" for h in hs]); ax.set_title(vl, loc="left", color=INK)
    ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Tahmin hatası varyansı payı (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2)
fig.text(0.01, 0.885, "Enerji şokunun payı ~%0 (gözlenen petrol fiyatı yok), gösterilmedi.", ha="left", fontsize=8, color=MUTED)
fig.suptitle("Varyans ayrıştırması: dalgalanmayı hangi şoklar sürüklüyor?", x=0.01, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0.08, 1, 0.93)); fig.savefig("est_2_varyans_ayristirmasi.png", dpi=150); plt.close(fig)

# ---------------- Tarihsel ayrıştırma (posterior ortalamada, düzleştirilmiş şoklar)
_P_cache["P"] = None
T_, R_, Z_ = state_space(theta_mean)
ll, (xs, eps) = kalman(Y, T_, R_, Z_, H, smooth=True)
Tn = len(Y); contrib = np.zeros((len(SHOCKS), Tn, 5))
for j in range(len(SHOCKS)):
    xj = np.zeros(T_.shape[0])
    for t in range(Tn):
        xj = T_ @ xj + R_[:, j] * eps[t, j]
        contrib[j, t] = Z_ @ xj
resid = Y - contrib.sum(0)
dates = Ydf.index
def hd_plot(col, scale, title, fn, ylab):
    k = OBS.index(col)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xx = np.arange(Tn)
    pos = np.zeros(Tn); neg = np.zeros(Tn)
    for (gl, gs), col in zip(GROUPS[:5], GCOL):
        v = scale * contrib[gidx(gs + (["e_oil"] if gl == "Arz (verimlilik/maliyet)" else [])), :, k].sum(0)
        ax.bar(xx, np.where(v > 0, v, 0), bottom=pos, color=col, width=0.85, label=gl, edgecolor=SURF, linewidth=0.4)
        ax.bar(xx, np.where(v < 0, v, 0), bottom=neg, color=col, width=0.85, edgecolor=SURF, linewidth=0.4)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    rv = scale * resid[:, k]
    ax.bar(xx, np.where(rv > 0, rv, 0), bottom=pos, color="#c3c2b7", width=0.85, label="Başlangıç koş. + ölçüm hatası")
    ax.bar(xx, np.where(rv < 0, rv, 0), bottom=neg, color="#c3c2b7", width=0.85)
    ax.plot(xx, scale * Y[:, k], color=INK, lw=1.8, label="Gözlenen (ortalamadan sapma)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ticks = [i for i, d in enumerate(dates) if d.endswith("Q1") and int(d[:4]) % 2 == 0]
    ax.set_xticks(ticks); ax.set_xticklabels([dates[i][:4] for i in ticks])
    ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", labelcolor=INK2)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.35)
    ax.set_title(title, x=0, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)
hd_plot("pi", 4, "Enflasyonun tarihsel şok ayrıştırması (yıllıklandırılmış, örneklem ortalamasından sapma)",
        "est_3_tarihsel_enflasyon.png", "Puan")
hd_plot("dy", 1, "GSYH büyümesinin tarihsel şok ayrıştırması (çeyreklik %, ortalamadan sapma)",
        "est_4_tarihsel_buyume.png", "Puan")

# özet sayılar
summ = {}
for s, _ in SHOCK_SET:
    a = np.array(irfs[s])
    summ[s] = {v: dict(med_peak=float(np.median(m * a[:, :, IX[v]], 0)[np.argmax(np.abs(np.median(m * a[:, :, IX[v]], 0)))]),
                       peak_q=int(np.argmax(np.abs(np.median(m * a[:, :, IX[v]], 0)))))
               for v, m, _ in VAR_SET}
print(json.dumps(summ, indent=1))
for h in (4, 20):
    print("FEVD", h); print(pd.DataFrame(100 * FV[h][[IX[v] for v, _ in fv_vars]], index=[l for _, l in fv_vars],
                                       columns=[SH_LAB[s] for s in SHOCKS]).round(1).to_string())
print("means", MEANS.round(3).to_dict())
# en büyük tarihsel katkılar (enflasyon) seçili dönemler
for d_ in ["2008Q3", "2011Q4", "2016Q4", "2018Q3", "2018Q4", "2019Q3"]:
    t = list(dates).index(d_)
    print(d_, "pi obs", round(4 * Y[t, 2], 2), {SH_LAB[s]: round(4 * contrib[j, t, 2], 2) for j, s in enumerate(SHOCKS)})
print("sm. shocks std", eps.std(0).round(2))
