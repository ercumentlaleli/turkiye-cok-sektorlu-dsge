"""SSK (yapısal) vs LP-IV (yarı-yapısal) sektörel tepkilerin karşılaştırılması + ağ etkisi testi."""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import lpiv
from network import net, NACE2IO
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
SECT = lpiv.SECT; H = lpiv.H
S = np.load("ssk_results.npz")
out, pooled, rr, F, m = lpiv.run()
res = {"F_first_stage_median": float(F)}

def ssk_med(key): return np.median(S[key], 0)          # (25, 12)
# 1) Sektör ortalaması (eşit ağırlık) — panel karşılaştırması
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6)); hh = np.arange(H + 1)
panels = [("mp", "PP", "e_m_PP", "Yİ-ÜFE seviyesi: faiz +1 puan"), ("mp", "IP", "e_m_IP", "Sanayi üretimi: faiz +1 puan"),
          ("oil", "PP", "e_oil_PP", "Yİ-ÜFE seviyesi: Brent +%10"), ("oil", "IP", "e_oil_IP", "Sanayi üretimi: Brent +%10")]
for ax, (sh, kind, key, title) in zip(axes.flat, panels):
    avg = S[key][:, :, :H + 1].mean(1)                    # çekiliş x ufuk, sektör ortalaması
    lp = pooled[(sh, kind)]
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.fill_between(hh, *np.percentile(avg, [5, 95], 0), color=BLUE, alpha=0.18, lw=0)
    ax.plot(hh, np.median(avg, 0), color=BLUE, lw=2, label="SSK uydu denklemleri (DSGE)")
    ax.fill_between(hh, lp[:, 0] - 1.645 * lp[:, 1], lp[:, 0] + 1.645 * lp[:, 1], color=ORANGE, alpha=0.15, lw=0)
    ax.plot(hh, lp[:, 0], color=ORANGE, lw=2, label="Panel LP-IV" if sh == "mp" else "Panel LP")
    ax.set_title(title, loc="left", color=INK, fontsize=10); ax.set_xlim(0, H); ax.set_xticks(range(0, H + 1, 2))
    ax.set_ylabel("% (seviye)")
    res[f"{sh}_{kind}"] = {"ssk_h4": round(float(np.median(avg, 0)[4]), 2), "lp_h4": round(float(lp[4, 0]), 2),
                           "lp_h4_se": round(float(lp[4, 1]), 2), "ssk_h8": round(float(np.median(avg, 0)[8]), 2),
                           "lp_h8": round(float(lp[8, 0]), 2), "lp_h8_se": round(float(lp[8, 1]), 2)}
for ax in axes[1]: ax.set_xlabel("Çeyrek")
axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
fig.suptitle("Sağlama: sektör ortalaması tepkiler — SSK (%90 posterior bant) ve LP-IV (%90 GA)", x=0.01, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("s4_1_ssk_vs_lp.png", dpi=150); plt.close(fig)

# 2) Kesitsel tutarlılık: sektör sıralamaları
def xsec(sh, kind, key, h):
    a = ssk_med(key)[:, h]; b = out[sh][kind][:, h, 0]
    rho, p = stats.spearmanr(a, b); sign = np.mean(np.sign(a) == np.sign(b))
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b))[0] for _ in range(2000)])
    return dict(spearman=round(float(rho), 2), perm_p=round(float(np.mean(perm >= rho)), 3), isaret_uyumu=round(float(sign), 2))
res["kesit"] = {f"{sh}_{kind}_h{h}": xsec(sh, kind, key, h) for sh, kind, key, _ in panels for h in (2, 4, 8)}

# 3) Ağ etkisi testi: sektörel fiyat tepkilerini doğrudan vs ağ (dolaylı) maruziyetle açıklama
netm = net.set_index("kod")
def exposure(col): return np.array([netm.loc[NACE2IO[s], col] for s in SECT])
def horse(y, d, tot, drop=None):
    msk = np.ones(len(y), bool)
    if drop is not None: msk[SECT.index(drop)] = False
    y, d, ind = y[msk], d[msk], (tot - d)[msk]
    X1 = np.column_stack([np.ones(msk.sum()), d]); X2 = np.column_stack([np.ones(msk.sum()), d, ind])
    b1, r1 = np.linalg.lstsq(X1, y, rcond=None)[:2]; b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    e1 = y - X1 @ b1; e2 = y - X2 @ b2; tss = ((y - y.mean()) ** 2).sum()
    R2a, R2b = 1 - e1 @ e1 / tss, 1 - e2 @ e2 / tss
    s2 = e2 @ e2 / (len(y) - 3); V = s2 * np.linalg.inv(X2.T @ X2)
    Fst = ((e1 @ e1 - e2 @ e2) / 1) / (e2 @ e2 / (len(y) - 3))
    return dict(R2_dogrudan=round(R2a, 2), R2_ag_dahil=round(R2b, 2), b_dolayli=round(b2[2], 2),
                t_dolayli=round(b2[2] / np.sqrt(V[2, 2]), 2), F_p=round(1 - stats.f.cdf(Fst, 1, len(y) - 3), 3))
en_d, en_t = exposure("ith_enerji_dogrudan"), exposure("ith_enerji_toplam")
im_d, im_t = exposure("ithal_dogrudan"), exposure("ithal_toplam")
S_rstar = None
tests = {}
for lab, y in [("SSK_petrol_h4", ssk_med("e_oil_PP")[:, 4]), ("LP_petrol_h4", out["oil"]["PP"][:, 4, 0]),
               ("SSK_petrol_h8", ssk_med("e_oil_PP")[:, 8])]:
    tests[lab] = {"tum": horse(y, en_d, en_t), "C19_haric": horse(y, en_d, en_t, drop="C19")}
for lab, y in [("SSK_faiz_h4", ssk_med("e_m_PP")[:, 4]), ("LP_faiz_h4", out["mp"]["PP"][:, 4, 0]),
               ("SSK_kurprimi_h4", ssk_med("e_rstar_PP")[:, 4])]:
    tests[lab] = {"tum": horse(y, im_d, im_t), "C19_haric": horse(y, im_d, im_t, drop="C19")}
res["ag_testi"] = tests

# 4) Kesit dağılım grafiği (petrol, h=4)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (key, sh, title) in zip(axes, [("e_oil_PP", "oil", "Brent +%10 → Yİ-ÜFE (h=4)"), ("e_m_PP", "mp", "Faiz +1 puan → Yİ-ÜFE (h=4)")]):
    a = ssk_med(key)[:, 4]; b = out[sh]["PP"][:, 4, 0]
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter(a, b, s=36, color=BLUE, edgecolor=SURF, linewidth=1.2, zorder=3)
    for s_, xa, yb in zip(SECT, a, b): ax.annotate(s_, (xa, yb), fontsize=7.5, color=INK2, xytext=(3, 3), textcoords="offset points")
    lim = [min(a.min(), b.min()) - 0.5, max(a.max(), b.max()) + 0.5]; ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    r = res["kesit"][f"{sh}_PP_h4"]
    ax.set_title(f"{title}\nSpearman ρ = {r['spearman']}, işaret uyumu = {r['isaret_uyumu']}", loc="left", color=INK, fontsize=10)
    ax.set_xlabel("SSK (posterior medyan, %)"); ax.set_ylabel("LP sektör tahmini (%)")
fig.tight_layout(); fig.savefig("s4_2_kesit.png", dpi=150); plt.close(fig)
json.dump(res, open("s4_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
