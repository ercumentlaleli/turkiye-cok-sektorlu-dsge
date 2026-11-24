import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import ms_model as M
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
SECT = M.SECT; K = M.K; H = 17
LAB = {"A": "Tarım", "B": "Madencilik", "C10-12": "Gıda-içecek-tütün", "C13-15": "Tekstil-giyim-deri", "C16": "Ağaç",
       "C17": "Kağıt", "C18": "Basım", "C19": "Rafine petrol", "C20": "Kimya", "C21": "Eczacılık", "C22": "Kauçuk-plastik",
       "C23": "Metalik olm. mineral", "C24": "Ana metal", "C25": "Fabrikasyon metal", "C26": "Elektronik", "C27": "Elektrikli teçh.",
       "C28": "Makine (+onarım)", "C29": "Motorlu taşıt", "C30": "Diğer ulaşım ar.", "C31-32": "Mobilya-diğer imalat",
       "D35": "Elektrik-gaz", "E": "Su-atık", "F": "İnşaat", "G": "Ticaret", "H": "Ulaştırma", "I": "Konaklama-yiyecek",
       "J": "Bilgi-iletişim", "K": "Finans", "L": "Gayrimenkul", "MN": "Mesleki-idari hiz.", "OQ": "Kamu-eğitim-sağlık",
       "RST": "Diğer hizmetler"}
p = M.base_params()
SOL = {"base": M.solve(p)}
theta_h = np.full(K, float((M.C_DOM * M.CALVO).sum()))
SOL["nonet"] = M.solve(p, network=False)
SOL["homog"] = M.solve(p, theta=theta_h)
SOL["nonet_homog"] = M.solve(p, theta=theta_h, network=False)
print({k: (v["n_stable"], round(v["resid"], 12)) for k, v in SOL.items()})

AGG_SHOCKS = [("e_m", "i1", "Faiz +1 puan"), ("e_rstar", "q1", "Kur primi (TL reel %1 değer kaybı)"),
              ("e_oil", "oil10", "Brent +%10"), ("e_ystar", None, "Dış talep (AB-4 öncü gös.) +1 std"), ("e_d", None, "Yurt içi talep +1 std"),
              ("e_cp", None, "Yerli maliyet +1 std")]
RES = {}
for sh, nm, lab in AGG_SHOCKS:
    RES[sh] = {k: M.summarize(M.irf(SOL[k], p, sh, H, nm)) for k in SOL}
for k_ in SECT:
    sh = f"e_u_{k_}"
    RES[sh] = {k: M.summarize(M.irf(SOL[k], p, sh, H)) for k in ("base", "nonet")}

# ---------- Şekil A: sektör x şok ısı haritası (fiyat düzeyi, 4 çeyrek)
mat = np.array([RES[sh]["base"]["P_level"][4] for sh, _, _ in AGG_SHOCKS]).T
mat_rel = mat / np.abs(mat).max(0, keepdims=True)
fig, ax = plt.subplots(figsize=(9.5, 10))
im = ax.imshow(mat_rel, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for i in range(K):
    for j in range(len(AGG_SHOCKS)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.2, color=INK if abs(mat_rel[i, j]) < 0.6 else "white")
ax.set_xticks(range(len(AGG_SHOCKS))); ax.set_xticklabels([l for _, _, l in AGG_SHOCKS], rotation=25, ha="right", fontsize=8.5)
ax.set_yticks(range(K)); ax.set_yticklabels([f"{k} {LAB[k]}" for k in SECT], fontsize=8)
ax.grid(False)
ax.set_title("Çok sektörlü ağlı model: sektörel fiyat düzeyi tepkisi, 4 çeyrek sonra (%)\nRenk: her şok içinde en büyük tepkiye göre ölçekli",
             loc="left", color=INK, fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig("s5_1_isi_haritasi.png", dpi=150); plt.close(fig)

# ---------- Şekil B: karşı-olgusallar
CF = [("base", "Tam model (ağ + heterojen Calvo)", BLUE), ("nonet", "Ağ yok", ORANGE),
      ("homog", "Tek tip Calvo", AQUA), ("nonet_homog", "Ağ yok + tek tip Calvo", "#898781")]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6)); hh = np.arange(H)
cf_tab = {}
for c, (sh, nm, lab) in enumerate(AGG_SHOCKS[:3]):
    for key, l, col in CF:
        z = RES[sh][key]
        axes[0, c].plot(hh, z["cpi_level"], color=col, lw=2, label=l)
        axes[1, c].plot(hh, z["y"], color=col, lw=2, label=l)
        cf_tab[(sh, key)] = dict(cpi_h4=z["cpi_level"][4], cpi_h8=z["cpi_level"][8], cpi_h16=z["cpi_level"][16],
                                 y_cum8=z["y"][:8].sum(), pi_peak=z["pi"][np.argmax(np.abs(z["pi"]))])
    axes[0, c].set_title(f"{lab}\nTÜFE düzeyi (%)", loc="left", color=INK, fontsize=9.5)
    axes[1, c].set_title("GSYH (%)", loc="left", color=INK, fontsize=9.5)
    for r in (0, 1): axes[r, c].axhline(0, color=MUTED, lw=0.8); axes[r, c].set_xlim(0, H - 1)
    axes[1, c].set_xlabel("Çeyrek")
axes[0, 0].legend(frameon=False, fontsize=7.8, labelcolor=INK2)
fig.suptitle("Ağın ve fiyat katılığı heterojenliğinin rolü: karşı-olgusal deneyler", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("s5_2_karsi_olgusal.png", dpi=150); plt.close(fig)

# ---------- Şekil C: sektörel maliyet şoklarının TÜFE'ye etkisi (doğrudan vs ağ)
rows = []
for j, k_ in enumerate(SECT):
    zb = RES[f"e_u_{k_}"]["base"]; zn = RES[f"e_u_{k_}"]["nonet"]
    own = zb["P_level"][4, j]
    direct = (1 - M.ALPHA_C) * M.C_DOM[j] * own                   # kendi fiyatının sepetteki payı kadar
    rows.append(dict(sektor=k_, ad=LAB[k_], cpi_h4=zb["cpi_level"][4], cpi_h4_agyok=zn["cpi_level"][4],
                     dogrudan=direct, ag=zb["cpi_level"][4] - direct, gsyh_cum8=zb["y"][:8].sum(),
                     tuketim_payi=M.C_DOM[j], kd_payi=M.VA_W[j], theta=M.CALVO[j]))
SS = pd.DataFrame(rows).set_index("sektor")
SS["ag_carpani"] = SS.cpi_h4 / SS.dogrudan.replace(0, np.nan)
SS = SS.sort_values("cpi_h4", ascending=False)
top = SS.head(14)[::-1]
fig, ax = plt.subplots(figsize=(10, 6.2))
y = np.arange(len(top))
ax.barh(y, top.dogrudan, color=BLUE, height=0.7, edgecolor=SURF, linewidth=1, label="Doğrudan (tüketim sepetindeki pay)")
ax.barh(y, top.ag, left=top.dogrudan, color=ORANGE, height=0.7, edgecolor=SURF, linewidth=1, label="Ağ üzerinden dolaylı")
ax.set_yticks(y); ax.set_yticklabels([f"{k} {LAB[k]}" for k in top.index]); ax.grid(axis="y", visible=False)
for yi, (v, mlt) in enumerate(zip(top.cpi_h4, top.ag_carpani)):
    ax.text(v + 0.002, yi, f"{v:.3f}  (×{mlt:.1f})" if np.isfinite(mlt) and mlt < 50 else f"{v:.3f}", va="center", fontsize=7.8, color=INK2)
ax.set_xlabel("TÜFE düzeyine etki, 4 çeyrek sonra (puan) — sektörde 1 std'lik (%1) maliyet şoku")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
ax.set_title("Hangi sektördeki maliyet şoku enflasyonu en çok etkiler? (parantez: ağ çarpanı)", loc="left", color=INK,
             fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig("s5_3_sektorel_soklar.png", dpi=150); plt.close(fig)

# ---------- Doğrulama: SSK ile karşılaştırma (sanayi sektörleri)
S = np.load("ssk_results.npz")
SSK_SECT = ["B"] + [f"C{k}" for k in range(10, 33)] + ["D35"]
MAP = {"B": ["B"], "C10-12": ["C10", "C11", "C12"], "C13-15": ["C13", "C14", "C15"], **{f"C{k}": [f"C{k}"] for k in range(16, 31)},
       "C31-32": ["C31", "C32"], "D35": ["D35"]}
val = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (sh, key, title) in zip(axes, [("e_m", "e_m_PP", "Faiz +1 puan"), ("e_oil", "e_oil_PP", "Brent +%10")]):
    a, b, labs = [], [], []
    for k_, mem in MAP.items():
        a.append(RES[sh]["base"]["P_level"][4, SECT.index(k_)])
        b.append(np.mean([np.median(S[key][:, SSK_SECT.index(mm), 4]) for mm in mem])); labs.append(k_)
    a, b = np.array(a), np.array(b); rho, pv = stats.spearmanr(a, b); r_, _ = stats.pearsonr(a, b)
    val[sh] = dict(spearman=round(rho, 2), p=round(pv, 3), pearson=round(r_, 2), ms_ort=round(a.mean(), 2), ssk_ort=round(b.mean(), 2))
    ax.scatter(a, b, s=36, color=BLUE, edgecolor=SURF, linewidth=1.2, zorder=3)
    for l, xa, yb in zip(labs, a, b): ax.annotate(l, (xa, yb), fontsize=7.5, color=INK2, xytext=(3, 3), textcoords="offset points")
    lim = [min(a.min(), b.min()) - 0.3, max(a.max(), b.max()) + 0.3]; ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.set_title(f"{title} → sektör fiyatı (h=4)\nSpearman ρ = {rho:.2f} (p = {pv:.3f})", loc="left", color=INK, fontsize=10)
    ax.set_xlabel("Çok sektörlü ağlı model (%)"); ax.set_ylabel("SSK uydu denklemleri (%)")
fig.tight_layout(); fig.savefig("s5_4_dogrulama.png", dpi=150); plt.close(fig)

# ---------- Kaydet: interaktif gezgin için veri
explorer = {"sectors": SECT, "labels": [LAB[k] for k in SECT], "theta": M.CALVO.round(3).tolist(),
            "c_share": M.C_DOM.round(4).tolist(), "shocks": []}
for sh, nm, lab in AGG_SHOCKS:
    z = RES[sh]["base"]; zn = RES[sh]["nonet"]
    explorer["shocks"].append(dict(id=sh, label=lab, kind="Makro şok", P=z["P_level"].round(3).T.tolist(), Y=z["y_k"].round(3).T.tolist(),
                                   P_nonet=zn["P_level"].round(3).T.tolist(),
                                   macro=dict(cpi=z["cpi_level"].round(3).tolist(), pi=z["pi"].round(3).tolist(), y=z["y"].round(3).tolist(),
                                              i=z["i"].round(3).tolist(), q=z["q"].round(3).tolist())))
for k_ in SECT:
    sh = f"e_u_{k_}"; z = RES[sh]["base"]; zn = RES[sh]["nonet"]
    explorer["shocks"].append(dict(id=sh, label=f"{k_} {LAB[k_]}: %1 maliyet şoku", kind="Sektörel şok", P=z["P_level"].round(4).T.tolist(),
                                   Y=z["y_k"].round(4).T.tolist(), P_nonet=zn["P_level"].round(4).T.tolist(),
                                   macro=dict(cpi=z["cpi_level"].round(4).tolist(), pi=z["pi"].round(4).tolist(), y=z["y"].round(4).tolist(),
                                              i=z["i"].round(4).tolist(), q=z["q"].round(4).tolist())))
json.dump(explorer, open("explorer_data.json", "w"), ensure_ascii=False)
summ = {"karsi_olgusal": {f"{a}|{b}": {k: round(float(v), 3) for k, v in d.items()} for (a, b), d in cf_tab.items()},
        "dogrulama": val, "sektorel_top": SS.head(10)[["ad", "cpi_h4", "cpi_h4_agyok", "dogrudan", "ag_carpani", "gsyh_cum8", "tuketim_payi", "theta"]].round(4).to_dict("index"),
        "ag_payi_ortalama": round(float((SS.ag / SS.cpi_h4).median()), 2)}
json.dump(summ, open("s5_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
SS.round(4).to_csv("s5_sektorel_sok_tablosu.csv")
