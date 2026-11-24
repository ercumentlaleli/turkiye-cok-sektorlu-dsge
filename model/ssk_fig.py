import numpy as np, matplotlib, json
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
SECT = ["B"] + [f"C{k}" for k in range(10, 33)] + ["D35"]
LAB = {"B": "B Madencilik", "C10": "C10 Gıda", "C11": "C11 İçecek", "C12": "C12 Tütün", "C13": "C13 Tekstil",
       "C14": "C14 Giyim", "C15": "C15 Deri", "C16": "C16 Ağaç", "C17": "C17 Kağıt", "C18": "C18 Basım",
       "C19": "C19 Rafine petrol", "C20": "C20 Kimya", "C21": "C21 Eczacılık", "C22": "C22 Kauçuk-plastik",
       "C23": "C23 Metalik olmayan min.", "C24": "C24 Ana metal", "C25": "C25 Fabrikasyon metal",
       "C26": "C26 Elektronik", "C27": "C27 Elektrikli teçh.", "C28": "C28 Makine", "C29": "C29 Motorlu taşıt",
       "C30": "C30 Diğer ulaşım", "C31": "C31 Mobilya", "C32": "C32 Diğer imalat", "D35": "D35 Elektrik-gaz"}
S = np.load("ssk_results.npz")
def panel(axes, keys, h, titles, cum=False):
    for ax, key, t in zip(axes, keys, titles):
        a = S[key][:, :, :h + 1].sum(2) if cum else S[key][:, :, h]
        med = np.median(a, 0); lo, hi = np.percentile(a, [16, 84], 0)
        y = np.arange(len(SECT))[::-1]
        col = np.where(med >= 0, "#eb6834", "#2a78d6")
        ax.barh(y, med, color=col, height=0.7, edgecolor=SURF, linewidth=1)
        ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
        ax.axvline(0, color=MUTED, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([LAB[s] for s in SECT])
        ax.set_title(t, loc="left", color=INK, fontsize=9.5); ax.grid(axis="y", visible=False)
fig, axes = plt.subplots(1, 3, figsize=(11, 7.4), sharey=True)
panel(axes, ["e_m_PP", "e_rstar_PP", "e_oil_PP"], 4,
      ["Faiz +1 puan", "Kur primi şoku (1 std)", "Brent +%10"])
for ax in axes: ax.set_xlabel("Yİ-ÜFE seviyesi, 4 çeyrek sonra (%)")
fig.suptitle("SSK uydu denklemleri: NACE-2 üretici fiyatı tepkileri (posterior medyan, %68 aralık)", x=0.01, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("s3_1_ssk_fiyat.png", dpi=150); plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(11, 7.4), sharey=True)
panel(axes, ["e_m_IP", "e_ystar_IP", "e_d_IP"], 0, ["Faiz +1 puan", "Dış talep (1 std)", "Yurt içi talep (1 std)"])
for ax in axes: ax.set_xlabel("Sanayi üretimi, etki anı (%)")
fig.suptitle("SSK uydu denklemleri: NACE-2 sanayi üretimi tepkileri (posterior medyan, %68 aralık)", x=0.01, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("s3_2_ssk_uretim.png", dpi=150); plt.close(fig)
fit = {k: np.median(S["fit" + k], 0) for k in ("PP", "IP")}
print("medyan R2 fiyat", np.median(fit["PP"]).round(2), "üretim", np.median(fit["IP"]).round(2))
for key in ["e_m_PP", "e_rstar_PP", "e_oil_PP"]:
    med = np.median(S[key][:, :, 4], 0); o = np.argsort(med)
    print(key, "en düşük:", [(SECT[i], round(med[i], 2)) for i in o[:3]], "en yüksek:", [(SECT[i], round(med[i], 2)) for i in o[-3:]], "ort", med.mean().round(2))
for key in ["e_m_IP", "e_ystar_IP"]:
    med = np.median(S[key][:, :, 0], 0); o = np.argsort(med)
    print(key, [(SECT[i], round(med[i], 2)) for i in o[:4]], [(SECT[i], round(med[i], 2)) for i in o[-3:]])
