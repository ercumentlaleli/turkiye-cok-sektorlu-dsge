from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import sys; sys.path.insert(0, str(MODEL_DIR))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tr_dsge import *

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

T = 20
sol = solve(BASE)
print({k: v for k, v in sol.items() if k not in ("P", "Q")})

# para politikası şoku: politika faizi etki anında +1 puan (yıllık) olacak şekilde ölçekle
zm = irf(sol, "e_m", 1.0); m_size = 0.25 / zm[0, IX["i"]]
CASES = [
    ("e_m", m_size, "1_para_politikasi.png", "Şok 1: Para politikası sıkılaştırması (politika faizi +1 puan)"),
    ("e_prem", 1.0, "2_risk_primi.png", "Şok 2: Ülke risk primi şoku (+400 baz puan, yıllık)"),
    ("e_oil", 20.0, "3_enerji.png", "Şok 3: İthal enerji fiyatlarında %20 artış"),
    ("e_ystar", -1.0, "4_dis_talep.png", "Şok 4: Dış talepte %1 düşüş (AB yavaşlaması)"),
]
PANELS = [("y", 1, "GSYH (% sapma)"), ("c", 1, "Özel tüketim (% sapma)"),
          ("pi", 4, "TÜFE enflasyonu (yıllık, puan)"), ("i", 4, "Faiz (yıllık, puan)"),
          ("q", 1, "Reel kur (% , + = TL değer kaybı)"), ("nx", 1, "Net ihracat (GSYH'nin %'si)")]
x = np.arange(T)
summary = {}
for sh, size, fn, title in CASES:
    z = irf(sol, sh, size, T)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))
    for ax, (v, mult, lab) in zip(axes.flat, PANELS):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot(x, mult * z[:, IX[v]], color=BLUE, lw=2, label="Politika faizi" if v == "i" else None)
        if v == "i":
            ax.plot(x, 4 * z[:, IX["rr"]], color=ORANGE, lw=2, label="Reel faiz (ex-ante)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")
        ax.set_title(lab, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, T - 1); ax.set_xticks([0, 4, 8, 12, 16])
    for ax in axes[1]: ax.set_xlabel("Çeyrek")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{MODEL_DIR}/" + fn, dpi=150); plt.close(fig)
    summary[sh] = {v: (round(float((m * z[:, IX[v]]).min()), 2), round(float((m * z[:, IX[v]]).max()), 2),
                       round(float(m * z[0, IX[v]]), 2)) for v, m, _ in PANELS + [("rr", 4, ""), ("de", 4, "")]}
for k, v in summary.items(): print(k, v)
