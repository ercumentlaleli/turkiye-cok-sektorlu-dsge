"""
TÜİK 2023 Girdi-Çıktı tabloları: ağ (network) maruziyet ölçüleri
 A_d  : yurt içi girdi katsayıları (64x64), Z_d = Z_toplam - Z_ithal
 m_k  : doğrudan ithal ara girdi payı  (ithal ara girdi / üretim)
 en_k : doğrudan enerji girdi payı (B, C19, D35 satırları; yurt içi + ithal)
 Toplam (ağ dahil) içerik: L' v,  L = (I - A_d)^{-1}  -> fiyat ikiliği: p = L' (doğrudan maliyet şoku)
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, pandas as pd
io = np.load(f"{MODEL_DIR}/data/io/io2023.npz", allow_pickle=True)
Z, Zm, x, F, va, comp = io["Z"], io["Zm"], io["x"], io["F"], io["va"], io["comp"]
codes = [str(c) for c in io["codes"]]; names = [str(n) for n in io["names"]]
n = len(codes)
xs = np.where(x > 0, x, 1.0)
Zd = Z - Zm
A_d = Zd / xs[None, :]                     # a_ij: j'nin 1 TL üretimi için i'den yurt içi girdi
m_dir = Zm.sum(0) / xs
E_ROWS = [codes.index(c) for c in ("B", "C19", "D35")]
en_dir = Z[E_ROWS].sum(0) / xs              # toplam enerji girdisi (yurt içi + ithal)
en_imp_dir = Zm[E_ROWS].sum(0) / xs         # ithal enerji
lab = comp / xs
L = np.linalg.inv(np.eye(n) - A_d)
m_tot = L.T @ m_dir                          # kur şokunun tam (ağ dahil) maliyet geçişi
en_tot = L.T @ np.where(np.isin(np.arange(n), E_ROWS), 0, en_imp_dir)  # ithal enerji fiyatı -> tam içerik
# enerji sektörlerinin kendi fiyatları da ithal enerjiye bağlı; basitlik için ithal enerji payını ağ boyunca yayıyoruz
en_tot = L.T @ en_imp_dir
lab_tot = L.T @ lab
# Ağ merkeziliği (Acemoglu vd. 2012 'etki vektörü') ve yukarı akışlılık (Antràs vd. 2012)
fd = F[:, :-1].sum(1) - F[:, 5] if F.shape[1] > 6 else F.sum(1)   # toplam nihai talep (stok ve toplamlar hariç kaba)
dom_share = np.clip(fd, 0, None) / np.clip(fd, 0, None).sum()
influence = L @ dom_share                    # satıcı tarafı etki (çıktı ağırlıklı)
upstream = np.linalg.inv(np.eye(n) - (Zd / xs[:, None])) @ np.ones(n)   # Antràs: (I-Δ)^{-1} 1, Δ_ij = Zd_ij / x_i
net = pd.DataFrame({"kod": codes, "ad": [s[:45] for s in names], "uretim": x,
                    "ithal_dogrudan": m_dir, "ithal_toplam": m_tot, "enerji_dogrudan": en_dir,
                    "ith_enerji_dogrudan": en_imp_dir, "ith_enerji_toplam": en_tot,
                    "emek_payi": lab, "etki": influence, "yukari_akis": upstream})
net["ithal_dolayli_pay"] = 1 - net.ithal_dogrudan / net.ithal_toplam.replace(0, np.nan)
net["enerji_dolayli_pay"] = 1 - net.ith_enerji_dogrudan / net.ith_enerji_toplam.replace(0, np.nan)

# NACE-2 (veri) -> I-O eşlemesi
NACE2IO = {"B": "B", "C10": "C10-C12", "C11": "C10-C12", "C12": "C10-C12", "C13": "C13-C15", "C14": "C13-C15",
           "C15": "C13-C15", **{f"C{k}": f"C{k}" for k in range(16, 31)}, "C31": "C31_C32", "C32": "C31_C32", "D35": "D35"}

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    w = net.uretim / net.uretim.sum()
    print("Üretim ağırlıklı ortalama: ithal doğrudan %.3f, toplam %.3f  -> dolaylı pay %.2f" %
          ((w * net.ithal_dogrudan).sum(), (w * net.ithal_toplam).sum(),
           1 - (w * net.ithal_dogrudan).sum() / (w * net.ithal_toplam).sum()))
    print("İthal enerji: doğrudan %.3f, toplam %.3f -> dolaylı pay %.2f" %
          ((w * net.ith_enerji_dogrudan).sum(), (w * net.ith_enerji_toplam).sum(),
           1 - (w * net.ith_enerji_dogrudan).sum() / (w * net.ith_enerji_toplam).sum()))
    sub = net[net.kod.isin(set(NACE2IO.values()))]
    print(sub[["kod", "ithal_dogrudan", "ithal_toplam", "ithal_dolayli_pay", "enerji_dogrudan",
               "ith_enerji_toplam", "enerji_dolayli_pay", "yukari_akis"]].round(3).to_string(index=False))
    net.to_csv(f"{MODEL_DIR}/data/io/network_measures.csv", index=False)
