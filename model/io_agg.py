"""TÜİK 2023 G-Ç tablosunu 32 model sektörüne toplar; maliyet ve talep paylarını üretir."""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, pandas as pd, glob, json
T = pd.read_excel(glob.glob(f"{MODEL_DIR}/data/io/73fb*.xls")[0], header=None)
IM = pd.read_excel(glob.glob(f"{MODEL_DIR}/data/io/0b45*.xls")[0], header=None)
num = lambda df, r0, r1, c0, c1: df.iloc[r0:r1, c0:c1].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
codes = [str(c).strip() for c in T.iloc[9:73, 1]]
Z, Zm = num(T, 9, 73, 3, 67), num(IM, 9, 73, 3, 67)
Ft, Fm = num(T, 9, 73, 68, 76), num(IM, 9, 73, 68, 76)      # 0:HH 1:NPISH 2:Devlet 3:NihaiTük 4:GSSO 5:Stok 6:TopSermaye 7:İhracat
x = num(T, 84, 85, 3, 67).ravel(); va = num(T, 83, 84, 3, 67).ravel(); tax = num(T, 74, 75, 3, 67).ravel()

GROUP = {}
def g(name, members):
    for m in members: GROUP[m] = name
g("A", ["A01", "A02", "A03"]); g("B", ["B"]); g("C10-12", ["C10-C12"]); g("C13-15", ["C13-C15"])
for k in range(16, 31): g(f"C{k}", [f"C{k}"])
g("C28", ["C28", "C33"]); g("C31-32", ["C31_C32"]); g("D35", ["D35"]); g("E", ["E36", "E37-E39"]); g("F", ["F"])
g("G", ["G45", "G46", "G47"]); g("H", ["H49", "H50", "H51", "H52", "H53"]); g("I", ["I"])
g("J", ["J58", "J59_J60", "J61", "J62_J63"]); g("K", ["K64", "K65", "K66"]); g("L", ["L68B", "L68A"])
g("MN", ["M69_M70", "M71", "M72", "M73", "M74_M75", "N77", "N78", "N79", "N80-N82"])
g("OQ", ["O84", "P85", "Q86", "Q87_Q88"]); g("RST", ["R90-R92", "R93", "S94", "S95", "S96", "T"])
SECTORS = ["A", "B", "C10-12", "C13-15"] + [f"C{k}" for k in range(16, 31)] + ["C31-32", "D35", "E", "F", "G", "H", "I", "J", "K", "L", "MN", "OQ", "RST"]
assert set(GROUP) == set(codes), set(codes) - set(GROUP)
K = len(SECTORS)
M = np.zeros((K, len(codes)))
for j, c in enumerate(codes): M[SECTORS.index(GROUP[c]), j] = 1
Zk, Zmk = M @ Z @ M.T, M @ Zm @ M.T
Ftk, Fmk = M @ Ft, M @ Fm
xk, vak, taxk = M @ x, M @ va, M @ tax
Zdk = Zk - Zmk; Fdk = Ftk - Fmk
EN = [SECTORS.index(s) for s in ("B", "C19", "D35")]
# Maliyet payları (vergiler hariç, toplam=1'e normalize)
dom_in = Zdk                              # yurt içi girdi (satır=satıcı, sütun=alıcı)
imp_en = Zmk[EN].sum(0); imp_ne = Zmk.sum(0) - imp_en
tot = dom_in.sum(0) + imp_en + imp_ne + vak
w_d = dom_in / tot[None, :]; w_me = imp_en / tot; w_m = imp_ne / tot; w_l = vak / tot
# Talep payları (yurt içi üretimin kullanımı)
d_int = Zdk / xk[:, None]                 # δ_kj: k'nin üretiminin j'ye ara girdi olarak giden payı
s_c = Fdk[:, 0] / xk; s_x = Fdk[:, 7] / xk; s_o = 1 - d_int.sum(1) - s_c - s_x
c_dom = Fdk[:, 0] / Fdk[:, 0].sum()       # yurt içi mallar tüketim sepeti payları
alpha_c = Fmk[:, 0].sum() / (Fdk[:, 0].sum() + Fmk[:, 0].sum())
va_w = vak / vak.sum()
out = dict(SECTORS=SECTORS, w_d=w_d.tolist(), w_me=w_me.tolist(), w_m=w_m.tolist(), w_l=w_l.tolist(),
           d_int=d_int.tolist(), s_c=s_c.tolist(), s_x=s_x.tolist(), s_o=s_o.tolist(), c_dom=c_dom.tolist(),
           alpha_c=alpha_c, va_w=va_w.tolist(), x=xk.tolist())
json.dump(out, open(f"{MODEL_DIR}/data/io/io32.json", "w"))
if __name__ == "__main__":
    df = pd.DataFrame({"yurtici_girdi": w_d.sum(0), "ithal_enerji": w_me, "ithal_diger": w_m, "katma_deger": w_l,
                       "ara_satis": d_int.sum(1), "hh_tuketim": s_c, "ihracat": s_x, "diger_nihai": s_o,
                       "tuketim_sepeti": c_dom, "KD_agirlik": va_w}, index=SECTORS)
    print(df.round(3).to_string()); print("tüketimde ithal payı:", round(alpha_c, 3))
