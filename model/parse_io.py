from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import pandas as pd, numpy as np, glob
def rd(g):
    return pd.read_excel(glob.glob(str(MODEL_DIR / 'data' / 'io' / g))[0], header=None)
T = rd('73fb*'); IM = rd('0b45*'); IU = rd('3008*')
codes = [str(c).strip() for c in T.iloc[9:73, 1]]
names = [str(s).split('\n')[0].strip() for s in T.iloc[9:73, 2]]
num = lambda df, r0, r1, c0, c1: df.iloc[r0:r1, c0:c1].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
Z = num(T, 9, 73, 3, 67)           # toplam ara kullanım (ürün x ürün)
F = num(T, 9, 73, 68, 76)          # nihai talep sütunları ... toplam kullanım
Zm = num(IM, 9, 73, 3, 67)         # ithal ara kullanım
x = num(T, 84, 85, 3, 67).ravel()  # üretim
imp = num(T, 85, 86, 3, 67).ravel()
va = num(T, 83, 84, 3, 67).ravel(); comp = num(T, 76, 77, 3, 67).ravel()
supply = num(T, 86, 87, 3, 67).ravel()
totuse = F[:, -1]
print("ürün sayısı", len(codes), "çıktı toplamı", x.sum()/1e9, "milyar? birim")
print("arz=kullanım kontrolü maks fark %:", np.nanmax(np.abs(supply - totuse) / np.maximum(supply, 1)) * 100)
print("sütun dengesi: ara+vergi+KD ≈ üretim, maks fark %:",
      np.nanmax(np.abs(num(T,75,76,3,67).ravel() + va - x) / np.maximum(x,1)) * 100)
print("ithal ara toplamı / toplam ara:", Zm.sum() / Z.sum())
np.savez(MODEL_DIR / 'data' / 'io' / 'io2023.npz', Z=Z, Zm=Zm, F=F, x=x, imp=imp, va=va, comp=comp, codes=codes, names=names,
         fd_cols=[str(T.iloc[5, c]).split('\n')[0] for c in range(68, 76)])
for k in range(64): print(k, codes[k], names[k][:50], round(x[k]/1e6,1))
