"""
Sektörel Calvo parametrelerinin tahmini (çeyreklik, 2006Ç1–2019Ç4)
Calvo kısmi uyarlama: p_t = θ p_{t-1} + (1-θ) p*_t  ->  Δp_t = a + b·t + (1-θ)(c_t - p_{t-1}) + e_t
c_t: G-Ç maliyet payları ile kurulan nominal maliyet endeksi (log):
     Σ_j w_d[j,k] p_j  +  w_m[k]·(e+p*)  +  w_me[k]·(e+p*+Brent$)  +  w_l[k]·TÜFE (ücret/KD vekili)
     (kendi sektör girdisi p_{k,t-1} ile — eşanlılığı önlemek için)
(e+p*) = log TÜFE - log REK (TL cinsinden yabancı fiyat düzeyi)
Fiyat serileri: sanayi için NACE-2 Yİ-ÜFE (toplulaştırılmış), tarım ve hizmetler için TÜFE alt endeksleri.
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, pandas as pd, json
io = json.load(open(f"{MODEL_DIR}/data/io/io32.json")); SECT = io["SECTORS"]; K = len(SECT)
w_d, w_m, w_me, w_l = (np.array(io[k]) for k in ("w_d", "w_m", "w_me", "w_l"))
s = pd.read_csv(f"{MODEL_DIR}/data/sectors.csv", index_col=0)
c = pd.read_csv(f"{MODEL_DIR}/data/cpi_groups.csv", index_col=0)
d = pd.read_csv(f"{MODEL_DIR}/data/evds_raw.csv", index_col=0).loc["2005Q4":"2019Q4"]
f = pd.read_csv(f"{MODEL_DIR}/data/foreign.csv", index_col=0).loc["2005Q4":"2019Q4"]
L = lambda x: np.log(x)
def wavg(cols, w): return sum(wi * L(s[ci]) for ci, wi in zip(cols, w))
P = pd.DataFrame(index=s.index)
P["A"] = L(c.CP_A); P["B"] = L(s.PP_B)
P["C10-12"] = wavg(["PP_C10", "PP_C11", "PP_C12"], [0.8, 0.1, 0.1])
P["C13-15"] = wavg(["PP_C13", "PP_C14", "PP_C15"], [0.45, 0.4, 0.15])
for k in range(16, 31): P[f"C{k}"] = L(s[f"PP_C{k}"])
P["C31-32"] = wavg(["PP_C31", "PP_C32"], [0.6, 0.4]); P["D35"] = L(s.PP_D35)
P["E"] = L(c.CP_SRV); P["F"] = (L(s.PP_C23) + L(s.PP_C25) + L(c.CP_SRV)) / 3
P["G"] = L(c.CP_SRV); P["H"] = L(c.CP_TRANS); P["I"] = L(c.CP_HOTEL); P["J"] = L(c.CP_COMM)
P["K"] = L(c.CP_OTHSRV); P["L"] = L(c.CP_RENT); P["MN"] = L(c.CP_OTHSRV); P["OQ"] = L(c.CP_SRV); P["RST"] = L(c.CP_OTHSRV)
SRC = {"A": "TÜFE işlenmemiş gıda", "E": "TÜFE hizmet", "F": "Yİ-ÜFE C23,C25 + TÜFE hizmet", "G": "TÜFE hizmet",
       "H": "TÜFE ulaştırma hiz.", "I": "TÜFE lokanta-otel", "J": "TÜFE haberleşme", "K": "TÜFE diğer hiz.",
       "L": "TÜFE kira", "MN": "TÜFE diğer hiz.", "OQ": "TÜFE hizmet", "RST": "TÜFE diğer hiz."}
P = P[SECT]
cpi = L(d.cpi_sa); fx_star = cpi - L(d.reer); brent = L(f.brent)
res = []
for k, sec in enumerate(SECT):
    p = P[sec]
    dom = sum(w_d[j, k] * (P.iloc[:, j] if j != k else p.shift(1)) for j in range(K))
    cost = dom + w_m[k] * fx_star + w_me[k] * (fx_star + brent) + w_l[k] * cpi
    cost = cost / (w_d[:, k].sum() + w_m[k] + w_me[k] + w_l[k])
    df = pd.DataFrame({"dp": p.diff(), "gap": cost - p.shift(1)}).loc["2006Q1":"2019Q4"].dropna()
    t = np.arange(len(df))
    X = np.column_stack([np.ones(len(df)), t, df.gap.values]); y = df.dp.values
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    V = (e @ e / (len(y) - 3)) * np.linalg.inv(X.T @ X)
    gam = b[2]; se = np.sqrt(V[2, 2])
    theta = float(np.clip(1 - gam, 0.05, 0.95))
    res.append(dict(sektor=sec, gamma=gam, se=se, theta=theta, theta_lo=float(np.clip(1 - gam - 1.645 * se, 0.05, 0.95)),
                    theta_hi=float(np.clip(1 - gam + 1.645 * se, 0.05, 0.95)), sure_ceyrek=1 / (1 - theta),
                    kaynak=SRC.get(sec, "Yİ-ÜFE")))
R = pd.DataFrame(res).set_index("sektor")
R.to_csv(f"{MODEL_DIR}/data/io/calvo_sector.csv", float_format="%.3f")
if __name__ == "__main__":
    print(R.round(3).to_string())
    cw = np.array(io["c_dom"]); print("Tüketim ağırlıklı θ:", round(float((cw * R.theta.values).sum()), 3),
                                      " KD ağırlıklı θ:", round(float((np.array(io["va_w"]) * R.theta.values).sum()), 3))
