"""
Çok sektörlü, girdi-çıktı ağlı küçük açık ekonomi Yeni Keynesyen modeli (32 sektör)
Bouakez-Cardia-Ruge-Murcia (2014) / Pasten-Schoenle-Weber (2020) çizgisinde, Türkiye'ye uyarlanmış:
 - Sektörel Calvo (tahmin edilen θ_k) + ortak endeksleme
 - Sektör k nominal marjinal maliyeti: yurt içi girdiler (TÜİK 2023 G-Ç, w_d), ithal ara girdi (kur),
   ithal enerji (kur + Brent), katma değer (ücret); hepsi yerli fiyat düzeyine göre göreli
 - Makro blok: tahmin edilen genişletilmiş DSGE (Euler, ücret katılığı, ithal fiyat NKPC, UIP+risk primi,
   NFA, Taylor kuralı) — 2006–2019 posterior ortalamaları
 - Yerli enflasyon π_H = Σ ĉ_k π_k (ĉ: G-Ç hanehalkı yurt içi tüketim payları) -> makroya tam geri besleme
 - Sektörel üretim: G-Ç talep sistemi (ara talep + tüketim + ihracat + diğer nihai talep), göreli fiyat esnekliği ε
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, json, pandas as pd, scipy.linalg as la

io = json.load(open(f"{MODEL_DIR}/data/io/io32.json"))
SECT = io["SECTORS"]; K = len(SECT)
W_D, W_M, W_ME, W_L = (np.array(io[k]) for k in ("w_d", "w_m", "w_me", "w_l"))
D_INT, S_C, S_X, S_O = (np.array(io[k]) for k in ("d_int", "s_c", "s_x", "s_o"))
C_DOM = np.array(io["c_dom"]); ALPHA_C = io["alpha_c"]; VA_W = np.array(io["va_w"])
# Enerji ağırlığı düzeltmesi: temel fiyatlı G-Ç tüketim payları (vergi/marj hariç) enerjinin TÜFE'deki ağırlığını
# düşük gösteriyor. D35 payı x2, yurt içi rafine ürün payı x10 -> doğrudan enerji ~%7 (+ ithal enerji ~%1).
# Bu kalibrasyonla Brent -> TÜFE geçişi tahmin edilen toplulaştırılmış modele yaklaşır.
_C_IO = C_DOM.copy()
C_DOM = C_DOM.copy(); C_DOM[SECT.index("D35")] *= 2.0; C_DOM[SECT.index("C19")] *= 10.0; C_DOM /= C_DOM.sum()
CALVO = pd.read_csv(f"{MODEL_DIR}/data/io/calvo_sector.csv", index_col=0).loc[SECT, "theta"].values

AGG = ["y", "c", "piH", "piF", "pi", "i", "q", "s", "psi", "de", "w", "nx", "nfa", "rr",
       "a", "d", "prem", "oil", "ystar", "rstar", "m", "cp"]
VARS = AGG + [f"pi_{k}" for k in SECT] + [f"r_{k}" for k in SECT] + [f"y_{k}" for k in SECT] + [f"u_{k}" for k in SECT]
SH_AGG = ["e_a", "e_d", "e_prem", "e_oil", "e_ystar", "e_rstar", "e_m", "e_cp"]
SHOCKS = SH_AGG + [f"e_u_{k}" for k in SECT]
IX = {v: i for i, v in enumerate(VARS)}; SX = {s: i for i, s in enumerate(SHOCKS)}
N = len(VARS)

def base_params():
    t = pd.read_csv(f"{MODEL_DIR}/s1_posterior_table.csv", index_col=0).post_mean.to_dict()
    p = dict(beta=0.985, phi=2.0, alpha=0.30, chi=0.01, eps=2.0, rho_u=0.5, sd_u=1.0)
    p.update(t)
    return p

def build(p, theta=None, network=True):
    theta = CALVO if theta is None else np.asarray(theta)
    A0, A1, A2 = (np.zeros((N, N)) for _ in range(3)); B = np.zeros((N, len(SHOCKS))); row = [0]
    def eq(terms, shocks=None):
        r = row[0]
        for (v, lag), c in terms.items():
            {1: A0, 0: A1, -1: A2}[lag][r, IX[v]] += c
        for s_, c in (shocks or {}).items(): B[r, SX[s_]] += c
        row[0] += 1
    b, sg, h, ph, al, et = (p[k] for k in ("beta", "sigma", "h", "phi", "alpha", "eta"))
    ac = ALPHA_C
    lamF = (1 - p["thF"]) * (1 - b * p["thF"]) / p["thF"]
    # --- makro blok
    eq({("c", 0): 1 + h, ("c", -1): -h, ("c", 1): -1, ("i", 0): (1 - h) / sg, ("prem", 0): p["zeta"] * (1 - h) / sg,
        ("pi", 1): -(1 - h) / sg, ("d", 0): -1})
    rw = p["rho_w"]
    eq({("w", 0): 1, ("w", -1): -rw, ("y", 0): -(1 - rw) * ph, ("a", 0): (1 - rw) * ph,
        ("c", 0): -(1 - rw) * sg / (1 - h), ("c", -1): (1 - rw) * sg * h / (1 - h)})
    eq({("piF", 0): 1 + b * p["dF"], ("piF", -1): -p["dF"], ("piF", 1): -b, ("psi", 0): -lamF, ("oil", 0): -lamF * p["om_e"]})
    eq({("pi", 0): 1, ("piH", 0): -(1 - ac), ("piF", 0): -ac})
    eq({("s", 0): 1, ("s", -1): -1, ("piF", 0): -1, ("piH", 0): 1})
    eq({("q", 0): 1, ("psi", 0): -1, ("s", 0): -(1 - ac)})
    eq({("psi", 0): 1, ("psi", -1): -1, ("de", 0): -1, ("piF", 0): 1})
    eq({("q", 0): 1, ("q", 1): -1, ("i", 0): 1, ("pi", 1): -1, ("rstar", 0): -1, ("prem", 0): -1, ("nfa", 0): p["chi"]})
    eq({("y", 0): 1 - al, ("c", 0): -(1 - al), ("s", 0): -al * et * (2 - al), ("psi", 0): -al * et, ("ystar", 0): -al})
    eq({("i", 0): 1, ("i", -1): -p["rho_i"], ("pi", 0): -(1 - p["rho_i"]) * p["phi_pi"],
        ("y", 0): -(1 - p["rho_i"]) * p["phi_y"], ("de", 0): -(1 - p["rho_i"]) * p["phi_e"], ("m", 0): -1})
    eq({("nx", 0): 1, ("y", 0): -1, ("c", 0): 1, ("s", 0): al, ("psi", 0): al, ("oil", 0): al * p["om_e"]})
    eq({("nfa", 0): 1, ("nfa", -1): -1 / b, ("nx", 0): -1})
    eq({("rr", 0): 1, ("i", 0): -1, ("pi", 1): 1})
    for v, s_ in zip(["a", "d", "prem", "oil", "ystar", "rstar", "m", "cp"], SH_AGG):
        eq({(v, 0): 1, (v, -1): -p["rho_" + v]}, {s_: -1})
    # --- yerli fiyat düzeyi normalizasyonu: Σ ĉ_k r_k = 0  (π_H'yi belirler)
    eq({(f"r_{k}", 0): C_DOM[j] for j, k in enumerate(SECT)})
    # --- sektör blokları
    dl = p["dH"]
    for j, k in enumerate(SECT):
        th = theta[j]; lam = (1 - th) * (1 - b * th) / th
        # u_k: sektör k'nın marjinal maliyetinde %'lik şok (λ_k ile geçer); cp: ekonomi geneli enflasyon (maliyet) şoku
        t = {(f"pi_{k}", 0): 1 + b * dl, (f"pi_{k}", -1): -dl, (f"pi_{k}", 1): -b, ("cp", 0): -1, (f"u_{k}", 0): -lam}
        # reel marjinal maliyet - göreli fiyat
        if network:
            for i2, k2 in enumerate(SECT):
                t[(f"r_{k2}", 0)] = t.get((f"r_{k2}", 0), 0) - lam * W_D[i2, j]
        t[(f"r_{k}", 0)] = t.get((f"r_{k}", 0), 0) + lam
        t[("psi", 0)] = -lam * (W_M[j] + W_ME[j]); t[("s", 0)] = -lam * (W_M[j] + W_ME[j] + W_L[j] * ac)
        t[("oil", 0)] = -lam * W_ME[j]; t[("w", 0)] = -lam * W_L[j]; t[("a", 0)] = lam
        eq(t)
        eq({(f"pi_{k}", 0): 1, ("piH", 0): -1, (f"r_{k}", 0): -1, (f"r_{k}", -1): 1})
        ty = {(f"y_{k}", 0): 1, ("c", 0): -S_C[j], ("s", 0): -S_C[j] * ac * et - S_X[j] * et, ("psi", 0): -S_X[j] * et,
              ("ystar", 0): -S_X[j], ("y", 0): -S_O[j], (f"r_{k}", 0): p["eps"] * (S_C[j] + S_X[j])}
        for i2, k2 in enumerate(SECT):
            ty[(f"y_{k2}", 0)] = ty.get((f"y_{k2}", 0), 0) - D_INT[j, i2]
        eq(ty)
        eq({(f"u_{k}", 0): 1, (f"u_{k}", -1): -p["rho_u"]}, {f"e_u_{k}": -1})
    assert row[0] == N, (row[0], N)
    return A0, A1, A2, B

def solve(p, theta=None, network=True, tol=1e-11, maxit=20000):
    A0, A1, A2, B = build(p, theta, network)
    P = np.zeros((N, N))
    for it in range(maxit):
        Pn = -np.linalg.solve(A0 @ P + A1, A2)
        if np.max(np.abs(Pn - P)) < tol: P = Pn; break
        P = Pn
    else: raise RuntimeError("yakınsamadı")
    Q = -np.linalg.solve(A0 @ P + A1, B)
    resid = np.max(np.abs(A0 @ P @ P + A1 @ P + A2))
    I, Z = np.eye(N), np.zeros((N, N))
    lam = la.eig(np.block([[Z, I], [-A2, -A1]]), np.block([[I, Z], [Z, A0]]), right=False)
    n_st = int(np.sum(np.abs(lam[np.isfinite(lam)]) < 1 - 1e-9))
    return dict(P=P, Q=Q, resid=resid, n_stable=n_st, maxeig=float(np.max(np.abs(np.linalg.eigvals(P)))))

def irf(sol, p, shock, H=16, norm=None):
    P, Q = sol["P"], sol["Q"]
    e = np.zeros(len(SHOCKS))
    e[SX[shock]] = p.get("sd_" + shock[2:], p["sd_u"]) if shock in SH_AGG else p["sd_u"]
    z = np.zeros((H, N)); z[0] = Q @ e
    for t in range(1, H): z[t] = P @ z[t - 1]
    if norm == "i1": z *= 0.25 / z[0, IX["i"]]
    if norm == "oil10": z *= 10 / z[0, IX["oil"]]
    if norm == "q1": z *= 1 / z[0, IX["q"]]
    return z

def summarize(z):
    """Sektörel fiyat düzeyi (kümülatif sektör enflasyonu, %), sektör üretimi (%), makro"""
    pi_k = z[:, [IX[f"pi_{k}"] for k in SECT]]
    return dict(P_level=np.cumsum(pi_k, 0), pi_ann=4 * pi_k, y_k=z[:, [IX[f"y_{k}"] for k in SECT]],
                cpi_level=np.cumsum(z[:, IX["pi"]]), pi=4 * z[:, IX["pi"]], piH=4 * z[:, IX["piH"]], y=z[:, IX["y"]],
                i=4 * z[:, IX["i"]], q=z[:, IX["q"]], ygap_va=z[:, [IX[f"y_{k}"] for k in SECT]] @ VA_W)

if __name__ == "__main__":
    p = base_params(); sol = solve(p)
    print("N =", N, "kararlı kök =", sol["n_stable"], "artık =", sol["resid"], "maks özdeğer =", round(sol["maxeig"], 4))
    z = summarize(irf(sol, p, "e_m", norm="i1"))
    print("Faiz +1: TÜFE yıllık h0..3", z["pi"][:4].round(2), "GSYH", z["y"][:4].round(2), "KD-ağırlıklı sektör üretimi", z["ygap_va"][:4].round(2))
    print("Sektör fiyat düzeyi h4:", dict(zip(SECT, z["P_level"][4].round(2))))
