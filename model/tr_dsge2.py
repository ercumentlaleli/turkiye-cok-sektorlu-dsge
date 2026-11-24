"""
Türkiye için küçük açık ekonomi Yeni Keynesyen DSGE modeli (v2: yerli maliyet şoku eklendi)
(Galí-Monacelli 2005 / Monacelli 2005 / Justiniano-Preston 2010 çizgisinde)

Özellikler:
  - Tüketimde alışkanlık (habit)
  - Yerli ve ithal mal fiyatlarında Calvo yapışkanlığı + endeksleme
  - Eksik kur geçişkenliği (tek fiyat yasası sapması, psi)
  - Eksik finansal piyasalar: borç-elastik risk primli UIP + NFA birikimi
  - Enerji (ithal girdi) fiyat şoku
  - Faiz düzleştirmeli, kura tepki verebilen Taylor kuralı
Çeyreklik frekans, log-doğrusal, durağan durumdan sapmalar.

Çözüm: A0 E[z_{t+1}] + A1 z_t + A2 z_{t-1} + B e_t = 0
        z_t = P z_{t-1} + Q e_t   (Rendahl 2017 zaman iterasyonu)
Belirlilik: eşlik (companion) kalemindeki genelleştirilmiş özdeğer sayımı.
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np
import scipy.linalg as la

VARS = ["y", "c", "piH", "piF", "pi", "i", "q", "s", "psi", "de", "mc", "w",
        "nx", "nfa", "rr",
        "a", "d", "prem", "oil", "ystar", "rstar", "m", "cp"]
SHOCKS = ["e_a", "e_d", "e_prem", "e_oil", "e_ystar", "e_rstar", "e_m", "e_cp"]
IX = {v: k for k, v in enumerate(VARS)}
SX = {v: k for k, v in enumerate(SHOCKS)}

# ---------------- Kalibrasyon (çeyreklik) ----------------
BASE = dict(
    beta=0.985,   # yüksek reel faiz ortamı -> düşük iskonto
    sigma=2.0,    # ters zamanlar arası ikame esnekliği
    h=0.70,       # alışkanlık
    phi=2.0,      # ters Frisch esnekliği
    alpha=0.30,   # dışa açıklık (ithalat/GSYH ~ %30)
    eta=0.5,      # yerli-yabancı mal ikame esnekliği (düşük)
    thH=0.75,     # yerli fiyat Calvo
    thF=0.45,     # ithal fiyat Calvo -> hızlı geçişkenlik
    dH=0.80,      # yerli enflasyon endeksleme (atalet)
    dF=0.70,      # ithal enflasyon endeksleme
    kap_oil=0.05, # enerjinin yerli marjinal maliyete doğrudan etkisi
    rho_w=0.90,   # reel ücret katılığı (Blanchard-Galí)
    gam_m=0.25,   # üretimde ithal ara malı payı (ithal girdi bağımlılığı)
    om_e=0.20,    # enerjinin ithalat içindeki payı
    zeta=1.5,     # risk priminin yurt içi kredi faizine geçişi (finansal kanal)
    chi=0.01,     # borç-elastik risk primi
    rho_i=0.75, phi_pi=1.8, phi_y=0.30, phi_e=0.10,  # para politikası kuralı
    rho_a=0.90, rho_d=0.70, rho_prem=0.80, rho_oil=0.70,
    rho_ystar=0.80, rho_rstar=0.85, rho_m=0.50, rho_cp=0.30,
)


def build(p):
    n, k = len(VARS), len(SHOCKS)
    A0, A1, A2 = (np.zeros((n, n)) for _ in range(3))
    B = np.zeros((n, k))
    row = [0]

    def eq(terms, shocks=None):
        r = row[0]
        for (v, lag), c in terms.items():
            {1: A0, 0: A1, -1: A2}[lag][r, IX[v]] += c
        for s_, c in (shocks or {}).items():
            B[r, SX[s_]] += c
        row[0] += 1

    b, sg, h, ph, al, et = (p[x] for x in ["beta", "sigma", "h", "phi", "alpha", "eta"])
    lamH = (1 - p["thH"]) * (1 - b * p["thH"]) / p["thH"]
    lamF = (1 - p["thF"]) * (1 - b * p["thF"]) / p["thF"]

    # 1 Euler (habit + talep şoku)
    #   Hanehalkının karşılaştığı faiz = politika faizi + zeta * ülke risk primi
    eq({("c", 0): 1 + h, ("c", -1): -h, ("c", 1): -1,
        ("i", 0): (1 - h) / sg, ("prem", 0): p["zeta"] * (1 - h) / sg,
        ("pi", 1): -(1 - h) / sg, ("d", 0): -1})
    # 2a Reel ücret (TÜFE cinsinden): w = rho_w*w(-1) + (1-rho_w)*MRS,
    #    MRS = phi*(y-a) + sigma/(1-h)*(c - h c(-1))
    rw = p["rho_w"]
    eq({("w", 0): 1, ("w", -1): -rw, ("y", 0): -(1 - rw) * ph, ("a", 0): (1 - rw) * ph,
        ("c", 0): -(1 - rw) * sg / (1 - h), ("c", -1): (1 - rw) * sg * h / (1 - h)})
    # 2b Reel marjinal maliyet: emek (yerli mal cinsinden ücret - verimlilik)
    #    + ithal ara girdi (s+psi) + enerji
    gm = p["gam_m"]
    eq({("mc", 0): 1, ("w", 0): -(1 - gm), ("s", 0): -((1 - gm) * al + gm),
        ("a", 0): (1 - gm), ("psi", 0): -gm, ("oil", 0): -p["kap_oil"]})
    # 3 Yerli NKPC (endeksli)
    eq({("piH", 0): 1 + b * p["dH"], ("piH", -1): -p["dH"], ("piH", 1): -b, ("mc", 0): -lamH,
        ("cp", 0): -1})   # yerli maliyet (cost-push) şoku
    # 4 İthal fiyat NKPC (eksik geçişkenlik; enerji maliyeti)
    eq({("piF", 0): 1 + b * p["dF"], ("piF", -1): -p["dF"], ("piF", 1): -b,
        ("psi", 0): -lamF, ("oil", 0): -lamF * p["om_e"]})
    # 5 TÜFE
    eq({("pi", 0): 1, ("piH", 0): -(1 - al), ("piF", 0): -al})
    # 6 Dış ticaret haddi
    eq({("s", 0): 1, ("s", -1): -1, ("piF", 0): -1, ("piH", 0): 1})
    # 7 Reel kur tanımı
    eq({("q", 0): 1, ("psi", 0): -1, ("s", 0): -(1 - al)})
    # 8 Tek fiyat yasası sapması
    eq({("psi", 0): 1, ("psi", -1): -1, ("de", 0): -1, ("piF", 0): 1})
    # (Nominal kur: de = dq + pi, 6-8 numaralı denklemlerden türetilir; ayrıca yazılmaz)
    # 10 UIP (reel, risk primli, borç-elastik)
    eq({("q", 0): 1, ("q", 1): -1, ("i", 0): 1, ("pi", 1): -1,
        ("rstar", 0): -1, ("prem", 0): -1, ("nfa", 0): p["chi"]})
    # 11 Mal piyasası dengesi
    eq({("y", 0): 1 - al, ("c", 0): -(1 - al), ("s", 0): -al * et * (2 - al),
        ("psi", 0): -al * et, ("ystar", 0): -al})
    # 12 Taylor kuralı
    eq({("i", 0): 1, ("i", -1): -p["rho_i"], ("pi", 0): -(1 - p["rho_i"]) * p["phi_pi"],
        ("y", 0): -(1 - p["rho_i"]) * p["phi_y"], ("de", 0): -(1 - p["rho_i"]) * p["phi_e"],
        ("m", 0): -1})
    # 13 Net ihracat (GSYH payı)
    eq({("nx", 0): 1, ("y", 0): -1, ("c", 0): 1, ("s", 0): al, ("psi", 0): al,
        ("oil", 0): al * p["om_e"]})   # enerji ithalat faturası
    # 14 Net dış varlık birikimi
    eq({("nfa", 0): 1, ("nfa", -1): -1 / b, ("nx", 0): -1})
    # 15 Ex-ante reel faiz
    eq({("rr", 0): 1, ("i", 0): -1, ("pi", 1): 1})
    # 16-22 Dışsal AR(1) süreçler
    for v, s_ in zip(["a", "d", "prem", "oil", "ystar", "rstar", "m", "cp"], SHOCKS):
        eq({(v, 0): 1, (v, -1): -p["rho_" + v]}, {s_: -1})
    assert row[0] == n
    return A0, A1, A2, B


def solve(p, tol=1e-12, maxit=20000):
    A0, A1, A2, B = build(p)
    n = A0.shape[0]
    P = np.zeros((n, n))
    for it in range(maxit):
        Pn = -np.linalg.solve(A0 @ P + A1, A2)
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    else:
        raise RuntimeError("Zaman iterasyonu yakınsamadı")
    Q = -np.linalg.solve(A0 @ P + A1, B)
    resid = np.max(np.abs(A0 @ P @ P + A1 @ P + A2))
    # Belirlilik: companion kalemi [[0,I],[-A2,-A1]] z = lam [[I,0],[0,A0]] z
    I, Z = np.eye(n), np.zeros((n, n))
    F = np.block([[Z, I], [-A2, -A1]])
    G = np.block([[I, Z], [Z, A0]])
    lam = la.eig(F, G, right=False)
    mod = np.abs(lam[np.isfinite(lam)])
    n_stable = int(np.sum(mod < 1 - 1e-9))
    n_unit = int(np.sum(np.abs(mod - 1) < 1e-9))
    return dict(P=P, Q=Q, resid=resid, n_stable=n_stable, n_unit=n_unit,
                max_eig_P=float(np.max(np.abs(np.linalg.eigvals(P)))), iters=it)


def irf(sol, shock, size, T=20):
    P, Q = sol["P"], sol["Q"]
    z = np.zeros((T, P.shape[0]))
    e = np.zeros(len(SHOCKS)); e[SX[shock]] = size
    z[0] = Q @ e
    for t in range(1, T):
        z[t] = P @ z[t - 1]
    return z
