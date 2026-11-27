"""
Türkiye için küçük açık ekonomi Yeni Keynesyen DSGE modeli
(v3: v2'ye yatırım, sermaye birikimi ve finansal hızlandırıcı eklendi)
(Galí-Monacelli 2005 / Monacelli 2005 / Justiniano-Preston 2010 çizgisinde)

Özellikler:
  - Tüketimde alışkanlık (habit)
  - Yerli ve ithal mal fiyatlarında Calvo yapışkanlığı + endeksleme
  - Eksik kur geçişkenliği (tek fiyat yasası sapması, psi)
  - Eksik finansal piyasalar: borç-elastik risk primli UIP + NFA birikimi
  - Enerji (ithal girdi) fiyat şoku
  - Faiz düzleştirmeli, kura tepki verebilen Taylor kuralı
  - v3: sermaye birikimi, yatırım düzenleme maliyeti (Christiano-Eichenbaum-Evans 2005, Smets-Wouters 2007),
        Tobin q, yatırım malının yüksek ithal içeriği, Bernanke-Gertler-Gilchrist (1999) tipi finansal
        hızlandırıcı (girişimci net servet + kaldıraca bağlı dış finansman primi; Gertler-Gilchrist-Natalucci 2007)
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
        "inv", "k", "l", "rk", "qk", "rke", "n", "spr",
        "a", "d", "prem", "oil", "ystar", "rstar", "m", "cp", "mu"]
SHOCKS = ["e_a", "e_d", "e_prem", "e_oil", "e_ystar", "e_rstar", "e_m", "e_cp", "e_mu", "e_nw"]
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
    chi=0.01,     # borç-elastik risk primi (UIP)
    chi_c=0.0,    # net dış varlığın hanehalkı kredi faizine etkisi (0 = kapalı)
    om_q=1.0,     # UIP'de ileriye dönük kur beklentisi ağırlığı (1 = tam rasyonel; <1 hibrit UIP)
    rho_i=0.75, phi_pi=1.8, phi_e=0.10,  # para politikası kuralı
    phi_y=0.0,    # kuralda çıktı düzeyine tepki: kapalı (bkz. not)
    phi_dy=0.25,  # kuralda çıktı büyümesine (y - y(-1)) tepki (tahmin edilir)
    # Not: Çıktı düzeyine tepki, net dış varlıkların çok yavaş eridiği şoklarda (dış talep, Brent) GSYH'nin
    # kalıcı denge düzeyinden uzun süre sapmasını çıktı açığı sayıp enflasyonu on yıllarca sıfırdan uzak
    # tutuyordu. Büyümeye tepki veren kural veriye aynı ölçüde uyar ve bu sorunu ortadan kaldırır.
    rho_a=0.70, sd_a=0.50,   # verimlilik şoku (tahmin edilir; saat başına sanayi verimliliği gözlenir)
    rho_d=0.70, rho_prem=0.80, rho_oil=0.70,
    rho_ystar=0.80, rho_rstar=0.85, rho_m=0.50, rho_cp=0.30,
    # --- v3: yatırım ve finansal hızlandırıcı
    kap=0.40,     # katma değerde sermaye payı
    delk=0.025,   # çeyreklik amortisman (yıllık ~%10)
    aI=0.50,      # yatırım malının ithal içeriği (G-Ç: yatırım/tüketim toplam ithal içerik oranı 1,68 x alpha)
    Sk=4.0,       # yatırım düzenleme maliyeti S''(.) (tahmin edilir)
    nu=0.05,      # dış finansman priminin kaldıraca esnekliği (tahmin edilir)
    gam_e=0.97,   # girişimci hayatta kalma oranı
    KN=2.0,       # durağan durum kaldıracı K/N
    rho_mu=0.70,  # yatırım etkinliği şoku kalıcılığı
    phi_fx=0.40,  # firma borcunun döviz cinsi payı (tahmin edilir)
)

def shares(p):
    """Durağan durum payları: rk = 1/beta - 1 + delta,  I/Y = delta*K/Y = delta*kap/rk"""
    rk = 1 / p["beta"] - 1 + p["delk"]
    iy = p["delk"] * p["kap"] / rk
    return iy, 1 - iy, 1 - p["beta"] * (1 - p["delk"])   # iy, cy, varpi (getiride kira payı)


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
        ("pi", 1): -(1 - h) / sg, ("d", 0): -1, ("nfa", 0): -(1 - h) / sg * p.get("chi_c", 0.0)})
    # 2a Reel ücret (TÜFE cinsinden): w = rho_w*w(-1) + (1-rho_w)*MRS,
    #    MRS = phi*(y-a) + sigma/(1-h)*(c - h c(-1))
    rw = p["rho_w"]
    eq({("w", 0): 1, ("w", -1): -rw, ("l", 0): -(1 - rw) * ph,
        ("c", 0): -(1 - rw) * sg / (1 - h), ("c", -1): (1 - rw) * sg * h / (1 - h)})
    # 2b Reel marjinal maliyet: emek (yerli mal cinsinden ücret - verimlilik)
    #    + ithal ara girdi (s+psi) + enerji
    #    katma değer maliyeti = (1-kap)*(yerli mal cinsinden ücret) + kap*(sermaye kirası) - verimlilik
    gm, kp = p["gam_m"], p["kap"]
    eq({("mc", 0): 1, ("w", 0): -(1 - gm) * (1 - kp), ("rk", 0): -(1 - gm) * kp, ("s", 0): -((1 - gm) * (1 - kp) * al + gm),
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
    #    hibrit UIP: E_t q_{t+1} yerine om_q*E_t q_{t+1} + (1-om_q)*q_{t-1}
    omq = p.get("om_q", 1.0)
    eq({("q", 0): 1, ("q", 1): -omq, ("q", -1): -(1 - omq), ("i", 0): 1, ("pi", 1): -1,
        ("rstar", 0): -1, ("prem", 0): -1, ("nfa", 0): p["chi"]})
    # 11 Mal piyasası dengesi
    #    yurt içi talep = cy*c + iy*inv
    iy, cy, vpi = shares(p)
    eq({("y", 0): 1 - al, ("c", 0): -(1 - al) * cy, ("inv", 0): -(1 - al) * iy, ("s", 0): -al * et * (2 - al),
        ("psi", 0): -al * et, ("ystar", 0): -al})
    # 12 Taylor kuralı
    eq({("i", 0): 1, ("i", -1): -p["rho_i"], ("pi", 0): -(1 - p["rho_i"]) * p["phi_pi"],
        ("y", 0): -(1 - p["rho_i"]) * (p["phi_y"] + p.get("phi_dy", 0.0)), ("y", -1): (1 - p["rho_i"]) * p.get("phi_dy", 0.0),
        ("de", 0): -(1 - p["rho_i"]) * p["phi_e"], ("m", 0): -1})
    # 13 Net ihracat (GSYH payı)
    eq({("nx", 0): 1, ("y", 0): -1, ("c", 0): cy, ("inv", 0): iy, ("s", 0): al, ("psi", 0): al,
        ("oil", 0): al * p["om_e"]})   # enerji ithalat faturası
    # 14 Net dış varlık birikimi
    eq({("nfa", 0): 1, ("nfa", -1): -1 / b, ("nx", 0): -1})
    # 15 Ex-ante reel faiz
    eq({("rr", 0): 1, ("i", 0): -1, ("pi", 1): 1})
    # ---------------- v3: yatırım, sermaye ve finansal hızlandırıcı
    dk, Sk, z = p["delk"], p["Sk"], p["zeta"]
    # üretim (katma değer): y = a + kap*k(-1) + (1-kap)*l
    eq({("y", 0): 1, ("a", 0): -1, ("k", -1): -kp, ("l", 0): -(1 - kp)})
    # faktör talebi (maliyet minimizasyonu): rk = w + alpha*s + l - k(-1)   (yerli mal cinsinden kira)
    eq({("rk", 0): 1, ("w", 0): -1, ("s", 0): -al, ("l", 0): -1, ("k", -1): 1})
    # sermaye birikimi (mu: yatırım etkinliği şoku)
    eq({("k", 0): 1, ("k", -1): -(1 - dk), ("inv", 0): -dk, ("mu", 0): -dk * (1 + b) * Sk})
    # yatırım Euler denklemi (Tobin q; yatırım malının TÜFE'ye göre göreli fiyatı = (aI - alpha)*s)
    eq({("inv", 0): 1, ("inv", -1): -1 / (1 + b), ("inv", 1): -b / (1 + b),
        ("qk", 0): -1 / ((1 + b) * Sk), ("s", 0): (p["aI"] - al) / ((1 + b) * Sk), ("mu", 0): -1})
    # sermayenin gerçekleşen reel getirisi (TÜFE cinsinden): rke = varpi*(rk - alpha*s) + (1-varpi)*qk - qk(-1)
    eq({("rke", 0): 1, ("rk", 0): -vpi, ("s", 0): vpi * al, ("qk", 0): -(1 - vpi), ("qk", -1): 1})
    # beklenen getiri = reel borçlanma faizi (politika faizi + ülke primi geçişi) + dış finansman primi
    eq({("rke", 1): 1, ("i", 0): -1, ("pi", 1): 1, ("prem", 0): -z, ("spr", 0): -1})
    # dış finansman primi kaldıraca bağlı: spr = nu*(qk + k - n)
    eq({("spr", 0): 1, ("qk", 0): -p["nu"], ("k", 0): -p["nu"], ("n", 0): p["nu"]})
    ge, KN = p["gam_e"], p["KN"]
    # girişimci net serveti (e_nw: net servet şoku). Borcun phi_fx kadarı döviz cinsidir: döviz borcunun gerçekleşen reel
    # maliyeti dış faiz + ülke primi + gerçekleşen reel değer kaybıdır (Céspedes-Chang-Velasco 2004); UIP nedeniyle ex ante
    # TL borçla aynı, ex post kur sürprizi kadar farklıdır -> TL değer kaybı net serveti düşürür.
    fx = p.get("phi_fx", 0.0)
    eq({("n", 0): 1, ("rke", 0): -ge * KN, ("i", -1): ge * (KN - 1) * (1 - fx), ("pi", 0): -ge * (KN - 1) * (1 - fx),
        ("rstar", -1): ge * (KN - 1) * fx, ("prem", -1): ge * (KN - 1) * (z + fx), ("nfa", -1): -ge * (KN - 1) * fx * p["chi"],
        ("q", 0): ge * (KN - 1) * fx, ("q", -1): -ge * (KN - 1) * fx,
        ("spr", -1): ge * (KN - 1), ("n", -1): -ge}, {"e_nw": -1})
    # 16-24 Dışsal AR(1) süreçler
    for v, s_ in zip(["a", "d", "prem", "oil", "ystar", "rstar", "m", "cp", "mu"], SHOCKS):
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
