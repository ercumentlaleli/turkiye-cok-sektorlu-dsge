"""
LP-IV sağlaması (Jordà 2005; Stock-Watson 2018; Ramey 2016)
 (a) Para politikası: Δi_t (yıllık puan), araç = Romer-Romer tipi politika şoku
     (Δi_t'nin cari ve gecikmeli enflasyon, büyüme, reel kur, Brent ve faiz düzeyine regresyon artığı)
 (b) Petrol: Δlog Brent_t (Türkiye için dışsal) — doğrudan LP, %10'luk artışa normalize
 y_{s,t+h} - y_{s,t-1} = a + β_h x_t + Σ_{l=1..2} kontroller + e
 Kontroller: sektörün kendi büyümesi, dy, pi, dq, Δi, doil (2 gecikme)
 HAC (Newey-West, h+1 gecikme). Ayrıca sektör sabit etkili havuzlanmış panel LP (Driscoll-Kraay).
"""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent  # model/ klasörü
import numpy as np, pandas as pd

SECT = ["B"] + [f"C{k}" for k in range(10, 33)] + ["D35"]
H = 8

def load(start="2006Q1", end="2019Q4"):
    d = pd.read_csv(f"{MODEL_DIR}/data/evds_raw.csv", index_col=0)
    f = pd.read_csv(f"{MODEL_DIR}/data/foreign.csv", index_col=0)
    s = pd.read_csv(f"{MODEL_DIR}/data/sectors.csv", index_col=0)
    rate = d["aofm"].fillna(d["polrate_bis"])
    m = pd.DataFrame({"dy": 100*np.log(d.gdp).diff(), "pi": 100*np.log(d.cpi_sa).diff(),
                      "dq": -100*np.log(d.reer).diff(), "i": rate, "di": rate.diff()}, index=d.index)
    m["doil"] = 100*np.log(f.brent).diff()
    lv = 100*np.log(s)
    return m, lv

def rr_shock(m):
    X = pd.concat([m[["pi", "dy", "dq", "doil"]], m[["pi", "dy", "dq", "doil", "di"]].shift(1).add_suffix("_l1"),
                   m[["i"]].shift(1).add_suffix("_l1")], axis=1)
    df = pd.concat([m["di"], X], axis=1).dropna()
    df = df.loc["2006Q1":"2019Q4"]
    Xm = np.column_stack([np.ones(len(df)), df.drop(columns="di").values])
    b = np.linalg.lstsq(Xm, df["di"].values, rcond=None)[0]
    return pd.Series(df["di"].values - Xm @ b, index=df.index, name="rr")

def nw_se(X, e, lags):
    T, k = X.shape; XtXi = np.linalg.inv(X.T @ X); u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1); G = u[l:].T @ u[:-l]; S += w * (G + G.T)
    return np.sqrt(np.diag(XtXi @ S @ XtXi))

def lp_one(yl, m, x_name, inst=None, h=0, own_g=None):
    """yl: log seviye (100x) serisi; x: politika değişkeni; inst: araç serisi (None ise OLS)"""
    lhs = yl.shift(-h) - yl.shift(1)
    C = pd.concat([own_g.shift(1), own_g.shift(2)] +
                  [m[c].shift(l) for c in ["dy", "pi", "dq", "di", "doil"] for l in (1, 2)], axis=1)
    cols = [lhs.rename("y"), m[x_name].rename("x")] + ([inst.rename("z")] if inst is not None else []) + [C]
    df = pd.concat(cols, axis=1).loc["2006Q1":"2019Q4"].dropna()
    W = np.column_stack([np.ones(len(df)), df.drop(columns=["y", "x"] + (["z"] if inst is not None else [])).values])
    def resid(v): return v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
    y, x = resid(df["y"].values), resid(df["x"].values)
    if inst is None:
        b = np.dot(x, y) / np.dot(x, x); e = y - b * x
        se = nw_se(x[:, None], e, h + 1)[0]; F = np.nan
    else:
        z = resid(df["z"].values)
        pi1 = np.dot(z, x) / np.dot(z, z); xh = pi1 * z
        b = np.dot(xh, y) / np.dot(xh, x); e = y - b * x
        se = nw_se(xh[:, None], e, h + 1)[0]
        ex = x - pi1 * z; F = (pi1 / nw_se(z[:, None], ex, 1)[0]) ** 2
    return b, se, F, len(df)

def run():
    m, lv = load()
    rr = rr_shock(m)
    out = {"mp": {"PP": np.zeros((len(SECT), H + 1, 2)), "IP": np.zeros((len(SECT), H + 1, 2))},
           "oil": {"PP": np.zeros((len(SECT), H + 1, 2)), "IP": np.zeros((len(SECT), H + 1, 2))}}
    Fs = []
    for k, s in enumerate(SECT):
        for kind in ("PP", "IP"):
            yl = lv[f"{kind}_{s}"]; g = yl.diff()
            for h in range(H + 1):
                b, se, F, n = lp_one(yl, m, "di", inst=rr, h=h, own_g=g)
                out["mp"][kind][k, h] = (b, se); Fs.append(F)
                b2, se2, _, _ = lp_one(yl, m, "doil", inst=None, h=h, own_g=g)
                out["oil"][kind][k, h] = (10 * b2, 10 * se2)      # %10 Brent artışı
    # havuzlanmış panel (sektör ortalaması): sektör tepkilerinin ortalaması + sektörler arası kovaryansı dikkate alan SE
    pooled = {}
    for sh in out:
        for kind in ("PP", "IP"):
            pooled[(sh, kind)] = pooled_lp(lv, m, rr, sh, kind)
    return out, pooled, rr, np.nanmedian(Fs), m

def pooled_lp(lv, m, rr, sh, kind):
    """Sektör sabit etkili panel LP; Driscoll-Kraay SE (zaman kümelemeli + NW)."""
    res = np.zeros((H + 1, 2))
    for h in range(H + 1):
        ys, xs, zs, ts = [], [], [], []
        for s in SECT:
            yl = lv[f"{kind}_{s}"]; g = yl.diff()
            lhs = yl.shift(-h) - yl.shift(1)
            C = pd.concat([g.shift(1), g.shift(2)] + [m[c].shift(l) for c in ["dy", "pi", "dq", "di", "doil"] for l in (1, 2)], axis=1)
            xn = "di" if sh == "mp" else "doil"
            df = pd.concat([lhs.rename("y"), m[xn].rename("x"), rr.rename("z"), C], axis=1).loc["2006Q1":"2019Q4"].dropna()
            W = np.column_stack([np.ones(len(df)), df.drop(columns=["y", "x", "z"]).values])
            r = lambda v: v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
            ys.append(r(df.y.values)); xs.append(r(df.x.values)); zs.append(r(df.z.values)); ts.append(df.index.values)
        y = np.concatenate(ys); x = np.concatenate(xs); z = np.concatenate(zs) if sh == "mp" else x
        t = np.concatenate(ts)
        b = np.dot(z, y) / np.dot(z, x); e = y - b * x
        # Driscoll-Kraay: zaman başına skor toplamı, NW
        tu = pd.Series(z * e, index=t).groupby(level=0).sum().sort_index().values
        S = tu @ tu
        for l in range(1, h + 2):
            S += 2 * (1 - l / (h + 2)) * tu[l:] @ tu[:-l]
        se = np.sqrt(S) / abs(np.dot(z, x))
        sc = 10 if sh == "oil" else 1
        res[h] = (sc * b, sc * se)
    return res

if __name__ == "__main__":
    out, pooled, rr, F, m = run()
    print("RR şoku std:", rr.std().round(2), " medyan ilk aşama F:", round(F, 1))
    for (sh, kind), v in pooled.items():
        print(sh, kind, "panel β_h:", v[:, 0].round(2), "se:", v[:, 1].round(2))
