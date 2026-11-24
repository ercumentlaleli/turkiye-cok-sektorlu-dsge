"""Türkiye çok sektörlü ağlı model — şok gezgini (ana sayfa)."""
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import FOOTNOTE, LINE, decimals, fmt, load_explorer, macro_lines, sector_bar


D = load_explorer()
SECT, LAB = D["sectors"], D["labels"]
shocks = D["shocks"]

st.title("Türkiye çok sektörlü ağlı model: şok gezgini")
st.caption("32 sektörlü, TÜİK 2023 girdi-çıktı ağlı ve sektörel Calvo parametreli küçük açık ekonomi modeli. "
           "Makro blok 2006–2019 Bayesçi tahminine dayanır. Bir şok seçin; sektörel fiyat ve üretim tepkileri ile makro etkiler gösterilir.")

with st.sidebar:
    st.header("Ayarlar")
    kind = st.radio("Şok türü", ["Makro şok", "Sektörel şok"], horizontal=True)
    opts = [i for i, s in enumerate(shocks) if s["kind"] == kind]
    idx = st.selectbox("Şok", opts, format_func=lambda i: shocks[i]["label"])
    var = st.radio("Değişken", ["Fiyat düzeyi", "Üretim"], horizontal=True)
    h = st.slider("Ufuk (çeyrek)", 0, 16, 4)
    ghost_on = st.checkbox("Ağsız modeli göster (fiyat)", value=True, disabled=(var != "Fiyat düzeyi"))
    st.markdown("---")
    st.caption("Diğer sayfalar: canlı senaryo analizi, model sonuçları ve metodoloji (soldaki menü).")

s = shocks[idx]
m = s["macro"]
isP = var == "Fiyat düzeyi"
vals = np.array([row[h] for row in (s["P"] if isP else s["Y"])])
ghost = np.array([row[h] for row in s["P_nonet"]]) if (isP and ghost_on) else None

# KPI satırı
k = [("TÜFE düzeyi (%)", m["cpi"][h]), ("TÜFE enflasyonu (yıllık puan)", m["pi"][h]), ("GSYH (%)", m["y"][h]),
     ("Politika faizi (yıllık puan)", m["i"][h]), ("Reel kur (%)", m["q"][h])]
d = decimals([x for _, x in k])
cols = st.columns(5)
for c, (lab, v) in zip(cols, k):
    c.metric(f"{lab} · {h}. çeyrek", fmt(v, d))

left, right = st.columns([1.6, 1])
with left:
    title = ("Sektörel fiyat düzeyi" if isP else "Sektörel üretim") + f" — {h}. çeyrek (%)"
    st.plotly_chart(sector_bar(SECT, LAB, vals, ghost, title=title), width="stretch")
with right:
    st.subheader("Makro tepkiler")
    for fig in macro_lines([("Tam model", m, LINE, False)], h_mark=h):
        st.plotly_chart(fig, width="stretch")

st.subheader(f"En çok etkilenen sektörler — {h}. çeyrek")
top = np.argsort(-np.abs(vals))[:10]
df = pd.DataFrame({"Sektör": [f"{SECT[i]} {LAB[i]}" for i in top],
                   ("Fiyat (%)" if isP else "Üretim (%)"): vals[top].round(4),
                   "Calvo θ": [D["theta"][i] for i in top],
                   "Tüketim payı (%)": [round(100 * D["c_share"][i], 1) for i in top]})
if ghost is not None:
    df.insert(2, "Ağsız (%)", ghost[top].round(4))
    df.insert(3, "Ağ katkısı (%)", (vals[top] - ghost[top]).round(4))
st.dataframe(df, hide_index=True, width="stretch")

with st.expander("Tüm sektörler ve tüm ufuklar (indirilebilir tablo)"):
    arr = np.array(s["P"] if isP else s["Y"])
    full = pd.DataFrame(arr, index=[f"{a} {b}" for a, b in zip(SECT, LAB)], columns=[f"h{t}" for t in range(arr.shape[1])])
    st.dataframe(full.round(4), width="stretch")
    st.download_button("CSV indir", full.to_csv().encode("utf-8"), file_name=f"{s['id']}_{'fiyat' if isP else 'uretim'}.csv")

st.caption(FOOTNOTE)
