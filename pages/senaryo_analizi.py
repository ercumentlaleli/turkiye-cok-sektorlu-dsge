"""Canlı senaryo analizi: parametreleri değiştirip çok sektörlü modeli yeniden çözer."""
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import BASELINE, LINE, decimals, fmt, macro_lines, sector_bar
import ms_model as M

st.title("Senaryo analizi: modeli canlı çözün")
st.caption("Para politikası kuralını, girdi-çıktı ağını ve fiyat katılıklarını değiştirerek çok sektörlü modeli yeniden çözer "
           "(150 denklem, yaklaşık 1 saniye). Kesikli çizgiler ve işaretler temel senaryoyu (2006–2019 tahmini) gösterir.")

BASE_P = M.base_params()
THETA0 = M.CALVO.copy()
THETA_AVG = float((M.C_DOM * THETA0).sum())

AGG = [("e_m", "i1", "Faiz +1 puan"), ("e_rstar", "q1", "Kur primi (TL reel %1 değer kaybı)"), ("e_oil", "oil10", "Brent +%10"),
       ("e_ystar", None, "Dış talep (AB-4 öncü gös.) +1 std"), ("e_d", None, "Yurt içi talep +1 std"), ("e_cp", None, "Yerli maliyet +1 std")]
LAB = {s: l for s, l in zip(M.SECT, [
    "Tarım", "Madencilik", "Gıda-içecek-tütün", "Tekstil-giyim-deri", "Ağaç", "Kağıt", "Basım", "Rafine petrol", "Kimya", "Eczacılık",
    "Kauçuk-plastik", "Metalik olm. mineral", "Ana metal", "Fabrikasyon metal", "Elektronik", "Elektrikli teçh.", "Makine (+onarım)",
    "Motorlu taşıt", "Diğer ulaşım ar.", "Mobilya-diğer imalat", "Elektrik-gaz", "Su-atık", "İnşaat", "Ticaret", "Ulaştırma",
    "Konaklama-yiyecek", "Bilgi-iletişim", "Finans", "Gayrimenkul", "Mesleki-idari hiz.", "Kamu-eğitim-sağlık", "Diğer hizmetler"])}


@st.cache_resource(show_spinner=False)
def solve_cached(key):
    pdict, theta, network = dict(key[0]), np.array(key[1]), key[2]
    p = dict(BASE_P); p.update(pdict)
    return M.solve(p, theta=theta, network=network), p


def run(key, shock, norm):
    sol, p = solve_cached(key)
    z = M.summarize(M.irf(sol, p, shock, 17, norm))
    macro = {"cpi": z["cpi_level"], "pi": z["pi"], "y": z["y"], "i": z["i"], "q": z["q"]}
    return z, macro, sol


with st.sidebar:
    st.header("Şok")
    kind = st.radio("Tür", ["Makro", "Sektörel"], horizontal=True)
    if kind == "Makro":
        j = st.selectbox("Makro şok", range(len(AGG)), format_func=lambda i: AGG[i][2])
        shock, norm, shock_lab = AGG[j][0], AGG[j][1], AGG[j][2]
    else:
        sec = st.selectbox("Sektör (%1 maliyet şoku)", M.SECT, format_func=lambda s: f"{s} {LAB[s]}")
        shock, norm, shock_lab = f"e_u_{sec}", None, f"{sec} {LAB[sec]}: %1 maliyet şoku"
    h = st.slider("Ufuk (çeyrek)", 0, 16, 4)
    var = st.radio("Sektörel değişken", ["Fiyat düzeyi", "Üretim"], horizontal=True)

    st.header("Para politikası kuralı")
    phi_pi = st.slider("Enflasyona tepki φπ", 1.05, 3.0, float(round(BASE_P["phi_pi"], 2)), 0.05)
    rho_i = st.slider("Faiz düzleştirme ρᵢ", 0.0, 0.95, float(round(BASE_P["rho_i"], 2)), 0.01)
    phi_e = st.slider("Kura tepki φe", 0.0, 0.5, float(round(BASE_P["phi_e"], 2)), 0.01)
    phi_y = st.slider("Üretime tepki φy", 0.0, 1.0, float(round(BASE_P["phi_y"], 2)), 0.01)

    st.header("Yapı")
    network = st.checkbox("Girdi-çıktı ağı açık", value=True)
    calvo_mode = st.radio("Sektörel fiyat katılığı", ["Tahmin edilen (heterojen)", "Tek tip (ortalama)", "Ölçekli"], index=0)
    scale = 1.0
    if calvo_mode == "Ölçekli":
        scale = st.slider("Fiyat değiştirme olasılığı çarpanı (1−θ)", 0.25, 3.0, 1.0, 0.05,
                          help="1'den büyük değerler fiyatların daha sık değiştiği (ör. yüksek enflasyon) bir ortamı temsil eder.")

if calvo_mode == "Tek tip (ortalama)":
    theta = np.full(M.K, THETA_AVG)
elif calvo_mode == "Ölçekli":
    theta = np.clip(1 - (1 - THETA0) * scale, 0.05, 0.97)
else:
    theta = THETA0
if calvo_mode == "Ölçekli" and scale == 1.0:
    theta = THETA0

pdict = {}
for k_, v_ in {"phi_pi": phi_pi, "rho_i": rho_i, "phi_e": phi_e, "phi_y": phi_y}.items():
    if abs(v_ - round(BASE_P[k_], 2)) > 1e-9:  # kaydırıcı varsayılanda ise tahmin edilen tam değeri koru
        pdict[k_] = v_
key_s = (tuple(sorted(pdict.items())), tuple(np.round(theta, 6)), network)
key_b = (tuple(), tuple(np.round(THETA0, 6)), True)

try:
    with st.spinner("Model çözülüyor..."):
        zb, mb, _ = run(key_b, shock, norm)
        zs, ms, sol = run(key_s, shock, norm)
except Exception as e:  # tekil olmayan denge vb.
    st.error(f"Bu parametrelerle model çözülemedi (tekil denge yok olabilir): {e}")
    st.stop()

if sol["n_stable"] != M.N:
    st.warning(f"Tekillik koşulu sağlanmıyor: {sol['n_stable']} kararlı kök / {M.N} değişken. Sonuçlar güvenilir değildir.")

st.subheader(shock_lab)
k = [("TÜFE düzeyi (%)", "cpi"), ("TÜFE enflasyonu (yıllık puan)", "pi"), ("GSYH (%)", "y"), ("Politika faizi (yıllık puan)", "i"), ("Reel kur (%)", "q")]
dd = decimals([ms[c][h] for _, c in k] + [mb[c][h] for _, c in k])
cols = st.columns(5)
for c, (lab, key) in zip(cols, k):
    c.metric(f"{lab} · {h}. çeyrek", fmt(ms[key][h], dd), delta=fmt(ms[key][h] - mb[key][h], dd) + " temele göre", delta_color="off")

left, right = st.columns([1.6, 1])
isP = var == "Fiyat düzeyi"
vs = zs["P_level"][h] if isP else zs["y_k"][h]
vb = zb["P_level"][h] if isP else zb["y_k"][h]
with left:
    st.plotly_chart(sector_bar(M.SECT, [LAB[s] for s in M.SECT], vs, ghost=vb, ghost_name="Temel senaryo",
                               title=("Sektörel fiyat düzeyi" if isP else "Sektörel üretim") + f" — {h}. çeyrek (%)"),
                    width="stretch")
with right:
    st.markdown("**Makro tepkiler** (düz: senaryo, kesikli: temel)")
    for fig in macro_lines([("Senaryo", ms, LINE, False), ("Temel", mb, BASELINE, True)], h_mark=h):
        st.plotly_chart(fig, width="stretch")

cum_y_s, cum_y_b = float(np.sum(zs["y"][:8])), float(np.sum(zb["y"][:8]))
c1, c2 = st.columns(2)
c1.metric("8 çeyrek kümülatif GSYH etkisi (%)", fmt(cum_y_s), delta=fmt(cum_y_s - cum_y_b) + " temele göre", delta_color="off")
if shock == "e_m" and abs(ms["cpi"][8]) > 1e-6:
    sr_s, sr_b = cum_y_s / ms["cpi"][8], cum_y_b / mb["cpi"][8]
    c2.metric("Fedakârlık oranı (GSYH kaybı / TÜFE düşüşü, 8 çeyrek)", f"{sr_s:.2f}".replace(".", ","),
              delta=f"{sr_s - sr_b:+.2f}".replace(".", ",") + " temele göre", delta_color="off")

with st.expander("Parametre özeti"):
    st.dataframe(pd.DataFrame({
        "Parametre": ["φπ", "ρi", "φe", "φy", "Ağ", "Tüketim ağırlıklı θ"],
        "Temel": [f"{BASE_P['phi_pi']:.3f}", f"{BASE_P['rho_i']:.3f}", f"{BASE_P['phi_e']:.3f}", f"{BASE_P['phi_y']:.3f}", "Açık", f"{THETA_AVG:.3f}"],
        "Senaryo": [f"{phi_pi:.3f}", f"{rho_i:.3f}", f"{phi_e:.3f}", f"{phi_y:.3f}", "Açık" if network else "Kapalı", f"{float((M.C_DOM * theta).sum()):.3f}"]}),
        hide_index=True, width="stretch")
st.caption("Not: Senaryo analizi, makro bloğun diğer parametrelerini 2006–2019 posterior ortalamalarında tutar. Parametre değişikliklerinin "
           "beklentiler üzerinden etkisi modelde tam olarak yer alır (rasyonel beklentiler); ancak sonuçlar modelin varsayımlarına bağlıdır.")
