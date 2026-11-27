"""Model sonuçları: tahmin, doğrulama ve çok sektörlü model grafikleri ve tabloları."""
import json

import pandas as pd
import streamlit as st

from app_utils import MODEL_DIR

st.title("Model sonuçları")
st.caption("Tüm grafikler ve tablolar depodaki kodla üretilmiştir (model/ klasörü). Ayrıntılı yorumlar teknik nottadır (Metodoloji sayfası).")


def img(fn, cap):
    p = MODEL_DIR / fn
    if p.exists():
        st.image(str(p), caption=cap, width="stretch")


tabs = st.tabs(["1. Kalibre model", "2. Bayesçi tahmin", "3. Dönem karşılaştırması", "4. Sektörel (SSK)", "5. LP-IV sağlaması",
                "6. Çok sektörlü model", "7. Rejim değişkenli model"])

with tabs[0]:
    st.markdown("Literatür ve Türkiye'nin yapısal özelliklerine dayalı kalibrasyonla elde edilen etki-tepkiler.")
    for fn, cap in [("1_para_politikasi.png", "Faiz +1 puan"), ("2_risk_primi.png", "Risk primi +400 baz puan"),
                    ("3_enerji.png", "İthal enerji fiyatı +%20"), ("4_dis_talep.png", "Dış talep −%1")]:
        img(fn, cap)

with tabs[1]:
    st.markdown("**Genişletilmiş model** (12 gözlenen, 10 şok; yatırım, finansal hızlandırıcı, döviz cinsi firma borcu, kredi/GSYH, "
                "enflasyon beklentisi ve sanayide saat başına verimlilik dahil; 2006Ç1–2019Ç4; dört zincir × 60.000 çekiliş).")
    st.info("Posterior tek tepelidir (tüm R-hat ≤ 1,03). Verimlilik şoku sanayide çalışılan saat başına üretimle, net servet şoku "
            "kredi/GSYH büyümesiyle tanımlanmaktadır.")
    t = pd.read_csv(MODEL_DIR / "s1_posterior_table.csv")
    t = t.rename(columns={"param": "Parametre", "prior": "Önsel", "prior_mean": "Önsel ort.", "post_mean": "Posterior ort.",
                          "p05": "%5", "p95": "%95", "rhat": "R-hat", "onceki_model": "İlk tahmin"})
    st.dataframe(t.round(3), hide_index=True, width="stretch")
    s1 = json.load(open(MODEL_DIR / "s1_summary.json", encoding="utf-8"))
    st.markdown("**Varyans ayrıştırması, 4 çeyrek (%)**")
    st.dataframe(pd.DataFrame(s1["fevd4"]).round(1), width="stretch")
    img("s1_1_irf.png", "Etki-tepki fonksiyonları (medyan, %68 ve %90 bantları)")
    img("s1_2_fevd.png", "Varyans ayrıştırması")
    img("s1_3_tarihsel_enflasyon.png", "Enflasyonun tarihsel şok ayrıştırması")
    with st.expander("İlk tahmin (5 gözlenen) grafikleri"):
        for fn, cap in [("est_1_irf_bantlari.png", "Etki-tepkiler"), ("est_2_varyans_ayristirmasi.png", "Varyans ayrıştırması"),
                        ("est_3_tarihsel_enflasyon.png", "Enflasyonun tarihsel ayrıştırması"), ("est_4_tarihsel_buyume.png", "GSYH büyümesinin tarihsel ayrıştırması")]:
            img(fn, cap)

with tabs[2]:
    st.markdown("2023Ç3–2026Ç2 dönemi (12 gözlem) için kısmi yeniden tahmin. Büyüklükler değil yönler dikkate alınmalıdır.")
    st.dataframe(pd.read_csv(MODEL_DIR / "s2_compare_table.csv").round(3), hide_index=True, width="stretch")
    img("s2_1_donem_irf.png", "Aynı şoka iki dönemde verilen tepki")

with tabs[3]:
    st.markdown("Schorfheide-Sill-Kryshko (2010) uydu denklemleri ile 25 NACE-2 sektörünün tepkileri.")
    img("s3_1_ssk_fiyat.png", "Üretici fiyatı tepkileri, 4 çeyrek sonra")
    img("s3_2_ssk_uretim.png", "Sanayi üretimi tepkileri, etki anı")

with tabs[4]:
    s4 = json.load(open(MODEL_DIR / "s4_summary.json", encoding="utf-8"))
    st.markdown(f"Romer-Romer tipi araç değişkenli yerel projeksiyon (medyan ilk aşama F ≈ {s4['F_first_stage_median']:.0f}).")
    rows = [{"Karşılaştırma": k, **v} for k, v in s4["kesit"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    img("s4_1_ssk_vs_lp.png", "Sektör ortalaması tepkiler: SSK ve LP-IV")
    img("s4_2_kesit.png", "Sektör bazında karşılaştırma")
    st.info("LP-IV'de faiz şokunun etki anında zayıf bir price puzzle görülmektedir (25 sektörün 19'unda pozitif, 8'inde anlamlı); "
            "iki çeyrek içinde düşüşe döner. Ayrıntı teknik nottadır.")

with tabs[5]:
    s5 = json.load(open(MODEL_DIR / "s5_summary.json", encoding="utf-8"))
    st.markdown("32 sektörlü, girdi-çıktı ağlı model: karşı-olgusal deneyler ve sektöre özgü şoklar.")
    cf = pd.DataFrame([{"Şok | model": k, **v} for k, v in s5["karsi_olgusal"].items()])
    st.dataframe(cf, hide_index=True, width="stretch")
    img("s5_2_karsi_olgusal.png", "Ağın ve fiyat katılığı heterojenliğinin rolü")
    img("s5_1_isi_haritasi.png", "Sektörel fiyat düzeyi tepkileri, 4 çeyrek sonra")
    img("s5_3_sektorel_soklar.png", "Sektöre özgü maliyet şoklarının TÜFE etkisi")
    img("s5_4_dogrulama.png", "Çok sektörlü model ile SSK karşılaştırması")
    st.markdown("**Sektörel Calvo parametreleri**")
    st.dataframe(pd.read_csv(MODEL_DIR / "data" / "io" / "calvo_sector.csv").round(3), hide_index=True, width="stretch")

with tabs[6]:
    st.markdown("**Markov rejim değişkenli DSGE (2006Ç1–2026Ç2).** İki bağımsız zincir: para politikası (sistematik/oynak) ve fiyatlama "
                "(normal/yüksek enflasyon); toplam dört rejim. Farmer-Waggoner-Zha çözümü ve Kim filtresiyle tahmin edilmiştir. "
                "Ajanlar rejim değişimi olasılığını beklentilerine yansıtır. Salgın çeyrekleri (2020Ç2–Ç4) eksik veri sayılmıştır.")
    pm = MODEL_DIR / "ms_summary.json"
    if pm.exists():
        ms = json.load(open(pm, encoding="utf-8"))
        rp = pd.DataFrame(ms["rejim_olasiligi_donem_ort"]).T.rename(columns={"oynak_politika": "Oynak politika olasılığı",
                                                                            "yuksek_enflasyon": "Yüksek enflasyon olasılığı"})
        st.markdown("**Dönem ortalaması rejim olasılıkları**")
        st.dataframe(rp, width="stretch")
        img("ms_1_rejim_olasiliklari.png", "Filtrelenmiş rejim olasılıkları")
        img("ms_2_rejim_irf.png", "Rejime göre etki-tepkiler")
        t = pd.read_csv(MODEL_DIR / "ms_posterior_table.csv").rename(columns={"Unnamed: 0": "Parametre", "prior": "Önsel",
            "prior_mean": "Önsel ort.", "post_mean": "Posterior ort.", "p05": "%5", "p95": "%95", "rhat": "R-hat"})
        st.markdown("**Posterior tablo** (A2_: oynak politika rejimi; B2_: yüksek enflasyon rejimi; off_: rejime özgü nominal sabitler; "
                    "pXX: rejimde kalma olasılıkları)")
        st.dataframe(t.round(3), hide_index=True, width="stretch")
