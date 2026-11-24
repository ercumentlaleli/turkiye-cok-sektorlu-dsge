"""Metodoloji özeti ve teknik dokümanlar."""
import streamlit as st

from app_utils import DOCS_DIR

st.title("Metodoloji")

st.markdown("""
Çalışma birbirini doğrulayan beş aşamadan oluşur:

| Aşama | Yöntem | Veri | Çıktı |
|---|---|---|---|
| 1. Toplulaştırılmış DSGE | Log-doğrusal Yeni Keynesyen küçük açık ekonomi modeli; doğrusal zaman iterasyonu ile çözüm | Kalibrasyon | Etki-tepkiler |
| 2. Bayesçi tahmin | Kalman filtresi, posterior mod, Metropolis-Hastings | EVDS: GSYH, tüketim, TÜFE, faiz, reel kur, Brent, AB-4 öncü göstergesi | Posterior dağılımlar, varyans ve tarihsel ayrıştırma |
| 3. SSK uydu denklemleri | DSGE durumlarına AR(1) hatalı uydu regresyonları | 25 NACE-2 sektörü (Yİ-ÜFE, sanayi üretimi) | Sektörel tepkiler |
| 4. LP-IV sağlaması | Yerel projeksiyon, Romer-Romer tipi araç | Aynı sektörel veri | Tutarlılık istatistikleri |
| 5. Çok sektörlü ağlı model | 32 sektörlü genel denge, tahmin edilen sektörel Calvo parametreleri | TÜİK 2023 girdi-çıktı tabloları | Karşı-olgusal deneyler, şok gezgini |

**Denklem sayıları:** genişletilmiş toplulaştırılmış DSGE 23 denklem (15 yapısal + 8 şok süreci); çok sektörlü model 150 denklem
(22 makro + 32 sektör × 4). Her iki modelde de denklem sayısı değişken sayısına eşittir ve tekillik koşulu sağlanır.

**Modelde yer alan başlıca kanallar:** faiz–talep, risk primi–kredi faizi, eksik kur geçişkenliği, ithal ara girdi ve enerji maliyeti,
girdi-çıktı ağı, sektörel fiyat katılığı heterojenliği, dış talep, net dış varlık–risk primi, ücret katılığı.

**Yer almayan kanallar:** yatırım ve kredi stoku, döviz cinsi borç bilanço etkisi, maliye politikası, beklentilerde öğrenme ve
güvenilirlik, rezerv ve makroihtiyati araçlar. Bu nedenle üretim tepkileri ve kur primi şokunun kısa vadeli büyüme etkisi temkinle yorumlanmalıdır.
""")

st.subheader("Dokümanlar")
for fn, lab in [("Turkiye_Cok_Sektorlu_Genel_Denge_Modeli_Teknik_Not.docx", "Teknik not (Word, 56 sayfa)"),
                ("Sok_Tanimlamasi_Teknik_Not_Eki.docx", "Şokların tanımlanması — teknik not eki (Word)")]:
    p = DOCS_DIR / fn
    if p.exists():
        st.download_button(lab, p.read_bytes(), file_name=fn,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
