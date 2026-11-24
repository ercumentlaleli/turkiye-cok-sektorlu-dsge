# Türkiye Çok Sektörlü Ağlı Genel Denge Modeli

Türkiye için küçük açık ekonomi Yeni Keynesyen DSGE modeli. Model 2006–2019 verisiyle Bayesçi yöntemle tahmin edilmiş,
sektörel uydu denklemleri (Schorfheide-Sill-Kryshko, 2010) ve LP-IV ile doğrulanmış, ardından TÜİK 2023 girdi-çıktı
tablolarına dayalı 32 sektörlü bir ağ modeline genişletilmiştir (Bouakez-Cardia-Ruge-Murcia; Pasten-Schoenle-Weber çizgisi).

Depoda tüm kod, veri, sonuçlar, teknik not (Word) ve etkileşimli bir Streamlit arayüzü bulunur.

## Streamlit arayüzü

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

| Sayfa | İçerik |
|---|---|
| Şok gezgini | 6 makro ve 32 sektörel şokun sektörel fiyat/üretim ve makro tepkileri. Ağsız modelle karşılaştırma ve CSV indirme içerir. |
| Senaryo analizi | Para politikası kuralı (φπ, ρᵢ, φe, φy), girdi-çıktı ağı ve sektörel fiyat katılıkları değiştirilerek 150 denklemli model canlı çözülür. Sonuç temel senaryoyla karşılaştırılır. |
| Model sonuçları | Kalibrasyon, Bayesçi tahmin, dönem karşılaştırması, SSK, LP-IV ve çok sektörlü model grafik ve tabloları |
| Metodoloji | Özet ve teknik notların indirilmesi |

## Klasör yapısı

```
streamlit_app.py          Uygulama giriş noktası (sayfa menüsü)
app_utils.py              Ortak grafik ve veri fonksiyonları
pages/                    Uygulama sayfaları
model/
  tr_dsge.py, tr_dsge2.py Toplulaştırılmış DSGE (v2: 23 denklem, 8 şok)
  dsge_est.py             Kalman filtresi, düzleştirici, posterior mod, RWMH, R-hat
  step1.py / s1_results.py  Genişletilmiş model tahmini (7 gözlenen) ve sonuçları
  step2.py / s2_results.py  2023Ç3–2026Ç2 kısmi yeniden tahmin
  ssk.py, ssk_fig.py      SSK uydu denklemleri (25 NACE-2 sektörü)
  lpiv.py, compare.py     LP-IV sağlaması ve ağ testi
  parse_io.py, io_agg.py, network.py  TÜİK girdi-çıktı tablosu işleme (64 → 32 sektör), Leontief ölçüleri
  calvo_est.py            Sektörel Calvo parametresi tahmini
  ms_model.py             32 sektörlü ağlı model (150 denklem)
  ms_experiments.py       Karşı-olgusal deneyler ve gezgin verisi (explorer_data.json)
  data/                   EVDS ve dış veriler, data/io/ altında TÜİK girdi-çıktı tabloları
docs/                     Teknik not (56 sayfa) ve şok tanımlaması eki (Word)
```

## Yeniden üretim sırası

Komutlar depo kökünden çalıştırılır; tüm yollar `model/` klasörüne göredir.

1. `python model/step1.py` ve ardından `python model/s1_results.py`: tahmin ve sonuçlar. Bu adım uzun sürer, MCMC zincirleri `.gitignore`'dadır. Depoda inceltilmiş posterior çekilişler (`s1_draws.npy`) bulunur.
2. `python model/step2.py` ve ardından `python model/s2_results.py`: dönem karşılaştırması.
3. `python model/ssk.py` ve ardından `python model/ssk_fig.py`: sektörel uydu denklemleri.
4. `python model/lpiv.py` ve ardından `python model/compare.py`: LP-IV sağlaması.
5. Sırayla `python model/parse_io.py`, `io_agg.py`, `network.py`, `calvo_est.py`, `ms_experiments.py`: çok sektörlü model.

## Başlıca sonuçlar

- **Para politikası (+1 puan faiz):** TÜFE düzeyi 4 çeyrekte %1,41, 8 çeyrekte %1,48 düşer. 8 çeyreklik kümülatif GSYH kaybı %3,4'tür.
- **Fiyat katılığı heterojenliği:** Tek tip Calvo varsayımı faiz şokunun dezenflasyon etkisini yaklaşık %30 abartır (4 çeyrekte −1,85'e karşı −1,41). Girdi-çıktı ağı bu etkiyi biraz güçlendirir (−1,34'ten −1,41'e).
- **Enerji (+%10 Brent):** Ağ, etki yılındaki TÜFE düzeyi etkisini yaklaşık iki katına çıkarır (0,07'den 0,15'e).
- **Doğrulama:** Çok sektörlü modelin sektörel fiyat tepkileri SSK ile tutarlıdır. Sıra korelasyonu faiz şokunda 0,51 (p = 0,02), petrol şokunda 0,78'dir. SSK ile LP-IV, faiz şokunun 4 çeyrekteki fiyat etkisinde işaret olarak %96 uyumludur. Üretim tepkilerinde uyum zayıftır.

## Sınırlar

- Modelde yatırım, kredi stoku, döviz cinsi borç bilanço etkisi ve maliye politikası yoktur. Bu nedenle üretim ve kur primi şoku tepkileri temkinle yorumlanmalıdır.
- Yabancı enflasyon sıfır varsayılmaktadır (yurt dışı fiyatlar dışsal ve sabit).
- 2023 sonrası dönem tahmini 12 gözleme dayanır; yalnızca yön bilgisi verir.
- LP-IV'de faiz şokunun etki anında zayıf bir price puzzle görülür.

## Veri kaynakları

- TCMB EVDS: GSYH, özel tüketim, TÜFE, politika faizi, reel efektif kur, sektörel Yİ-ÜFE ve sanayi üretimi
- TÜİK: 2023 girdi-çıktı, ithalat kullanım ve ithalat girdi-çıktı tabloları
- OECD: AB-4 öncü göstergesi
- Brent petrol fiyatı (`model/data/foreign.csv`)

## Lisans ve atıf

Araştırma amaçlıdır. Sonuçlar resmi kurum görüşü değildir.
