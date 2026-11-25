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
| Senaryo analizi | Para politikası kuralı (φπ, ρᵢ, φe, φΔy), girdi-çıktı ağı ve sektörel fiyat katılıkları değiştirilerek 150 denklemli model canlı çözülür. Sonuç temel senaryoyla karşılaştırılır. |
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

1. `python model/step1.py mode`, ardından dört zincir için `python model/step1.py chain <1-4> 60000` ve `python model/s1_results.py`: tahmin ve sonuçlar. Bu adım uzun sürer, MCMC zincirleri `.gitignore`'dadır. Depoda inceltilmiş posterior çekilişler (`s1_draws.npy`) bulunur.
2. `python model/step2.py mode`, ardından `python model/step2.py chain 2023Q3 <1-4> 60000` ve `python model/s2_results.py`: dönem karşılaştırması. Zincirler `.gitignore`'dadır; `s2_results.py` bunları ısınma sonrası 10'da bir inceltip `s2_draws.npy`'ye yazar ve depodaki bu dosyayla da çalışır.
3. `python model/ssk.py` ve ardından `python model/ssk_fig.py`: sektörel uydu denklemleri.
4. `python model/lpiv.py` ve ardından `python model/compare.py`: LP-IV sağlaması.
5. Sırayla `python model/parse_io.py`, `io_agg.py`, `network.py`, `calvo_est.py`, `ms_experiments.py`: çok sektörlü model.

## Başlıca sonuçlar

- **Para politikası (+1 puan faiz):** TÜFE düzeyi 4 çeyrekte %1,86, 8 çeyrekte %2,04 düşer. 8 çeyreklik kümülatif GSYH kaybı %4,4'tür.
- **Fiyat katılığı heterojenliği:** Tek tip Calvo varsayımı faiz şokunun 4 çeyrekteki dezenflasyon etkisini yaklaşık %8 abartır (−2,01'e karşı −1,86). Heterojenlik fedakârlık oranını yaklaşık %50 artırır. Girdi-çıktı ağı faiz şokunun etkisini biraz güçlendirir (−1,79'dan −1,86'ya).
- **Enerji (+%10 Brent):** Ağ, 4 çeyrekteki TÜFE düzeyi etkisini yaklaşık %11 artırır (0,24'ten 0,27'ye). Brent geçişinin hızını asıl olarak enerji sektörlerinin esnek fiyatları belirler.
- **Doğrulama:** Çok sektörlü modelin sektörel fiyat tepkileri SSK ile tutarlıdır. Sıra korelasyonu faiz şokunda 0,49 (p = 0,03), petrol şokunda 0,74'tür. SSK ile LP-IV, faiz şokunun 4 çeyrekteki fiyat etkisinde işaret olarak %96 uyumludur. Üretim tepkilerinde uyum zayıftır.

## Modelleme notları

- **Para politikası kuralı:** Tahmin edilen kural çıktı düzeyine değil çıktı büyümesine tepki verir (φΔy). Çıktı düzeyine tepki veren kural, net dış varlıkların çok yavaş eridiği şoklarda (dış talep, Brent) enflasyonu on yıllarca sıfırdan uzak tutuyor ve TÜFE düzeyinin sürekli kaymasına yol açıyordu. İki kural veriye aynı ölçüde uyar (Laplace log marjinal olabilirlik −894,0 ve −894,5).
- **İki tepeli posterior:** 2006–2019 posteriorunun ikinci bir tepesi vardır (kalıcı maliyet şoku, ρcp ≈ 0,95; daha esnek yerli fiyatlar, θH ≈ 0,66). Laplace yaklaşımına göre olasılığın yaklaşık %87'si baskın tepededir; tüm sonuçlar bu tepeye (ρcp < 0,75) koşulludur.
- **Şok normalizasyonları:** Faiz +1 puan, kur primi TL reel %1 değer kaybı, Brent +%10; dış talep, yurt içi talep ve yerli maliyet şokları etki anında %1 (sırasıyla AB-4 öncü göstergesi, özel tüketim, yerli üretici fiyatları).

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
