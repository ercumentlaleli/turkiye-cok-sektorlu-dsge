# Türkiye Çok Sektörlü Ağlı Genel Denge Modeli

Türkiye için küçük açık ekonomi Yeni Keynesyen DSGE modeli. Model 2006–2019 verisiyle Bayesçi yöntemle tahmin edilmiş,
yatırım, sermaye birikimi ve finansal hızlandırıcı içerecek şekilde genişletilmiş, sektörel uydu denklemleri (Schorfheide-Sill-Kryshko, 2010) ve LP-IV ile doğrulanmış, ardından TÜİK 2023 girdi-çıktı
tablolarına dayalı 32 sektörlü bir ağ modeline genişletilmiştir (Bouakez-Cardia-Ruge-Murcia; Pasten-Schoenle-Weber çizgisi).

Depoda tüm kod, veri, sonuçlar, teknik not (Word) ve etkileşimli bir Streamlit arayüzü bulunur.

## Streamlit arayüzü

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

| Sayfa | İçerik |
|---|---|
| Şok gezgini | 8 makro ve 32 sektörel şokun sektörel fiyat/üretim ve makro tepkileri. Ağsız modelle karşılaştırma ve CSV indirme içerir. |
| Senaryo analizi | Para politikası kuralı (φπ, ρᵢ, φe, φΔy), girdi-çıktı ağı ve sektörel fiyat katılıkları değiştirilerek 159 denklemli model canlı çözülür. Sonuç temel senaryoyla karşılaştırılır. |
| Model sonuçları | Kalibrasyon, Bayesçi tahmin, dönem karşılaştırması, SSK, LP-IV ve çok sektörlü model grafik ve tabloları |
| Metodoloji | Özet ve teknik notların indirilmesi |

## Klasör yapısı

```
streamlit_app.py          Uygulama giriş noktası (sayfa menüsü)
app_utils.py              Ortak grafik ve veri fonksiyonları
pages/                    Uygulama sayfaları
model/
  tr_dsge.py, tr_dsge2.py Toplulaştırılmış DSGE'nin önceki sürümleri (v1 kalibre, v2: 23 denklem, 8 şok)
  tr_dsge3.py             Tahmin edilen model (v3: yatırım + finansal hızlandırıcı; 32 denklem, 10 şok)
  dsge_est.py             Kalman filtresi, düzleştirici, posterior mod, RWMH, R-hat
  step1.py / s1_results.py  Genişletilmiş model tahmini (9 gözlenen) ve sonuçları
  step2.py / s2_results.py  2023Ç3–2026Ç2 kısmi yeniden tahmin
  ssk.py, ssk_fig.py      SSK uydu denklemleri (25 NACE-2 sektörü)
  lpiv.py, compare.py     LP-IV sağlaması ve ağ testi
  parse_io.py, io_agg.py, network.py  TÜİK girdi-çıktı tablosu işleme (64 → 32 sektör), Leontief ölçüleri
  calvo_est.py            Sektörel Calvo parametresi tahmini
  ms_model.py             32 sektörlü ağlı model (159 denklem)
  ms_experiments.py       Karşı-olgusal deneyler ve gezgin verisi (explorer_data.json)
  data/                   EVDS ve dış veriler (financial.csv: reel yatırım ve ticari kredi faizi), data/io/ altında TÜİK girdi-çıktı tabloları
docs/                     Teknik not (56 sayfa) ve şok tanımlaması eki (Word)
```

## Yeniden üretim sırası

Komutlar depo kökünden çalıştırılır; tüm yollar `model/` klasörüne göredir.

1. `python model/step1.py mode post` (önceki posteriordan başlatılan mod; ana mod olarak `s1_mode.json`'a kopyalanır), ardından dört zincir için `python model/step1.py chain <1-4> 60000` ve `python model/s1_results.py`: tahmin ve sonuçlar. Bu adım uzun sürer, MCMC zincirleri `.gitignore`'dadır. Depoda inceltilmiş posterior çekilişler (`s1_draws.npy`) bulunur.
2. `python model/step2.py mode`, ardından `python model/step2.py chain 2023Q3 <1-4> 60000` ve `python model/s2_results.py`: dönem karşılaştırması. Zincirler `.gitignore`'dadır; `s2_results.py` bunları ısınma sonrası 10'da bir inceltip `s2_draws.npy`'ye yazar ve depodaki bu dosyayla da çalışır.
3. `python model/ssk.py` ve ardından `python model/ssk_fig.py`: sektörel uydu denklemleri.
4. `python model/lpiv.py` ve ardından `python model/compare.py`: LP-IV sağlaması.
5. Sırayla `python model/parse_io.py`, `io_agg.py`, `network.py`, `calvo_est.py`, `ms_experiments.py`, `ms_bands.py`: çok sektörlü model ve belirsizlik bantları.

## Başlıca sonuçlar

- **Para politikası (+1 puan faiz):** Çok sektörlü modelde TÜFE düzeyi 4 çeyrekte %1,24, 8 çeyrekte %1,58 düşer; 8 çeyreklik kümülatif GSYH kaybı %4,2'dir. Yerli fiyatlar çok katıdır (θH ≈ 0,93).
- **Şokların payı (4 çeyrek):** TÜFE dalgalanmasının %64'ü yerli maliyet, %20'si kur primi şoklarından gelir. GSYH'de yurt içi talep %30, yerli maliyet %22, yatırım etkinliği %21, para politikası %18 paya sahiptir.
- **2023–2026 rejimi:** Yerli fiyat katılığı 0,93'ten 0,55'e düşer (ikinci bir posterior bölgesinde ≈ 0,1), yerli maliyet şoklarının oynaklığı yaklaşık üç katına çıkar; faiz şokunun etki anındaki dezenflasyon etkisi yaklaşık 1,1 puandır.
- **Fiyat katılığı heterojenliği:** Fedakârlık oranını yaklaşık %60 artırır (2,65'e karşı 1,71). Girdi-çıktı ağının makro şoklara etkisi sınırlıdır; asıl rolü girdi sağlayıcı sektörlerdeki (elektrik-gaz, tarım) maliyet şoklarını büyütmesidir.
- **Finansal kanal:** Kredi/GSYH verisi net servet şokunu tanımlar (yatırım dalgalanmasının %8'i); dış finansman priminin kaldıraca esnekliği küçüktür (ν ≈ 0,021).
- **Doğrulama:** Brent şokunda sektörel fiyat sıralaması SSK ile güçlü biçimde örtüşür (ρ = 0,80); faiz şokunda ortalama etki aynıdır ama sıralamadaki örtüşme zayıftır (ρ = 0,32, anlamlı değil). LP-IV'deki büyük gecikmeli üretim etkisi (8 çeyrekte −%4,2) yapısal modelde yakalanmamaktadır.

## Modelleme notları

- **Gözlenenler (11):** GSYH, tüketim, yatırım, TÜFE, politika faizi, reel kur, Brent, AB-4 öncü göstergesi, ticari kredi faizi farkı, kredi/GSYH büyümesi (BIS) ve 12 ay sonrası TÜFE beklentisi (Piyasa Katılımcıları Anketi, 2013 sonrası; öncesi eksik veri).
- **Verimlilik şoku:** Mevcut gözlenenlerle tanımlanamadığından parametreleri sabittir (σa = 0,50, ρa = 0,70). İstihdam verisi denenmiş; işsizlik kanalı olmayan modelde GSYH dalgalanmasının %82'sini verimlilik şokuna yüklediği için kullanılmamıştır.
- **Posterior:** 2006–2019'da dört zincir × 60.000 çekiliş, tek tepeli, tüm R-hat < 1,03. 2023–2026'da iki bölge vardır; geniş (ana) bölge raporlanır.
- **Belirsizlik bantları:** Şok gezgininde sektörel ve makro tepkiler, makro blok parametrelerinin 200 posterior çekilişiyle elde edilen %68/%90 bantlarıyla gösterilir (`model/ms_bands.py`). Çok sektörlü model QZ (genelleştirilmiş Schur) ile çözülür.
- **Yatırım ve finansal hızlandırıcı:** Sermaye birikimi, yatırım düzenleme maliyeti, Tobin q, yatırım malının yüksek ithal içeriği ve Bernanke-Gertler-Gilchrist tipi dış finansman primi.
- **Para politikası kuralı:** Kural çıktı düzeyine değil çıktı büyümesine tepki verir (φΔy).
- **Şok normalizasyonları:** Faiz +1 puan, kur primi TL reel %1 değer kaybı, Brent +%10; dış talep, yurt içi talep, yerli maliyet ve yatırım etkinliği şokları etki anında %1; finansal şok dış finansman priminde yıllık +1 puan.

## Sınırlar

- Modelde banka bilançosu ve kredi stoku, döviz cinsi borç bilanço etkisi ve maliye politikası yoktur. Bu nedenle büyük gecikmeli üretim etkileri ve kur değer kaybının bilanço kaynaklı daraltıcı etkisi eksik kalabilir.
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
