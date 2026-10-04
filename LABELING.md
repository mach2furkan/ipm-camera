# Etiketleme ve eğitim kabul kuralları

Sınıf sırası sabittir. ID değiştirmek mevcut modelle uyumu bozar.

| ID | Sınıf | Kabul edilen nesne ve kaynak sınırı |
|---|---|---|
| 0 | insan | İnsan; Open Images Person kutuları |
| 1 | kitap | Kitap; dergi ve kâğıt yığını otomatik olarak kitap sayılmaz |
| 2 | dizustu_bilgisayar | Dizüstü bilgisayar; monitör ayrı sınıftır |
| 3 | monitor | Bilgisayar monitörü; COCO TV kutusu inceleme olmadan kabul edilmez |
| 4 | klavye | Bilgisayar klavyesi |
| 5 | fare | Bilgisayar faresi; hayvan değildir |
| 6 | telefon | Cep telefonu; kaynak genel telefon sınıfını kapsamaz |
| 7 | bardak | Bardak/fincan; Open Images Coffee cup yalnızca bir alt tür sağlar |
| 8 | sise | Şişe |
| 9 | kalem | Yazı kalemi; kaynak Pen, kurşun kalem çeşitliliğini garanti etmez |

Her görüntüde **on hedef sınıfın tamamı** kontrol edilmelidir. Kaynak etiketinin
olmaması nesnenin görüntüde olmadığı anlamına gelmez. COCO kalemi etiketlemez;
COCO'dan boş kutuyla gelen görüntüler doğrulanmış negatif değildir. Open Images
etiketleri de tüm hedef nesnelerin eksiksiz işaretlendiğine kanıt değildir.
TV monitör gibi görünse bile sınıf kararı kaynak ID'sinden çıkarılmamalıdır.

Kamerada fiziksel nesneler hedeflenir. Duvar fotoğrafındaki insan, ekran içindeki
nesne ve çizimler gerçek nesne gibi etiketlenmez. Open Images kaynak alanları
bu ayrımı bütün görüntüler için otomatik garanti etmez.

Görünür nesneyi sıkı çevreleyen kutu kullanın; nesnenin görünmeyen kısmını tahmin
etmeyin. Grup, çizim ve içerideki nesne kutuları otomatik veri hazırlamada elenir.
Küçük kutular tek başına silinmez: görüntü incelemeye ayrılır. Boş etiket dosyası
ancak bütün hedef sınıfların bulunmadığı gözle kontrol edilmişse onaylanır.

## İnceleme

Tarayıcıdan doğrudan kayıt yapabilen yerel düzenleyici:

```powershell
& .venv/Scripts/python.exe -m tools.review_server
```

`http://127.0.0.1:8765` adresini açın. **Taslağı kaydet** etiketleri kaydeder,
görüntüyü onaylamaz. **İncelemeyi onayla ve kaydet** tüm hedef sınıflar kontrol
edildikten sonra onay verir. **Son kaydı geri al** önceki etiketleri geri yükler ve
yeniden inceleme ister. Eski etiket hashleri veya eski inceleme sürümüyle yapılan
eşzamanlı kayıtlar reddedilir. CLI ve yerel hizmet aynı dosya kilidini kullanır.
Hizmet yalnızca bu bilgisayarda dinler; görüntü dışındaki dosyaları HTTP üzerinden
açmaz. Kapatmak için çalışan terminalde Ctrl+C kullanın.

Bağımsız HTML/JSON akışı da kullanılabilir:

Kutuları görsel olarak düzeltmek için:

```powershell
& .venv/Scripts/python.exe -m tools.label_editor
```

Oluşan `dataset_cctv_desk/label-editor.html` dosyasını açın. %100/%200 büyütme,
yeni kutu çizme, sınıf değiştirme, silme ve geri alma desteklenir. Her sınıf için
kontrol kutusunu işaretleyin, inceleyen ve oturum kimliğini girin; açık bir bulgu
varsa düzeltildiğini ayrıca belirtin. Onay kaydını JSON olarak indirin.

```powershell
& .venv/Scripts/python.exe -m tools.label_editor --import-patch C:\dosya\goruntu.review.json
```

İçe aktarma güncel görüntü/etiket hashlerini, kutuları ve oturum bölünmesini
kontrol eder. Önceki etiket ve inceleme kaydı `review_history` altında saklanır.
Hatalı veya eski JSON etiketleri değiştirmeden reddedilir. Sayfa onayları diske
doğrudan yazmaz; içe aktarmadan sonra yeni görüntü/etiket hashleriyle oluşturulur.

```powershell
& .venv/Scripts/python.exe -m tools.review_dataset
```

Komut `dataset_cctv_desk/review_previews` altında kutu ve sınıf adlarını çizer,
`review.json` içinde her görüntüyü başlangıçta `pending` tutar. Önizlemeyle birlikte
orijinali tam çözünürlükte kontrol edin; özellikle uzaktaki kalemleri inceleyin.
Yanlış/eksik kutuları `labels` dosyasında düzeltin, komutu tekrar çalıştırın.
İnceleyen kişi yalnızca kontrol ettiği kayıtları `status: approved`, kendi adıyla
`reviewer`, gerçek çekim/video oturumu için `session` ve
`all_target_objects_checked: true` ile imzalar. Kamu görüntülerinde oturum bilgisi
bilinmiyorsa aynı sahne/seri görüntüler birlikte gruplanmalı; bilgi belirsizliği
çözülmeden güvenilir kamera testi sayılmamalıdır.

Görüntü veya etiket değişirse SHA-256 imzası geçersiz olur. İmzaları elle güncelleyip
incelemeyi atlamayın. Araç kendiliğinden onay üretmez. Train/val arasında aynı
oturum ve aynı çözümlenmiş görüntü bulunamaz. Yakın kopyalar ve aynı sahnenin
farklı kareleri ayrıca gözle kontrol edilmelidir; piksel kontrolü bunları kapsamaz.

```powershell
& .venv/Scripts/python.exe run_pipeline.py --check
```

Eğitim ve `--check` aynı zorunlu insan incelemesi kontrolünü kullanır. Teknik
doğrulama testlerde ayrıca çalıştırılabilir; teknik doğrulamanın geçmesi anlamsal
etiket doğruluğunun kanıtı değildir. İndirme hataları eğitimden önce durdurulur.

## Model yardımıyla bütün verinin taranması

```powershell
& .venv/Scripts/python.exe -m tools.model_audit
```

Yerel YOLO-World ve CLIP ağırlıklarıyla CUDA üzerinde çalışır. Ağırlık indirmez.
Görüntü ve etiket hashleriyle kesinti sonrasında kaldığı yerden sürer.
`model_audit_summary.json` olası eksik etiket, sınıf uyuşmazlığı ve kutu sınırı
adaylarını sıralar. `model_audit_status.json` gerçek ilerlemeyi gösterir.
Sonuçlar düzenleyicide kesikli kutular olarak görünür; doğrulanmadan eklenmez.

Öğretmen model duvar fotoğraflarını gerçek insan sanabilir veya gerçek nesneleri
kaçırabilir. Dolayısıyla aday sayısı etiket hata sayısı değildir. Kaynak etiketlerini
otomatik değiştirmez ve hiçbir görüntüyü otomatik onaylamaz. Görsel olarak kontrol
edilmiş kısmi düzeltmeler `needs_correction` kalır; eksiksiz inceleme ayrı adımdır.

## Kamera kabul ölçümleri

`python -m tools.dataset_quality` sınıf başına görüntü/kutu sayılarını, piksel
alanına göre küçük/orta/büyük kutuları ve bölümler arasında aynı dHash değerine
sahip veya en fazla 4 bit farkı olan yakın kopya adaylarını `quality_report.json` dosyasına yazar. dHash yakınlığı
aynı sahne olduğunun kanıtı değildir; adayları görsel olarak inceleyin. Bu rapor
kutulara kendiliğinden onay vermez.

Kamu verisiyle öğrenme, kamera başarısının kanıtı değildir. Ayrı tutulmuş kamera
oturumlarında gündüz/gece, yansıma, kapanma, boş masa ve 2,5 metrede kalem örnekleri
etiketlenmelidir. Her sınıf için precision/recall, karışıklık matrisi ve boş sahne
yanlış alarm oranı raporlanmadan model üretim için kabul edilmez. Başarı eşikleri
kamera kullanımına göre belirlenir; bu depo henüz bu kabulü tamamlamamıştır.

İncelenmiş, eğitimden ayrı kamera kareleri ve model tahminleriyle ölçüm:

```powershell
& .venv/Scripts/python.exe -m tools.evaluate_camera camera_predictions.json --output camera_metrics.json
```

Girdi `names` alanında sabit 10 sınıfı, `frames` alanında her kare için `image`,
`session`, `reviewed: true`, `held_out: true`, `truth` ve `predictions` alanlarını
içerir. `truth` kutuları YOLO biçiminde `[class_id,x,y,w,h]`; her tahmin
`{"box":[class_id,x,y,w,h],"confidence":0.9}` biçimindedir. Koordinatlar 0–1'e
normalize edilir. İncelenmiş boş karelerde `truth: []` kullanılır.

Rapor sınıf başına TP/FP/FN, precision/recall ve karışıklık matrisi üretir.
Tek nesne üzerindeki tekrarlanan tahminler false positive sayılır. Veri olmayan
sınıfa yapay başarı verilmez; metrik `null` olur. Boş karelerde yanlış alarm oranı
kare bazındadır, saat başına olay oranı değildir. Bu araç AP hesaplamaz; AP eğitim
doğrulamasındaki Ultralytics raporundadır. İncelendi/ayrı tutuldu alanlarının
gerçekliği kaynak veriden doğrulanmalıdır; JSON tek başına bunun kanıtı değildir.

Resmi kaynaklar:
- https://storage.googleapis.com/openimages/web/download_v7.html
- https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
- https://cocodataset.org/dataset/detection-2017.htm
