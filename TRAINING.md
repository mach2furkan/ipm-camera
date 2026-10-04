# CCTV desk eğitimi

Zorunlu etiket incelemesi ve sınıf kapsamı: [LABELING.md](LABELING.md).
Eğitim ve `--check`, her görüntü için güncel görüntü/etiket hashleriyle insan
inceleme kaydı ister. Kamu kaynakları kendiliğinden onaylanmaz.

Veri hazırlama varsayılan olarak gerçek `Computer monitor` etiketlerini kullanan
Open Images kaynağını seçer. COCO karışımı isteğe bağlıdır:
`python -m tools.prepare_desk_data --source mixed`. Varsayılan seçimin 15.000
görüntü sağladığı iddia edilmez; gerçek sayılar kalite raporunda yer alır.
Open Images seçimi doğrulanmış boş kamera sahneleri sağlamaz; bu sahneler
ayrıca hazırlanıp incelenmeden eğitim kapısı açılmaz.

Otonom hat veri hazırlamadan sonra model yardımıyla etiket taramasını çalıştırır;
insan incelemesi tamamlanmamışsa durumu `awaiting_label_review` olarak kaydeder.
Canlı tespitte pen/pencil aynı sabit sınıfa çevrilip tekrar kutuları birleştirilir.
Canlı takip ilk karedeki tek bir tahminle kesinleşmez; iki ardışık eşleşme ister.
`--max-age-ms` varsayılan 250 ms ile eski kareler çıkarılır; yeniden bağlantıda
takip sıfırlanır. Bu davranışlar saha başarısının veya yanlış alarmın sıfır
olduğunun kanıtı değildir; gerçek kamera kabul testi hâlâ gereklidir.

Bu hat hazır, tamamı gözden geçirilmiş YOLO etiketleri gerektirir; veri indirme veya
etiket oluşturma işlemini yapmaz. COCO category ID eşlemesi sırasıyla
1, 84, 73, 72 (tv: monitor için yalnızca bir vekil), 76, 74, 77, 47, 44'tür.
Kalem kaynağı için Open Images sınıf açıklaması yanında kutu etiketlerinin gerçekten
mevcut olduğunu doğrulayın; görüntü düzeyi etiketlerden kutu üretmeyin.
Bir görüntüdeki tüm hedef sınıfları etiketleyin. Kamera boş sahnelerini elle kontrol
edin; boş etiket dosyaları oluşturun. Aynı kamera/video oturumunu iki bölüme ayırmayın.

Veri dizinleri: `dataset_cctv_desk/images/{train,val}` ve
`dataset_cctv_desk/labels/{train,val}`. Her iki bölümde tüm sınıflar ve hard-negative
örnekler bulunmalıdır. Kaynak çözünürlükte alanı 16 pikselden küçük kutular otomatik
silinmez; inceleme için doğrulama durur. Toplam verilen sayılar 15.000'dir;
hard-negative görüntüler train/val bölümlerine dahildir.

```powershell
& ./.venv/Scripts/python.exe run_pipeline.py --check
& ./.venv/Scripts/python.exe run_pipeline.py --benchmark-image kamera_ornegi.jpg
```

Eğitim öncesi CUDA uyumlu TensorRT ve ONNX dışa aktarım bağımlılıkları kurulmalıdır.
Hat TensorRT eksikse eğitimden önce durur. Otomatik batch, 8 GB belleğin %70'ini
hedefler; bellek tüketimi garantisi değildir. Eğitime dört saat ayrılır, dışa aktarım
ve doğrulama ayrıca zaman alır. Veri hazırlama dahil toplam 4,5 saat garantisi yoktur.
Windows'ta ana işlem koruması vardır; `nohup` veya Linux `kill` kullanmayın.

Başarılı eğitim ardından batch=1, 960x960, FP16, statik şekil ve NMS ile engine
üretilir. `deployment_report.json` mAP50, sınıf metrikleri ve 20 ısınma sonrası
100 tahminin inference p50 ve uçtan uca predict p50/p95 ölçümlerini içerir.
Sentetik görüntü ölçümü kamera kabul testi sayılmaz. Engine GPU/TensorRT ortamına
özgüdür. Boş kamera sahnesi testi, 2,5 metre kalem testi ve hedef gecikme sağlanmadan
üretime kabul etmeyin. Geometri filtresi sahte alarmları sıfırlama garantisi vermez.

```powershell
& ./.venv/Scripts/python.exe -m tools.live_detect --closed-model --model cctv_desk_project/yolo11m_cctv_run/weights/best.engine --imgsz 960
```

Kamera parolası mevcut `HIK_PASS` ortam değişkeninden okunur.

## Otonom çalışma

`python -m tools.autonomous_desk` veri indirme/hazırlama, veri denetimi ve eğitim
hattını sırasıyla yürütür. `autonomous_status.json` aşamayı, `training-*.log`
dosyaları ilerlemeyi gösterir. Veriler ve kaynak etiketler proje altında tutulur.
Open Images kaynağında 1.011 train, 44 validation kalem görüntüsü doğrulandı;
masaüstü örnekleri eklenir, aynı görüntü çoğaltılarak sayı şişirilmez.

Bu kamu verisi başlangıç eğitimidir. COCO'daki tv etiketi monitor için vekildir;
COCO kalemleri etiketlemez; hedef kutusu olmayan COCO görüntüleri doğrulanmış
kamera hard-negative örneği değildir. Bu eksikler raporda belirtilir. Gerçek
kamera etiketleriyle ek ince ayar ve ayrı kabul testi yapılmadan üretim başarısı
iddia edilmez.

Windows ekranını **Win+L ile kilitleyin**. Oturumu kapatmak, yeniden başlatmak
veya bilgisayarı kapatmak işlemi durdurur. Çalışma süresince uykuya geçişi
engelleyen süreç kapsamlı istek kullanılır; güç planı kalıcı değiştirilmez.
Eğitim kesilirse `python run_pipeline.py --resume <last.pt_yolu>` ile devam edilir.
Veri indirme tekrar başlatıldığında mevcut dosyalar kullanılır.
