# Kamera sistemi icin kontrollu ogrenme

## Mevcut YOLO-World'den resmi kaynakla aday egitimi

`python -m tools.train_world_candidate` mevcut `yolov8m-worldv2.pt` dosyasindan
baslar, resmi COCO train2017 bolumunden 5000 goruntu ve val2017 bolumunden
ayri 500 goruntu hazirlar. 80 ozgun COCO sinifi ve her secilen goruntunun tum
COCO kutulari korunur. `tv` etiketi monitor diye degistirilmez; kalem bu kaynakta
etiketli degildir. Bu islem kamera icin onaylanmis 10 sinifli egitim degil,
resmi kaynak etiketlerine dayanan ayri bir YOLO-World deneyidir.

Varsayilan aday ayarlari: 20 epoch, 640 piksel, batch 8, ilk 10 model modulunu
dondurma, AdamW ve 0.0001 ilk ogrenme orani. Bu secimler eski kabiliyetlerin
korunmasini garanti etmez. Eski model degismez ve aday otomatik yayinlanmaz.
Kaynak agirligin SHA256 imzasi egitim oncesinde ve surec bitiminde kontrol edilir.
Her epoch sonunda ilerleme `cctv_desk_project/world_candidates/status.json`
dosyasina yazilir. Ayri tarihli run klasorunde checkpointler tutulur.

Bekleyen veri indirmesi dahil tum surec boyunca gecici Windows sistem uyaniklik
istegi kullanilir. Ekran kapanabilir ve Windows kilitlenebilir; bu istek otomatik
uykuyu engeller. Bilgisayari kapatma, elle uykuya alma ve guc kaybi egitimi durdurur.
Surec kapaninca uyaniklik istegi kaldirilir. Bu aday, kamera testinde ayni 11
YOLO-World promptuyla `benchmark_camera --world` uzerinden mevcut modelle
karsilastirilmalidir; kalem ve monitor testleri de gereklidir.
`python -m tools.world_validation_watch` aday egitimi bittikten sonra eski model
ve adayin ayni 500 resmi dogrulama goruntusundeki AP degerlerini olcer. Sonuc
`world_candidates/source_comparison.json` dosyasina yazilir. Sinif bazindaki
gerilemeler ayrica listelenir. Bu sonuc kamera kabulunun yerine gecmez ve
model otomatik devreye alinmaz. Egitim basarisizsa veya alti saatte bitmezse
dogrulama bekleyicisi GPU testi baslatmadan cikis yapar.

Mevcut canli model ve varsayilan canli calistirma yolu degistirilmedi.
Yeni modeller ayri aday olarak egitilir. Bu belgede planlanan entegrasyonlar,
kurulmus veya egitilmis model olarak kabul edilmemelidir.

## Calisan yeni altyapi

- `tools.active_learning`: guncel goruntu/etiket hashlerini kontrol ederek model
  uyusmazligi ve sinif kapsamina gore inceleme kuyrugu hazirlar. Etiketleri degistirmez.
  Kaynak/oturum basina kota uygular; oturum bilgisi yoksa split kotasi kullanilir.
- `run_pipeline.py --train-only`: TensorRT gerektirmeyen ayri aday egitimi.
  Insan incelemesi, acik bulgular, yakin kopya incelemesi ve negatif ornek kontrolleri zorunludur.
- `tools.benchmark_camera`: incelenmis, egitimden ayrilmis kamera goruntulerinde
  yerel YOLO veya YOLO-World agirliklarini olcer; goruntu ve model SHA256 kaydeder.
  Gecikme model tahmini ve son islemedir; RTSP, decode, ekran ve takip dahil degildir.
- `tools.compare_candidates`: ayni goruntuler, oturumlar ve gercek etiketlerde sinif
  basina TP azalmasini ve FP artisini reddeder. En az 20 pozitif/sinif, 20 negatif
  kare ve p95 gecikmede en fazla %5 artis varsayilan politika olarak kullanilir.
  En az bir algilama iyilesmesi gerekir. Bu esikler ilk politikadir, kalite garantisi degildir.
  Basarili sonuc yalnizca incelemeye uygunluk bildirir; canli modeli degistirmez.

## Komutlar

```powershell
.venv/Scripts/python.exe -m tools.active_learning --limit 100 --per-session 50
.venv/Scripts/python.exe -m tools.review_server
.venv/Scripts/python.exe run_pipeline.py --check
.venv/Scripts/python.exe run_pipeline.py --train-only --name candidate_yolo11m_v1
.venv/Scripts/python.exe -m tools.training_watch --wait
```

Egitim once tum etiketlerin ve bulgularin incelemesini tamamlamayi gerektirir.
Model tahminleri kendi kendine gercek etiket ilan edilmez.
`training_watch --wait` her 30 saniyede inceleme durumunu kontrol eder. Onaylar
tamamlaninca tum kalite kontrollerini yeniden uygular ve tek bir aday egitimini
tarih/saat ile ayrilan klasorde baslatir. Durum `cctv_desk_project/training_queue_status.json`
dosyasina yazilir; ayni anda ikinci bir bekleyici dosya kilidi ile engellenir.

Kamera test JSON dosyasi `names` sabit sinif listesini ve `frames` listesini icerir.
Her karede `image` (JSON klasoru icindeki goreli yol), `session`,
`reviewed: true`, `held_out: true`, `truth` bulunur.
`truth` kutulari `[sinif_id, merkez_x, merkez_y, genislik, yukseklik]` biciminde normalize edilir.
Bos sahneler `truth: []` ile acikca isaretlenir. Test oturumlari egitimde kullanilmaz.

```powershell
.venv/Scripts/python.exe -m tools.benchmark_camera camera-evaluation/frames.json --world --weights yolov8m-worldv2.pt --output camera-evaluation/baseline.json
.venv/Scripts/python.exe -m tools.benchmark_camera camera-evaluation/frames.json --weights cctv_desk_project/candidate_yolo11m_v1/weights/best.pt --output camera-evaluation/candidate.json
.venv/Scripts/python.exe -m tools.compare_candidates camera-evaluation/baseline.json camera-evaluation/candidate.json --output camera-evaluation/comparison.json
```

Gercek kamera test verisi olmadan skor uydurulmaz. Kisa ornek testi uzun sureli
yanlis alarm/saat ve takip kimligi testinin yerini tutmaz. Yeni egitimde eski
onayli ornekler de korunur; her surum ayni sabit test setiyle karsilastirilir.

## Arastirilan 15 gelistirme ve durum

| Gelistirme | Resmi kaynak | Durum / sonraki kabul kosulu |
|---|---|---|
| Kameraya ozel YOLO egitimi | https://github.com/ultralytics/ultralytics | Ayri egitim yolu hazir; veri incelemesi bekleniyor |
| YOLOE ile acik sozluk/gorsel ornek | https://github.com/THU-MIG/yoloe | Kurulmadi; ayri ortamda test ve sinif esleme gerekli |
| RF-DETR alternatif aday | https://github.com/roboflow/rf-detr | Kurulmadi; egitim, adapter ve ayni kamera testi gerekli |
| SAHI kucuk nesne taramasi | https://github.com/obss/sahi | Kurulmadi; bolge birlestirme ve gecikme testi gerekli |
| Grounding DINO etiket denetimi | https://github.com/IDEA-Research/GroundingDINO | Kurulmadi; mevcut YOLO-World denetimine ikinci gorus adayi |
| SAM2 maskeleri | https://github.com/facebookresearch/sam2 | Kurulmadi; once cevrimdisi etiket yardimi |
| SAM3 kavram/video maskeleri | https://github.com/facebookresearch/sam3 | Kurulmadi; agirlik erisimi ve VRAM testi gerekli |
| BoxMOT takip karsilastirmasi | https://github.com/mikel-brostrom/boxmot | Kurulmadi; mevcut ByteTrack korunuyor; IDF1/HOTA testi gerekli |
| DINOv3 gorsel cesitlilik | https://github.com/facebookresearch/dinov3 | Kurulmadi; agirlik erisimi ve ozellik adapteri gerekli |
| CVAT etiket kalite sureci | https://github.com/cvat-ai/cvat | Kurulmadi; mevcut yerel editor ve imzali inceleme kullaniliyor |
| FiftyOne aktif ogrenme | https://github.com/voxel51/fiftyone | Kurulmadi; yerel hash kontrollu kuyruk eklendi; embedding entegrasyonu sonraki adim |
| Avalanche surekli ogrenme | https://github.com/ContinualAI/avalanche | Kurulmadi; detection adapteri ve unutma testleri gerekli |
| MMPose durus/eller | https://github.com/open-mmlab/mmpose | Kurulmadi; yeni gorev icin ayri etiketli test gerekli |
| Anomalib normalden sapma | https://github.com/open-edge-platform/anomalib | Kurulmadi; normal/anormal kamera ornekleri gerekli |
| TensorRT ve MLflow | https://github.com/NVIDIA/TensorRT ; https://github.com/mlflow/mlflow | Export yolu mevcut; MLflow kurulmus degil; kamera kabul kapisi eklendi |

Kaynaklar 2026-10-04 tarihinde resmi depolardan arastirildi. Bu bileşenler tek bir
model degildir; farkli egitim, etiketleme, takip ve calistirma gorevleri vardir.
Mevcut 8 GB GPU'da agir modellerin hepsini ayni anda calistirma varsayimi yapilmaz.
Yeni bagimliliklar mevcut calisan ortam yerine ayri deney ortaminda denenmelidir.
Lisanslar model agirligi ve varyant bazinda kontrol edilir.
