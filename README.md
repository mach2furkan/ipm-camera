# ipcam — Hikvision Akış ve Donanım Entegrasyon Katmanı (Faz 1)

Hikvision IP kameralar için düşük gecikmeli RTSP veri düzlemi ve ISAPI kontrol düzlemi.
Çıkarım motorunu her zaman **en güncel, bozulmamış** kareyle besler; ağ kopmalarında,
donmalarda ve kod çözücü hatalarında kendini toparlar; kameranın röle, hoparlör ve olay
akışını tek bir asenkron arayüz arkasında toplar.

```
                      ┌──────────────────────── CameraNode (asyncio facade) ────────────────────────┐
                      │                                                                              │
 RTSP 102 (TCP) ──►  RTSPStreamReader ──► keyframe gate ──► HW decoder ──► LatestSlot(1) ──► get_latest_frame()
   nobuffer           ▲  packet taps                    (NVDEC/VAAPI/QSV/   drop-oldest       FrameResult + latency
                      │                                   D3D11VA/SW)
                StreamWatchdog  (3 s heartbeat, backoff 1-2-4-8-15 s, zombie quarantine)

 RTSP 101 (TCP) ──►  RTSPStreamReader (packet-only) ──► PacketRing (GOP-aligned pre-roll)
   lease-based          └─ decode lease (SAHI)           └─► EvidenceRecorder (remux, bit-exact MKV/MP4)

 ISAPI (HTTP Digest) ─► HikvisionISAPIClient ── ircutFilter ─► IRStateResolver (ISAPI + chroma) ─► IR ağırlıkları
                                             ├─ IO/outputs/N/trigger (retriggerable röle darbesi)
                                             ├─ audio play / two-way audio (G.711)
                                             └─ alertStream ─► AlertStreamListener ─► EventBus (asyncio, drop-oldest)
```

## Kurulum

```bash
uv venv && uv pip install -e ".[dev]"          # çekirdek + testler
uv pip install -e ".[display]"                 # OpenCV pencereli smoke test
uv pip install -e ".[nvdec]"                   # NVDEC zero-copy (PyNvVideoCodec + torch)
```

Gereksinimler: Python ≥ 3.10, PyAV ≥ 12 (donanımsal hızlandırma için PyAV ≥ 14), httpx, numpy.

## Hızlı başlangıç

```python
import asyncio
from ipcam import CameraConfig, CameraNode

async def main() -> None:
    cfg = CameraConfig(host="192.168.1.64", username="admin", password="***")
    async with CameraNode(cfg) as node:
        while True:
            res = node.latest_frame(max_age_ms=300)       # en yeni kare veya None
            if res is not None:
                weights = "ir" if node.illumination.use_ir_weights else "rgb"
                detections = model[weights](res.image)    # numpy BGR (veya CUDA tensör)
                print(f"seq={res.seq} gecikme={res.latency_ms:.1f} ms atlanan={res.skipped}")
                if violates(detections):
                    await node.respond_to_violation(relay_output=1, relay_s=5, audio_id=2, record_s=15)
            await asyncio.sleep(1 / 30)

asyncio.run(main())
```

Bileşenler bağımsız da kullanılabilir:

```python
from ipcam import CameraConfig, RTSPStreamReader, StreamWatchdog, StreamRole

reader = RTSPStreamReader(CameraConfig.from_env(), StreamRole.SUB)
with StreamWatchdog(reader):
    seq = 0
    while True:
        res = reader.wait_frame(seq, timeout=1.0)        # blocking, newest only
        if res:
            seq = res.seq
            infer(res.image)
```

## Doğrulama / benchmark betiği

```bash
python -m tools.main_pipeline_test --ip 192.168.1.64 --user admin --password '***' --display
HIK_IP=192.168.1.64 HIK_PASS=*** python -m tools.main_pipeline_test --duration 120 --max-p95-ms 150 --json report.json
```

ISAPI üzerinden cihaz bilgisini, alt akış kodlayıcı ayarlarını ve IR-cut durumunu okur; alt
akıştan 30 FPS sabit hızla kare çeker; her karede gecikmeyi ağ / decode / dönüşüm / kuyruk
kırılımıyla ekrana basar; sonunda p50/p95/p99 raporu verir ve kapanışta hiçbir native
handle'ın sızmadığını doğrular. Çıkış kodları: `0` başarılı, `1` kare yok, `2` gecikme
bütçesi aşıldı, `3` kaynak sızıntısı.

Testler (kamera gerektirmez; sentetik video ile uçtan uca):

```bash
.venv/Scripts/python -m pytest -q
```

## Tasarım kararları

### Veri düzlemi
| Gereksinim | Uygulama |
|---|---|
| RTSP over TCP | `rtsp_transport=tcp`; UDP kaynaklı makroblok/smear yok. |
| < 100 ms ek gecikme | `fflags=nobuffer+discardcorrupt`, `flags=low_delay`, `max_delay=500000`, `reorder_queue_size=0`, `probesize=32768`, `analyzeduration=500000`; `allowed_media_types=video` ile ses kanalı SETUP'ı atlanır. Kod çözücü **slice threading** ile çalışır (frame threading N-1 kare gecikme ekler). |
| FFmpeg sürüm farkı | FFmpeg ≥ 5'te soket zaman aşımı `timeout`, FFmpeg 4'te `stimeout`. FFmpeg 4'te `timeout` vermek demuxer'ı sessizce *dinleme moduna* sokar; sürüm otomatik algılanır. |
| Drop-oldest | `LatestSlot` boyut-1 üzerine yazmalı tampon; tüketici geride kalırsa birikme yerine en yeni kareye atlar. `FrameResult.skipped` kaç karenin atlandığını söyler. |
| Anahtar kare | Bağlantı/yeniden bağlantı, bozuk paket veya decode hatası sonrası bir sonraki IDR'a kadar hiçbir kare yayınlanmaz (gri/bozuk kare modele asla ulaşmaz). Her bağlantıda ISAPI `requestKeyFrame` çağrılarak GOP beklemesi ortadan kaldırılır. |
| Tembel dönüşüm | YUV→BGR (ve hwaccel'de GPU→host indirmesi) yalnızca kare okunduğunda yapılır; üzerine yazılan kareler hiç dönüştürülmez. |
| Donanımsal decode | `HWAccel.AUTO`: CUDA → VA-API/D3D11VA → QSV → yazılım. `decoder_name` ile `h264_cuvid`, `h264_v4l2m2m`, `h264_nvmpi` vb. zorlanabilir. Jetson / VA plugin için `GStreamerSession` + `build_gstreamer_pipeline`. |
| Zero-copy | `HWAccel.NVDEC`: PyAV yalnızca demux eder, paketler PyNvVideoCodec'e gider, NV12→RGB dönüşümü GPU'da yapılır; kare `cuda:N` üzerinde `torch.Tensor (3,H,W)` olarak gelir, host RAM'e hiç inmez. |
| Ana akış | Lease tabanlı: SAHI decode lease'i alır, delil kaydı yalnızca paket lease'i alır (remux, decode yok). Son lease bitince oturum `main_idle_linger_s` sonra kapanır. `main.preroll_s > 0` ise ana akış paket modunda açık kalır ve GOP hizalı halka tampon olay öncesi görüntüyü tutar. |
| Anında açılış | Ana akışta decode açıldığında önbellekteki GOP yeniden çözülür; ilk kare bir sonraki IDR beklenmeden gelir. |

### Direnç
* **Watchdog**: her oturum yeni bir worker thread'de çalışır. 3 s paket/kare gelmezse oturum
  zorla kapatılır (FFmpeg soket zaman aşımı bloklu okumayı 3 s ile sınırlar) ve 1-2-4-8-15 s
  (±%10 jitter) ile yeniden bağlanılır. Geri sayım, oturum 10 s sağlıklı kalmadan sıfırlanmaz.
* **İlk kare toleransı**: H.265+/Smart Codec GOP'ları 4-8 s olabildiği için ilk IDR beklenirken
  kare zaman aşımı `keyframe_wait_s` (10 s) olarak uygulanır; paket akışı yine 3 s ile izlenir.
* **Zombie karantinası**: teardown sonrası çıkmayan worker (sürücü kilitlenmesi vb.) beklenmez;
  jenerasyonu iptal edilir, bir daha slot'a kare yazamaz, yeni oturum hemen başlar.
* **Kimlik doğrulama hataları**: Hikvision birkaç hatalı denemeden sonra istemci IP'sini kilitler.
  RTSP 401'de en az 60 s, alertStream'de 30 s→5 dk beklenir; Digest katmanı geçerli nonce ile
  reddedilen kimlik bilgisini tekrar denemez.
* **Kaynak yönetimi (RAII)**: tüm `InputContainer`, `CodecContext`, `VideoCapture` ve NVDEC
  nesneleri `NativeHandle` ile açılır, deterministik kapanır ve süreç geneli kayıt defterinde
  sayılır (`ipcam.live_handles()`). Kimlik bilgisi içeren FFmpeg hata mesajları loglara
  maskelenerek yazılır, istisna zincirine eklenmez.

### Kontrol düzlemi
* **Digest Auth (RFC 7616/2617)**: MD5, SHA-256, SHA-512-256, `-sess`, `auth-int`, `userhash`,
  `stale=true`. Önbelleğe alınan nonce ve artan `nc` ile ön-yetkilendirme: ilk challenge'dan
  sonra her ISAPI çağrısı tek tur.
* **Röle**: `pulse_alarm_output` yeniden tetiklenebilir monostable'dır; aktif darbe sırasında gelen
  ihlal röleyi aç/kapa yapmak yerine süreyi uzatır. İptalde ve `aclose()`'da röle mutlaka
  bırakılır (tekrar denemeli).
* **Ses**: `play_audio` önce `/ISAPI/System/Audio/channels/<ch>/play`, desteklenmiyorsa AcuSense
  `AudioAlarm` uç noktasını dener ve çalışanı önbelleğe alır. `play_wav` / `stream_pcm` iki yönlü
  ses kanalı üzerinden herhangi bir PCM'i G.711 µ-law/A-law'a çevirip gerçek zamanlı gönderir.
* **alertStream**: multipart (Content-Length'li/siz) ve çerçevesiz XML desteklenir. Kameranın
  ~10 s'lik `videoloss/inactive` kalp atışı sayesinde 35 s sessizlik ölü TCP olarak algılanır.
  Tekrarlanan `active` bildirimleri START/UPDATE/END yaşam döngüsüne indirgenir; bağlantı
  koptuğunda açık olaylar END ile kapatılır.
* **IR durumu**: `ircutFilter` `auto` modunda anlık durumu bildirmez. `IRStateResolver`,
  `day`/`night` zorlanmışsa ISAPI'ye, aksi halde alt akış karelerinin kroma enerjisine (IR'da
  U/V≈128) histerezis ve bekleme süresiyle karar verir.

## Faz 5 — Çoklu nesne takibi ve mekânsal analitik (`ipcam.analytics`)

```python
from ipcam.analytics import AnalyticsEngine, build_rules
from ipcam.analytics.actions import ActionSpec, EventActionDispatcher

engine = AnalyticsEngine(rules=build_rules("examples/zones.json", width=704, height=576))
dispatcher = EventActionDispatcher(node, {"ana-kapi": ActionSpec(relay_output=1, relay_s=2, audio_id=1)},
                                   loop=asyncio.get_running_loop())

res = node.latest_frame()
dets = model(res.image)                               # (N, 5|6): x1, y1, x2, y2, conf[, cls]
out = engine.update(dets, t=res.frame.arrival_ns / 1e9)
dispatcher.dispatch(out.events)                       # bloklamaz; röle/ses/kayıt asyncio'da
```

| Bileşen | Uygulama |
|---|---|
| Kalman | SORT durum uzayı `[u, v, s, r, u', v', s']`, hızlar **px/s**. Her tahmin gerçek `dt` ile yapılır: Faz 1 kare düşürdüğü için kare sayısına dayalı filtre boşluklarda hareketi eksik tahmin eder. Gürültü kutu yüksekliğiyle ölçeklenir (15 px ile 300 px hedef aynı göreli belirsizlikte); Joseph formu ile kovaryans simetrik kalır. Tüm izler `einsum` ile toplu tahmin edilir. |
| ByteTrack | Aşama 1: izlenen + kayıp izler × yüksek skorlu (≥ 0.6) tespitler. Aşama 2: hâlâ izlenenler × düşük skorlu (0.15–0.6) tespitler — ağaç/direk arkasındaki insanın izi kopmaz. Aşama 3: geçici izler. Aşama 4: ≥ 0.7 skorla yeni iz. Kayıp izler 1.5 s yeniden tanınmak için tutulur. |
| CCTV eklemeleri | **Buffered IoU (C-BIoU)**: küçük/hızlı ya da kare atlanmış hedefte düz IoU 0 olsa da eşleşme sürer. **Mahalanobis kapısı**: kesişen yayalarda geometrik olarak imkânsız eşleşmeleri reddeder. Sınıf farkında eşleştirme; saat geri giderse (akış yeniden başladı) güvenli sıfırlama. |
| Tripwire | Çapraz çarpım işaretine **histerezis bandı** eklenir (çizgi üzerinde duran kişi alarm yağdırmaz) ve hareket **doğru parçasını** kesmelidir (tel ucunun etrafından dolaşmak geçiş değildir). Yön, operatörün verdiği `inside` referans noktasıyla belirlenir. Henüz olgunlaşmamış bir izin geçişi bekletilir, iz doğrulanınca raporlanır. |
| Poligon | Vektörel ray-casting (içbükey poligonlar, köşeden geçen ışın tek sayılır), önce bbox ile eleme. Referans nokta **ayak tabanı**. İhlal `min_dwell_s` sonrası, çıkış `exit_grace_s` sonrası. Kapanma (lost) süresince bölge durumu sıfırlanmaz, dondurulur. |
| Aylak dolaşma | Bölgede ≥ 15 s **ve** son 15 s'deki ayak noktaları ağırlık merkezinden ≤ 30 px. |
| Heuristik filtre | Ham tespitlere değil **izlere** uygulanır (aksi halde ByteTrack'in 2. aşaması aç kalır). Medyan H/W < 0.8 (kedi, köpek, far), medyan yükseklik < 8 px, 80. persentil hız > 6 vücut boyu/s (kuş, parlama). Tepeden bakan kameralar için `min_aspect=None`. |
| Eylemler | `EventActionDispatcher` kural başına cooldown uygular, röleyi ayrı ve ilk görev olarak tetikler, olaydan röleye gecikmeyi ölçer (`latency_ms`, KPI < 200 ms). |

Doğrulama (kamerasız): sentetik CCTV sahnesi — 8 kişi, 18–140 px boy aralığı, %8 kaçırma,
%10 düşük skor (kapanma), sahte tespitler ve 40–120 ms düzensiz kare aralıkları — üzerinde
3 farklı tohumla **MOTA > 0.85** ve ≤ 4 ID değişimi doğrulanır; 40 kişilik sahnede takipçi
kare başına < 25 ms.

## Faz 7 — Çoklu kamera füzyonu, Re-ID, PTZ angajmanı, C2 (`ipcam.fusion`)

```
edge (kamera başına)                         merkez
AnalyticsResult ─► EdgePublisher ──protobuf──► bus ─► FusionService ─► GlobalTrackManager ─► C2Server (SSE)
  homografi + kovaryans, Re-ID kapısı            events.camera.<id>         │                     └─► tarayıcı haritası
                                                 alerts.camera.<id> ────────┴─► SlewToCueEngagement ─► ISAPI PTZ
```

Kamerasız canlı demo (simüle saha → üretim kodu): `python -m tools.fusion_demo` → http://127.0.0.1:8080

| Bileşen | Uygulama |
|---|---|
| Homografi | Normalize DLT + RANSAC; birinci dereceden belirsizlik yayılımı (`J Σ Jᵀ`) ile her ayak noktası için 2×2 metrik kovaryans; ufuk koruması. Kalibrasyon **işaretleyicilerin dışbükey zarfını** saklar: zarf dışındaki ölçümlerde edge kovaryansı 3× şişirir. (Yalnızca yakın alanda işaretleyiciyle yapılan kalibrasyon 3 cm RMSE gösterip 30 m'de ~2 m hata verebiliyor; test bunu yeniden üretip yakalıyor.) |
| Metrik füzyon | `[X, Y, VX, VY]` Kalman; her kameranın ölçümü kendi kovaryansıyla sırayla işlenir, yani yakın/net kamera otomatik olarak daha ağır basar. Örtüşen FOV'da 0.8 m + χ² kapısı ile birleştirme; yan yana duran iki farklı kişiyi Re-ID vetosu ayırır. |
| Re-ID | Seçici çıkarım: ≥ 64×128 px, Laplacian varyansı ≥ 120, önündeki kişilerce kapanma < %25, iz başına ≤ 1 kare/s, kadraj kenarına değen kutular hariç. `OnnxReIDExtractor` (OSNet-AIN/FastReID ONNX; TensorRT → CUDA → CPU). Galeri prototipi **eleman bazlı medyan** (kirli tek kırpıntıya dayanıklı). |
| Vektör arama | `FlatIndex` (tam sonuç, O(1) silme/güncelleme) veya FAISS `HNSWIndex` (silme işaretleme + yeniden kurma). |
| Devir (handover) | Kayıp varlıklar arasında arama: topoloji kapısı (`t_min ≤ Δt ≤ t_max`) + Gaussian geçiş olasılığı (3σ dışı reddedilir, içeride `d_eff = D_C + 0.1·(1−LR)`) + **kinematik kapı** (düz mesafe / Δt > 3.5 m/s ise imkânsız). D_C < 0.28 hemen; 0.28–0.40 arası 3 vektörün medyanıyla karar. Yeni kişi haritada hemen *geçici* kimlikle görünür, eşleşince eski ID'ye birleştirilir. Onaylanan devirler topoloji modelini çevrimiçi günceller. |
| PTZ | Yönelim/zoom geometrisi; hareket süresi kadar ileriyi hedefleyen **lead** nişan; açı uzayında PD + hedef açısal hız ileri beslemesi (zoom'dan bağımsız kararlılık). Durum makinesi: SLEWING → ACQUIRING → LOCKED. Kilitliyken sabit kameraların göremediği kör bölgede de görsel takip sürer. ISAPI: `ptz_absolute` (AbsoluteHigh, 0.1° birim), `ptz_continuous`, `get_ptz_status`. |
| Taşıma | `proto/surveillance/v1/event.proto` (spec alanları 1–7 + geriye uyumlu 8–10, kare başına `FrameBatch`). Bağımlılıksız proto3 codec'i; `google.protobuf` ile bayt düzeyinde uyumluluk testli; gömülü vektör `np.frombuffer` ile kopyasız çözülür. `InProcessBus` / `NatsJetStreamBus`. |
| C2 | Bağımlılıksız asyncio HTTP + SSE; canvas haritada kamera FOV izdüşümleri, PTZ nişan konisi, hedef rotaları, konum belirsizlik halkası, seçili hedef detayı (ilk görülme, hız, kamera geçmişi, birleşen ID'ler), olay akışı. |

### Doğrulama ve KPI durumu

| KPI | Hedef | Sonuç | Not |
|---|---|---|---|
| Homografi hatası, 30 m | ≤ 0.4 m | p95 ≤ 0.4 m (testli) | **Çözünürlüğe bağlı**: 1920 px'te 1 px ≈ 0.13 m, 1280 px'te ≈ 0.19 m zemin. 704×576 alt akış ayak noktasıyla bu KPI tutmaz; ana akış koordinatı veya ölçeklenmiş kutu kullanılmalı. |
| Kameralar arası kimlik | Rank-1 ≥ %88 | Simülasyonda 20 tohum: **0 yanlış birleştirme**, devir duyarlılığı **191/193**, parçalanma 1.02 | Sentetik vektörlerle. Gerçek Rank-1, gerçek OSNet modeli ve sahadan etiketli verinin ölçümüdür. |
| Vektör arama, 10 000 | ≤ 1.2 ms | Flat < 5 ms (testli eşik), HNSW çok daha hızlı | Makineye bağlı; ölçüm testte. |
| Slew-to-cue | ≤ 450 ms | Simülasyonda ilk kilit ≤ 450 ms (testli); demoda 282–321 ms, uzun dönüşte 607 ms | Gerçek süre kafanın preset hızına bağlı (200°/s varsayıldı). |
| E2E edge → harita | ≤ 25 ms | Süreç içi bus'ta birkaç ms | NATS üzerinden ağ ölçümü yapılmadı. |

## Termal PT serisi entegrasyonu (DS-2TD / HM-TD)

Kaynak: *ISAPI — Security thermal cameras, PT series* belgesi. İlk devreye alma için salt-okunur sonda:

```bash
python -m tools.pt_thermal_probe --ip 192.168.1.64 --user admin --password '***'
python -m tools.pt_thermal_probe --ip 192.168.1.64 --https --pin <sertifika SHA-256>
```

| Konu | Uygulama |
|---|---|
| Ad alanı | Bu nesil `http://www.isapi.org/ver20/XMLSchema` kullanıyor. İstemci ad alanını ilk yanıttan öğreniyor; istek gövdeleri cihazın konuştuğu ad alanında üretiliyor. Oku-değiştir-yaz işlemleri `ns0:` öneki üretmeden belgeyi koruyor. |
| Hata kodları | `statusCode/subStatusCode` yanında `errorCode/errorMsg` ayrıştırılıyor; sahada sık görülen kodlar için operatör ipucu (`ISAPIError.hint`). |
| HTTPS | Cihazlar kendi imzalı sertifikayla HTTPS açık geliyor; CA doğrulaması anlamsız olduğu için yaprak sertifikanın SHA-256 parmak izi sabitlenebiliyor (`tls_fingerprint_sha256`). |
| Keşif (`PTThermalCamera.discover`) | Optik/termal kanal **yoklanarak** bulunuyor (termal API'ler yalnızca termal kanalda yanıt veriyor). Sistem, PTZ, `absoluteEx` ve termal yetenek ağaçları `Capability` ile ayrıştırılıyor (`min/max/opt/def`, XML ve JSON). Desteklenmeyen çağrılar cihaza gitmeden yerelde `NotSupportedError`. |
| PTZ | `absoluteEx`: float derece (0.001°), zoom **oran veya odak uzaklığı (mm)**, hedef **mesafesine göre odak** (`objectDistance`), eğim sensörü (`lookDownUpAngle`); değerler cihazın yetenek aralıklarına kırpılıyor. `position3D` (0–255 normalize nokta/dikdörtgen, uzaklaştırma için Start.x > End.x), sürekli hareket (+ lens dönüşü), tek dokunuş odak, ev konumu, PTZ kilidi, yardımcı cihazlar (silecek/ışık), presetler. Destekleyen modellerde füzyondan gelen coğrafi hedef `objectInfo` ile kameranın kendi sürekli takibine devredilebiliyor. Angajman motoru için `ISAPIPTZEx` denetleyicisi hedefe dönerken menzile göre önceden odaklıyor (uzun termal lenslerde netlik alanı birkaç metre). |
| Termografi | Ölçüm modu (normal/expert/AI), `basicParam` oku-değiştir-yaz, gerçek zamanlı kural sıcaklıkları, `jpegPicWithAppendData` (JPEG + sıcaklık matrisi). |
| Yangın/duman | Yapılandırma `FireDetection` adlı **multipart form** birimiyle gönderiliyor (belgeye uygun; eski firmware'de düz XML'e geri düşüyor). |
| Metadata | Tür bazlı açma (`thermometry`, `fireDetection`, `shipsDetection`, `behaviorAnalysis`…) ve `subscribeType` ile metadata içeren RTSP URI. |
| Saat ofseti | Olay `dateTime` değerleri cihaz saatinden geliyor; `measure_clock_offset` NTP tarzı en düşük RTT örneğiyle ofseti ölçüyor, PT köprüsü olay zamanlarını buna göre düzeltiyor. |
| Olaylar | Çevre ihlali (`fielddetection`, `linedetection`, bölge giriş/çıkış, aylak) olaylarından `TargetRect`, hedef türü, **lazer menzil / hedef mesafesi**, hız, **olay anındaki PTZ pozu** (görünür ve termal), cihaz GNSS konumu; TMA/TMPA/TDA'dan kural tipi, eşik, anlık değer, preset. alertStream resimleri form adlarıyla (`visibleLightImage`, `thermalImage`, `targetImage`) olaya bağlanıyor. |

### Ham termal akış (RTSP, PT 109)

FFmpeg `thermalStream` yükünü çözemediği için kendi RTSP/1.0 istemcisi yazıldı (`ipcam/stream/rtsp_raw.py`):

- Bağlantı ve kimlik doğrulama: TCP interleaved, Digest/Basic, SDP'den iz seçimi, GET_PARAMETER ile oturumu canlı tutma.
- RTP ayrıştırma: CSRC, başlık uzantısı ve dolgu destekli.
- Kare birleştirme: kareler marker bitiyle birleştiriliyor. Sıra boşluğu olan kare **bütünüyle atılıyor**, böylece kaymış bir sıcaklık matrisi asla modele ulaşmıyor.

Yük düzeni belgede tanımlı değil. `ThermalPayloadDecoder` düzeni veriden bir kez çıkarıp kilitliyor; bunun için kesin boyut uyumu, fiziksel olarak makul sıcaklık dağılımı ve uzamsal düzgünlük birlikte aranıyor. Desteklenen biçimler:
- float32 °C veya uint16 sayım (deci-Kelvin / centi-Kelvin)
- herhangi bir başlık uzunluğu

İstenirse kesin düzen elle verilebiliyor.

### Radyometrik analiz ve görüntü

| Bileşen | Açıklama |
|---|---|
| `PlateauAGC` | Termal çekirdeklerin kullandığı plato histogram eşitleme + zamansal yumuşatma: büyük gökyüzü/zemin alanları gri seviyeleri tüketmiyor, 3 °C'lik küçük hedef görünür kalıyor (doğrusal germede yangın varken kayboluyor; testli). |
| Paletler | beyaz-sıcak, siyah-sıcak, demir, gökkuşağı, arktik; izoterm. PNG bağımlılıksız kodlanıyor. |
| `HotspotDetector` / `PersonCandidates` | Medyan + k·MAD uyarlamalı eşik; insan adayları görünür sıcaklık + geometri kapısıyla (gece/sis için dedektörden bağımsız ipucu). |
| `FireDetector` | Mutlak sıcaklık **veya** sıcaklık artış hızı; aynı konumda çoklu kare teyidi (alev titremesi ve su yansıması elenir). |
| `RoiMonitor` | Poligon bazlı maks/min/ort, ön alarm/alarm, histerezis, bekleme süresi, ve **en küçük kareler artış hızı** — mutlak eşik aşılmadan önce erken uyarı (testte önce "hızlı ısınma", sonra ön alarm, sonra alarm). |
| Bispektral | Optik→termal kayıt: sonsuzdaki homografi + **menzile bağlı paralaks** (`f·b/R`); `ThermalConfirmation` optik kutunun termal imzasını çevre halkasına göre z-skoruyla değerlendiriyor (gölge, far yansıması, poster ısı yaymaz). |

### Tek kameradan konum (`ipcam/fusion/geo.py`)

Bir PT olayındaki poz, kutu ve menzilden hedefin WGS-84 konumu hesaplanıyor:
- **Hesap yöntemi:** kesin ECEF↔ENU dönüşümleri kullanılıyor. Menzil yoksa düz arazi kesişimiyle konum bulunuyor.
- **Belirsizlik:** ışın boyunca menzil hatası, ışına dik yönde açı hatası olarak 2×2 kovaryansa yayılıyor.
- **Kuzey ofseti:** ≥ 2 nirengi noktasından dairesel ortalamayla kalibre ediliyor.
- **Küresel izleyiciye bağlantı:** `PTEventGeoBridge`, yinelenen olayları sözde yerel izlere bağlayıp küresel izleyiciye besliyor. Böylece PT kamera sabit kameralarla aynı hedefi birleştirebiliyor.

### Operatör konsolu

Konsol bilinçli olarak sahada kullanılan bir operatör ekranı gibi tasarlandı:
- **Görsel dil:** grafit zemin, ince çizgiler, köşesiz paneller. Tek vurgu rengi kehribar; kırmızı yalnızca alarmda. Rakamlar eşit genişlikli. Gradyan, gölge, emoji ve "AI" etiketi yok.
- **Harita:** tekerlekle yakınlaştırma ve sürükleyerek kaydırma; ölçek çubuğu, kuzey oku ve imleç koordinatı var. Etiket çakışması önleniyor. Kamera kapsaması etkin menzille kırpılıyor.
- **Termal görüntü:** imleç altında gerçek sıcaklık okunuyor; palet seçimi, ölçek, sıcak nokta/yangın/ROI katmanları var. Bağlantı koptuğunda son kare soluklaşıyor ve "bağlantı yok" yazıyor.
- **PTZ:** pusula kadranı, mutlak açı/zoom/odak bilgisi. Shift+tık ile haritadaki noktaya yönlendirme; durdurma, odak, ev konumu ve seçili hedefi izleme.
- **Diğer:** olay kaydı tablo biçiminde. `?still=1` ile rapor için durağan görüntü alınabiliyor.

Kontrol uç noktası, süreç başına üretilen bir belirteçle özel başlık (`X-C2-Token`) istiyor; başka bir kökenden gelen sayfa PTZ'yi süremez.

## Gecikme metriği nasıl okunmalı

`LatencyBreakdown` dört bileşen raporlar:

* `network_lag_ms` — paketin varış zamanı ile PTS arasındaki farkın kayan pencere minimumuna
  göre sapması: ağda, kamerada veya soket kuyruğunda biriken gecikme. Kamera ve sunucu saatleri
  senkron olmadığından mutlak glass-to-glass gecikmeyi değil, *sonradan eklenen* tamponlamayı ölçer.
* `decode_ms` — paketin demux edilmesinden resmin decoder'dan çıkmasına.
* `convert_ms` — tembel renk dönüşümü / GPU→host indirmesi.
* `queue_ms` — slot'ta bekleme süresi (drop-oldest sayesinde en fazla bir kare periyodu).

Mutlak glass-to-glass ölçümü için kamera OSD saatini (NTP senkron) ekrana çekip karşılaştırmak
tek güvenilir yöntemdir.

## Firmware notları

* Ses çalma ve `requestKeyFrame` uç noktaları ürün ailesine göre değişir; desteklenmeyen
  durumlarda `NotSupportedError` fırlatılır (`requestKeyFrame` hatası sessizce yutulur).
* `ircutFilter` yolu bazı firmware'lerde `IrcutFilter` olarak büyük harflidir; istemci her
  ikisini de dener.
* Kamera başına eşzamanlı RTSP oturum sınırı vardır (genelde 6-20). Ana akışın talep üzerine
  açılması bu bütçeyi korur.

## Modül yapısı

```
ipcam/
  config.py          CameraConfig, StreamProfile, LowLatencyOptions, DecoderOptions
  errors.py          hata hiyerarşisi + FailureKind sınıflandırması
  backoff.py         jitter'lı üstel geri çekilme
  metrics.py         percentiller, FPS, PTS-drift gecikme tahmini
  resources.py       NativeHandle (RAII) + sızıntı kayıt defteri
  bus.py             asyncio EventBus (abone başına drop-oldest)
  node.py            CameraNode: tüm katmanlar tek arayüzde
  isapi/             digest, client, alert_stream, multipart, models, audio, xmlutil
  stream/            reader, watchdog, session, decoders, frame, slot, recorder, manager
  vision/ir_state.py IRStateResolver
  analytics/         kalman, matching, bytetrack, geometry, rules, heuristics, engine, config, actions, render
  isapi/pt_thermal.py, isapi/events_pt.py   termal PT serisi API'si ve olay ayrıntıları
  stream/rtsp_raw.py RTSP/RTP istemcisi (thermalStream, isapi.metadata)
  thermal/           stream, render (AGC/palet/PNG), radiometry, bispectral, sim (sahne + sahte RTSP sunucusu)
  fusion/            homography, worldkf, reid, vector_index, topology, global_tracker, ptz, edge, service, sim,
                     geo (WGS-84/ENU, PT konum bulma), pt_bridge
    transport/       codec (proto3), bus (in-process / NATS JetStream)
    c2/              server, thermal_panel, static/index.html
proto/surveillance/v1/event.proto
examples/zones.json, examples/topology.json
tools/main_pipeline_test.py, tools/fusion_demo.py, tools/pt_thermal_probe.py
tests/
```
