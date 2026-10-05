# Kamera bağlantısı ve trafik sayımı

Windows'ta `start-camera.cmd` dosyasını açın veya `.venv\Scripts\python.exe -m tools.camera_app` çalıştırın.
Trafik seçeneği açık başlatmak için `takibirunet.cmd` dosyasına çift tıklayın veya PowerShell'de `.\takibirunet.cmd` çalıştırın.
IP/ağ adı, kullanıcı adı ve şifreyi başlangıç ekranına girin. Şifre maskelenir; diske veya alt süreç komut satırına yazılmaz.
Her pencere bir kameraya bağlanır; başka kamera için uygulamayı yeniden açabilirsiniz. Birden fazla pencere GPU yükünü artırır.

PowerShell komutu:
```powershell
Set-Location 'C:\Users\Mach2\Documents\GitHub\ipm camera'
& .\.venv\Scripts\python.exe -m tools.camera_app --traffic
```
Trafik seçeneği işaretli açılır; formdan kapatabilirsiniz.
Çıkarım tek bir arka plan iş parçacığında çalışır; aynı modele paralel çağrı yapılmaz ve kare kuyruğu biriktirilmez.
GPU çalışırken pencere olayları işlenir. Etiket önbelleği sınırlıdır; bilgi paneli saniyede dört kez yenilenir.
Kısa kare aralarında bekleme ekranı yanıp sönmez. Çözünürlük ve algılama eşikleri düşürülmez.
Görüntü FPS'i yine kamera ve GPU hızına bağlıdır; bu düzenleme belirli bir FPS garantisi vermez.

Markayı `Otomatik` seçerseniz Dahua ve Hikvision yolları görüntü alınarak kontrol edilir; başarısız olursa ONVIF Media1 üzerinden
yayın adresi aranır. ONVIF cihazda etkin olmalı ve ONVIF hesabı izleme yetkisine sahip olmalıdır; ONVIF web portunu formdan girebilirsiniz.
`Dahua` seçildiğinde ana yayın `/cam/realmonitor?channel=1&subtype=0`, alt yayın `subtype=1` biçiminde hazırlanır.
`Hikvision` seçildiğinde kanal ve main/sub yayınını seçin. Diğer RTSP kameralarında `Özel RTSP` ile cihazın belgelendirdiği yolu girin;
örneğin `/cam/realmonitor?channel=1&subtype=0` veya `/live/ch00_0`. RTSP portunun erişilebilir olması ve cihazda RTSP'nin açık olması gerekir.
Özel yol seçildiğinde main/sub seçimi yolu değiştirmez; kullanılacak yayını yol belirler.
Form önce çözülebilen bir video karesini doğrular; bağlantı hatasında form kapanmaz ve bilgiler düzeltilebilir.
Yanlış kullanıcı adı/şifre (401/403) sonrası başka yollarla giriş denenmez. Şifreler bağlantı hatasına eklenmez.
Bağlantı açılıyor fakat canlı görüntü çözülemiyorsa `Uyumlu görüntü çözme` seçeneğini deneyin; bu seçenek video çözümünü CPU'ya alır,
algılama modelini ve sınıfları değiştirmez. ONVIF devre dışı, yalnız Media2 destekli veya kendine özel protokollü cihazlarda otomatik yol
bulma başarısız olabilir; doğru özel RTSP yolu kullanılmalıdır. ONVIF kanal sırası bildirilen video kaynakları sırasıdır.

192.168.1.149 adresindeki Dahua kamera için:
```powershell
.\takibirunet.cmd --ip 192.168.1.149 --brand dahua
```

## Korunan algılama

Kalem/kurşun kalem sınıfı kullanıcı isteğiyle canlı uygulamada kapatılmıştır. Diğer 9 sınıfın ayarları korunur.
Modelin istem listesi değiştirilmez; kalem sonuçları takip ve çizimden önce süzülür.

Varsayılan model orijinal `yolov8m-worldv2.pt` dosyasıdır. Son eğitilen aday kullanılmaz.
İnsan, kitap, dizüstü bilgisayar, monitör, klavye, kalem/kurşun kalem, fare, telefon, bardak/kupa ve şişenin
11 İngilizce istemi, 10 sınıflı eşlemesi ve güven eşikleri korunur. Ağırlıklar değiştirilmez ve eğitim başlatılmaz.
Bu teknik koruma gerçek kameradaki doğruluğa ilişkin yüzde yüz garanti değildir; ışık, açı ve boyut algılamayı etkiler.

## İsteğe bağlı trafik

Formda trafik seçeneğini açın. Orijinal ağırlıkların ayrı bir YOLO-World örneği yalnız otomobil, kamyon,
otobüs, motosiklet ve bisiklet istemlerini kullanır; masa nesnelerinin modeline yeni istem eklenmez.
İki algılayıcı aynı ham kareyi işler. Trafik ek GPU belleği ve işlem süresi kullanır; varsayılan kapalıdır.

ByteTrack kimlikleriyle araçlar izlenir. Alt orta noktanın sonlu sayım çizgisini geçmesi sayılır.
Çizgi çevresindeki titreşim histerezisle bastırılır; her takip kimliği bir kez sayılır.
Sarı çizgide A ve B ile işaretlenmiş tarafa geçişler ayrı sayılır; ekranda araç türüne göre A/B toplamları görünür.
`L` ardından iki tıklama yeni çizgi ve yeni sayaç başlatır. `R` sayaçları sıfırlar.
Kamera yeniden bağlanınca konum geçmişi silinir, toplamlar tutulur. Çözünürlük değişince çizgi ölçeklenir.
Kimlik parçalanması, örtüşme ve uzun kopmalar sayım hatasına neden olabilir; sonuçlar doğrulanmış trafik istatistiği değildir.

`F`: tam ekran, `Q/Esc`: çıkış, `S`: görüntü kaydet, `+/-`: masa nesnelerinin güven eşiklerini ayarla.
`C`: mevcut bağlantıyı kapatıp bağlantı formuna dön. IP, marka, kanal ve özel yol aynı süreçte korunur;
şifre tekrar girilir ve diske kaydedilmez. Bağlantı kontrolünde `İptal / kapat` devam eden yol aramasını durdurur;
devam eden tek ağ isteği kendi zaman aşımına kadar sürebilir. Başlatma hatalarında yeniden deneme seçeneği sunulur.

## Optik / termal operatör ekranı

Bağlantı formunda ana/optik kanal ile isteğe bağlı termal kanal ayrı seçilir.
Termal kanal doldurulunca iki RTSP video yayını bağımsız yeniden bağlantıyla yan yana gösterilir.
Üstteki Optik / Çift görüntü / Termal düğmeleri görünümü değiştirir; en-boy oranı korunur.
Sağdaki sürekli görünür PTZ yönleri, DUR, optik zoom ve hız düğmeleri doğrudan kontrol sağlar.
PTZ ayarlarından kontrol protokolü, web portu ve kanal değiştirilebilir.
Bir saniyeden uzun süredir yeni termal kare yoksa eski kare kaldırılır ve bekleme durumu gösterilir.
Trafik analizi optik/ana yayında yapılır; termal görüntüde sıcaklık ölçümü veya radyometrik analiz yapılmaz.
Kanal numaraları modele bağlıdır; termal görüntü otomatik olarak başka kanaldan varsayılmaz.
Komut satırı: `python -m tools.camera_app --thermal-channel 2` (modelin gerçek kanalını kullanın).

### Akıcılık ve analiz

Optik önizleme analizden bağımsız olarak en fazla 30 Hz yenilenir; yalnız en yeni kare tutulur.
Model meşgulken optik ve termal video yenilenmeye devam eder, PTZ ve çıkış kontrolleri işlenir.
Analiz sonuçları kendi kareleriyle eşleştirilir: daha yeni video gösterilmişse eski analiz karesiyle
görüntü geri sarılmaz ve eski kutular yeni kareye çizilmez. Analiz hızı ve nesne özeti altta gösterilir.
Bir saniye boyunca yeni optik kare alınmazsa eski önizleme kaldırılır.
YOLO-World varsayılan olarak en-boy oranına uygun minimum dolgu ile çalışır; görüntü kırpılmaz.
Eski kare dolgusunu kullanmak için `--square-inference` verilebilir; sabit sınıflı modelin dolgu düzeni korunur.
Aynı video karesi yeniden çizilirken ölçeklenmiş görüntü önbellekten kullanılır; yeni kare ve görünüm değişimi önbelleği yeniler.

## Manuel PTZ ve optik zoom

Sağdaki `PTZ ayarları` düğmesi veya `P` tuşu ayrı bir kontrol paneli açar.
Panel görüntü penceresinin üstünde görünür. Sol/sağ/yukarı/aşağı,
yakınlaştır/uzaklaştır, hız, web portu ve kontrol kanalı bulunur. Otomatik hedef izleme eklenmemiştir.
Hikvision'da ISAPI continuous komutları, Dahua'da CGI start/stop kullanılır. Panelden protokol değiştirilebilir.
ISAPI kanal numarası 1, Dahua PTZ kanal indeksi 0 ile başlar; RTSP yayın kanalı değiştirilmez.

Her tıklama 250 ms hareket isteği yapar, ardından durdurma gönderir. Bir komut sürerken yeni hareketler sıraya alınmaz.
`DUR` veya panelde boşluk tuşu hareketin durmasını ister; önceki komut bitmişse durdurmayı tekrar gönderir.
Panel kapatılırken veya kamera değişirken devam eden hareket için durdurma istenir ve kontrol bağlantıları kapatılır.
Komutun zaman aşımına uğraması, kameranın onu almamış olduğunu garanti etmez; durdurma başarısızlığı panelde belirtilir.
Panelin açık olması kamerayı hareket ettirmez; yalnız operatör komutları hareket gönderir.
PTZ komutu devam ederken trafik geçişleri sayılmaz, toplamlar korunur ve sonrasında konum geçmişi yeniden başlatılır.
Optik zoom, kamera donanımı ve hesap yetkisi tarafından desteklenmelidir; sabit lensli kamerada optik zoom sağlanamaz.

Protokol kaynakları: [Hikvision ISAPI](https://open.hikvision.com/hardware/v2/08%E5%8D%8F%E8%AE%AE%E9%80%8F%E4%BC%A0/%E4%BA%91%E5%8F%B0%E5%92%8C%E8%B7%9F%E9%9A%8F%E5%AE%9A%E4%BD%8D.html),
[Dahua HTTP API, PTZ bölüm 7.2](https://community.jeedom.com/uploads/short-url/tTQJPaNah7gZnU12VGGN9ZHEhOk.pdf).

Referans: [HodenX/python-traffic-counter-with-yolo-and-sort](https://github.com/HodenX/python-traffic-counter-with-yolo-and-sort).
Referansın YOLO + takip + çizgi sayımı fikri mevcut YOLO-World/ByteTrack/Tripwire bileşenleriyle bağımsız uygulanmıştır;
YOLOv3/SORT kaynak kodu kopyalanmamıştır. Bu eklenti yeni bir model eğitimi değildir.
