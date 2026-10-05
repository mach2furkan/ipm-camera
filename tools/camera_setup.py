"""Local camera connection form. Credentials are never written to disk or argv."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl
from dataclasses import dataclass, replace
from ipcam.stream.connect import BRANDS, common_paths, resolve_camera

from ipcam.config import CameraConfig


@dataclass(frozen=True)
class CameraSelection:
    camera: CameraConfig
    stream: str
    traffic: bool
    brand: str = "Hikvision"
    onvif_port: int = 80
    compatible: bool = False
    custom_path: str = ""


def selection_from_fields(host: str, username: str, password: str, port: str,
                          channel: str, path: str, stream: str, traffic: bool,
                          brand="Hikvision", onvif_port="80", compatible=False) -> CameraSelection:
    host = host.strip().strip("[]")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not host or re.fullmatch(r"[0-9.]+", host) or len(host) > 253 or not all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in host.split(".")
        ):
            raise ValueError("Geçerli bir IP adresi veya kamera adı girin.") from None
    try:
        port_n, channel_n, onvif_n = int(port), int(channel), int(onvif_port)
    except ValueError:
        raise ValueError("Port ve kanal sayı olmalıdır.") from None
    if not 1 <= port_n <= 65535 or not 1 <= channel_n <= 999 or not 1 <= onvif_n <= 65535:
        raise ValueError("Port 1–65535, kanal 1–999 arasında olmalıdır.")
    if not username.strip() or not password:
        raise ValueError("Kullanıcı adı ve şifre gerekli.")
    path = path.strip()
    if path and (not path.startswith("/") or path.startswith("//") or
                 any(c.isspace() or ord(c) < 32 for c in path) or "#" in path):
        raise ValueError("RTSP yolu / ile başlamalı; boşluk veya # içermemeli.")
    # Keep secrets out of redacted URLs, including user-supplied query strings.
    secret_query = {"password", "passwd", "pass", "pwd", "username", "user", "token", "access_token", "auth"}
    if path and ("@" in path or any(key.lower() in secret_query
            for key, _ in parse_qsl(path.partition("?")[2], keep_blank_values=True))):
        raise ValueError("RTSP yoluna şifre veya erişim anahtarı eklemeyin; ayrı şifre alanını kullanın.")
    if stream not in ("main", "sub"):
        raise ValueError("Geçersiz yayın seçimi.")
    if brand not in BRANDS:
        raise ValueError("Geçersiz kamera markası.")
    custom_path = path
    if not path and brand == "Dahua":
        path = common_paths(brand, channel_n, stream)[0]
    if not path and brand == "Özel RTSP":
        raise ValueError("Kameranın özel RTSP yayın yolunu girin.")
    return CameraSelection(CameraConfig(host, username.strip(), password,
        channel=channel_n, rtsp_port=port_n, rtsp_path=path or None), stream, traffic, brand, onvif_n, compatible, custom_path)


class ConnectionForm:
    def __init__(self, root, *, host="192.168.1.64", username="admin", stream="main",
                 port=554, channel=1, path="", traffic=False, brand="Otomatik", onvif_port=80, compatible=False):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.result = None
        self.root = root
        root.title("IP Kamera • Bağlantı ve algılama")
        root.resizable(False, False)
        panel = ttk.Frame(root, padding=24)
        panel.grid()
        ttk.Label(panel, text="Kamerana bağlan", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))
        self.fields = {}
        entries = [("host", "Kamera IP / ağ adı", host), ("username", "Kullanıcı adı", username),
                   ("password", "Şifre", ""), ("port", "RTSP portu", str(port)),
                   ("channel", "Kanal", str(channel)), ("path", "Özel RTSP yolu (isteğe bağlı)", path)]
        for row, (key, label, value) in enumerate(entries, 1):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", padx=(0, 18), pady=6)
            var = tk.StringVar(value=value)
            entry = ttk.Entry(panel, textvariable=var, width=32, show="*" if key == "password" else "")
            entry.grid(row=row, column=1, sticky="ew", pady=6)
            self.fields[key] = var
            if key == "host":
                entry.focus_set()
        self.stream = tk.StringVar(value=stream)
        ttk.Label(panel, text="Yayın").grid(row=7, column=0, sticky="w", pady=6)
        ttk.Combobox(panel, textvariable=self.stream, values=("main", "sub"), state="readonly").grid(
            row=7, column=1, sticky="ew")
        self.traffic = tk.BooleanVar(value=traffic)
        self.brand = tk.StringVar(value=brand)
        ttk.Label(panel, text="Kamera markası / bağlantı").grid(row=8, column=0, sticky="w", pady=6)
        ttk.Combobox(panel, textvariable=self.brand, values=BRANDS, state="readonly").grid(
            row=8, column=1, sticky="ew")
        self.onvif_port = tk.StringVar(value=str(onvif_port))
        ttk.Label(panel, text="ONVIF web portu").grid(row=9, column=0, sticky="w", pady=6)
        ttk.Entry(panel, textvariable=self.onvif_port).grid(row=9, column=1, sticky="ew")
        ttk.Checkbutton(panel, text="Trafik takibi ve sayımı da açık olsun", variable=self.traffic).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=8)
        self.compatible = tk.BooleanVar(value=compatible)
        ttk.Checkbutton(panel, text="Uyumlu görüntü çözme (bağlantı var ama görüntü yoksa)",
                        variable=self.compatible).grid(row=11, column=0, columnspan=2, sticky="w")
        ttk.Label(panel, text="9 nesne sınıfı aktif; kalem/kurşun kalem kapalı. Trafik ek GPU gücü kullanır.\n"
                  "Otomatik: Dahua/Hikvision yolları, ardından ONVIF yayını aranır.\n"
                  "Özel yayın yolu girilirse marka yolunun yerine kullanılır.\n"
                  "Şifre kaydedilmez. Her uygulama penceresi bir kameraya bağlanır.",
                  foreground="#555555").grid(row=12, column=0, columnspan=2, sticky="w", pady=8)
        self.status = tk.StringVar(value="Bağlanmadan önce görüntü kontrol edilir.")
        ttk.Label(panel, textvariable=self.status, wraplength=460).grid(row=13, column=0, columnspan=2, sticky="w", pady=8)
        self.busy = False
        import threading
        self.cancelled = threading.Event()
        def cancel():
            self.cancelled.set()
            self.fields["password"].set("")
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", cancel)
        def connect():
            if self.busy:
                return
            try:
                selection = selection_from_fields(**{k: v.get() for k, v in self.fields.items()},
                    stream=self.stream.get(), traffic=self.traffic.get(), brand=self.brand.get(),
                    onvif_port=self.onvif_port.get(), compatible=self.compatible.get())
            except ValueError as exc:
                messagebox.showerror("Bağlantı bilgilerini kontrol et", str(exc), parent=root)
                return
            import queue
            import threading
            outcomes = queue.Queue()
            updates = queue.Queue()
            self.busy = True
            self.button.configure(state="disabled")
            self.status.set("Kamera yayını aranıyor ve görüntü doğrulanıyor…")
            def check():
                try:
                    camera = resolve_camera(selection.camera, selection.stream, selection.brand, selection.onvif_port,
                                            progress=updates.put, cancelled=self.cancelled)
                    outcomes.put((replace(selection, camera=camera), None))
                except Exception as exc:
                    from ipcam.stream.connect import ConnectionCheckError
                    outcomes.put((None, str(exc) if isinstance(exc, ConnectionCheckError)
                                  else "Bağlantı kontrolü tamamlanamadı. Kamera ayarlarını kontrol edin."))
            def poll():
                if self.cancelled.is_set():
                    return
                while not updates.empty():
                    self.status.set(updates.get_nowait())
                try:
                    result, error = outcomes.get_nowait()
                except queue.Empty:
                    root.after(100, poll)
                    return
                self.busy = False
                self.button.configure(state="normal")
                if error:
                    self.status.set(error)
                    return
                self.result = result
                self.fields["password"].set("")
                root.destroy()
            threading.Thread(target=check, name="camera-connect", daemon=True).start()
            root.after(100, poll)
        self.button = ttk.Button(panel, text="Bağlantıyı kontrol et ve başlat", command=connect)
        self.button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(panel, text="İptal / kapat", command=cancel).grid(row=15, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        root.bind("<Return>", lambda _: connect())
        root.bind("<Escape>", lambda _: cancel())


def ask_camera(**defaults) -> CameraSelection | None:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    form = ConnectionForm(root, **defaults)
    # Also show the form when the Windows console was launched with SW_HIDE.
    root.after(100, root.deiconify)
    root.mainloop()
    return form.result
