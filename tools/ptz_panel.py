"""Optional standard ttk control window; preserves the existing video/connection UI."""
from __future__ import annotations

import threading
from dataclasses import replace

from ipcam.ptz_control import ManualPTZ


class PTZPanel:
    def __init__(self, camera):
        self.camera = camera
        self.root = None
        self.closing = threading.Event()
        self.controller = None

    def show(self):
        if self.root is not None:
            self.root.lift()
            self.root.focus_force()
            return
        self.closing.clear()
        self._run()

    def pulse(self, direction, speed=30):
        if self.controller is None:
            protocol = 'Dahua' if (self.camera.rtsp_path or '').startswith('/cam/realmonitor') else 'ISAPI'
            channel = self.camera.channel-1 if protocol == 'Dahua' else self.camera.channel
            self.controller = ManualPTZ(self.camera, protocol, channel)
        return self.controller.pulse(direction, speed, .25)

    def pump(self):
        # Tk and OpenCV windows stay on the main UI thread. HTTP commands run separately.
        if self.root is not None:
            self.root.update()

    def close(self):
        self.closing.set()
        controller = self.controller
        if controller is not None:
            controller.stop()
        if self.root is not None:
            self.root.destroy()
            self.root = None
        if controller is not None:
            controller.close()
            self.controller = None

    @property
    def moving(self):
        controller = self.controller
        return controller is not None and controller.future is not None and not controller.future.done()

    @property
    def motion_token(self):
        controller = self.controller
        return None if controller is None else (id(controller), controller.motion_serial)

    def _run(self):
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        self.root = root
        root.title("Kamera • PTZ kontrol")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        from tools.camera_setup import apply_corporate_style
        apply_corporate_style(root)
        tk.Frame(root, background="#0B3A75", height=6).grid(sticky="ew")
        box = ttk.Frame(root, padding=22)
        box.grid()
        ttk.Label(box, text="PTZ ve optik yakınlaştırma", style="Title.TLabel", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        initial_protocol = "Dahua" if (self.camera.rtsp_path or "").startswith("/cam/realmonitor") else "ISAPI"
        protocol = tk.StringVar(value=initial_protocol)
        port = tk.StringVar(value=str(self.camera.http_port))
        channel = tk.StringVar(value=str(self.camera.channel-1 if initial_protocol == "Dahua" else self.camera.channel))
        speed = tk.IntVar(value=30)
        status = tk.StringVar(value="Her tıklama kısa bir hareket yapar ve durur.")
        ttk.Label(box, text="Protokol").grid(row=1, column=0, sticky="w")
        selector = ttk.Combobox(box, textvariable=protocol, values=("ISAPI", "Dahua"), state="readonly", width=18)
        selector.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        def changed(_):
            channel.set(str(self.camera.channel-1 if protocol.get() == "Dahua" else self.camera.channel))
        selector.bind("<<ComboboxSelected>>", changed)
        ttk.Label(box, text="Web portu").grid(row=2, column=0, sticky="w")
        ttk.Entry(box, textvariable=port, width=10).grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(box, text="Kontrol kanalı").grid(row=3, column=0, sticky="w")
        ttk.Entry(box, textvariable=channel, width=10).grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(box, text="Hız").grid(row=4, column=0, sticky="w")
        ttk.Scale(box, from_=10, to=100, variable=speed).grid(row=4, column=1, columnspan=2, sticky="ew", pady=8)
        config = None
        def stop():
            if self.controller is not None:
                self.controller.stop()
            status.set("Durdurma istendi.")
        def move(direction):
            nonlocal config
            try:
                current = (protocol.get(), int(port.get()), int(channel.get()))
                if not 1 <= current[1] <= 65535 or not 0 <= current[2] <= 999 or (current[0] == "ISAPI" and current[2] == 0):
                    raise ValueError()
            except ValueError:
                status.set("Port/kanal geçersiz. ISAPI kanal 1, Dahua kanal 0 ile başlar.")
                return
            if self.controller is not None and self.controller.future is not None and not self.controller.future.done():
                status.set("Komut sürüyor; yeni hareket sıraya alınmadı.")
                return
            if current != config:
                if self.controller is not None:
                    self.controller.close()
                self.controller = ManualPTZ(replace(self.camera, http_port=current[1]), current[0], current[2])
                config = current
            self.controller.pulse(direction, int(speed.get()), .25)
            status.set("Komut gönderiliyor…")
        for title, direction, row, col in [("Yukarı", "up", 5, 1), ("Sol", "left", 6, 0),
            ("Sağ", "right", 6, 2), ("Aşağı", "down", 7, 1),
            ("Yakınlaştır +", "zoom_in", 8, 0), ("Uzaklaştır −", "zoom_out", 8, 2)]:
            ttk.Button(box, text=title, command=lambda d=direction: move(d)).grid(row=row, column=col, padx=4, pady=4, sticky="ew")
        ttk.Button(box, text="DUR", command=stop, style="Danger.TButton").grid(row=6, column=1, sticky="ew", padx=4)
        ttk.Label(box, textvariable=status, wraplength=350).grid(row=9, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Label(box, text="Kamera PTZ/optik zoom donanımı ve kontrol yetkisi gerekli.", wraplength=350).grid(
            row=10, column=0, columnspan=3, sticky="w")
        seen = None
        def close():
            self.close()
        def poll():
            nonlocal seen
            if self.closing.is_set():
                close()
                return
            if self.controller is not None:
                future = self.controller.future
                if future is not None and future.done() and future is not seen:
                    seen = future
                    if future.exception() is None:
                        status.set("Hareket tamamlandı; durdurma komutu kabul edildi.")
                    else:
                        status.set("PTZ komutu başarısız. Protokol, kanal, PTZ desteği ve yetkiyi kontrol edin; durdurma doğrulanamadı.")
            root.after(100, poll)
        root.protocol("WM_DELETE_WINDOW", close)
        root.bind("<Escape>", lambda _: close())
        root.bind("<space>", lambda _: stop())
        root.after(100, poll)
