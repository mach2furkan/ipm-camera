"""Operator dashboard: independent optical/thermal video and visible controls."""
import cv2
import numpy as np


class CameraDashboard:
    def __init__(self, panel, painter, host, thermal_channel=None, optical_channel=1, traffic=None):
        self.panel, self.painter, self.host = panel, painter, host
        self.thermal_channel, self.optical_channel = thermal_channel, optical_channel
        self.traffic = traffic
        self.mode = 'dual' if thermal_channel else 'optical'
        self.buttons = []
        self.optical_mapping = None
        self.status = 'Hazır • Hareket için yön düğmesine tıklayın'
        self.speed = 30
        self.seen = None
        self.analysis_text = 'Analiz hazırlanıyor'
        self._background = np.empty((900, 1600, 3), np.uint8)
        self._background[:] = (21, 17, 13)
        self._tile_cache = {}

    def button(self, canvas, label, rect, action, active=False):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (145, 95, 35) if active else (57, 43, 31), -1)
        self.painter.put(canvas, label, x+10, y+10, (145, 95, 35) if active else (57, 43, 31))
        self.buttons.append((rect, action))

    def tile(self, canvas, frame, rect, label, live):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (28, 23, 18), -1)
        mapping = None
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(w/fw, (h-48)/fh)
            dw, dh = max(1, int(fw*scale)), max(1, int(fh*scale))
            dx, dy = x+(w-dw)//2, y+48+(h-48-dh)//2
            cached = self._tile_cache.get(label)
            if cached is None or cached[0] is not frame or cached[1] != (dw, dh):
                resized = cv2.resize(frame, (dw, dh),
                    interpolation=cv2.INTER_AREA if dw < fw else cv2.INTER_LINEAR)
                cached = (frame, (dw, dh), resized)
                self._tile_cache[label] = cached
            canvas[dy:dy+dh, dx:dx+dw] = cached[2]
            mapping = (dx, dy, dw, dh, fw, fh)
        else:
            self.painter.put(canvas, 'Görüntü bekleniyor • otomatik yeniden bağlantı', x+16, y+h//2, (28, 23, 18))
        self.painter.put(canvas, label, x+12, y+10, (28, 23, 18))
        self.painter.put(canvas, 'CANLI' if live else 'BAĞLANTI BEKLENİYOR', x+max(12, w-230), y+10,
                         (45, 95, 30) if live else (35, 60, 115))
        return mapping

    def render(self, optical, thermal=None, optical_live=True, thermal_live=False):
        canvas = self._background.copy()
        self.buttons = []
        self.painter.put(canvas, 'CAMERA / OPERATÖR MERKEZİ', 24, 18, (21, 17, 13))
        self.painter.put(canvas, f'{self.host} • Bağımsız kanal bağlantıları', 24, 50, (21, 17, 13))
        modes = [('Optik', 'optical')]
        if self.thermal_channel:
            modes += [('Çift görüntü', 'dual'), ('Termal', 'thermal')]
        for i, (label, mode) in enumerate(modes):
            self.button(canvas, label, (760+i*170, 20, 160, 46), lambda m=mode: setattr(self, 'mode', m), self.mode == mode)
        self.optical_mapping = None
        if self.mode == 'dual':
            self.optical_mapping = self.tile(canvas, optical, (24, 100, 650, 690),
                f'OPTİK • Kanal {self.optical_channel}', optical_live)
            self.tile(canvas, thermal, (690, 100, 650, 690), f'TERMAL • Kanal {self.thermal_channel}', thermal_live)
        elif self.mode == 'optical':
            self.optical_mapping = self.tile(canvas, optical, (24, 100, 1316, 690),
                f'OPTİK • Kanal {self.optical_channel}', optical_live)
        else:
            self.tile(canvas, thermal, (24, 100, 1316, 690), f'TERMAL • Kanal {self.thermal_channel}', thermal_live)
        self.painter.put(canvas, 'PTZ KONTROL', 1360, 110, (21, 17, 13))
        for label, direction, x, y in [('↑', 'up', 1430, 170), ('←', 'left', 1360, 230),
                ('→', 'right', 1500, 230), ('↓', 'down', 1430, 290),
                ('Zoom +', 'zoom_in', 1360, 365), ('Zoom −', 'zoom_out', 1470, 365)]:
            self.button(canvas, label, (x, y, 65 if y < 350 else 100, 50), lambda d=direction: self.move(d))
        self.button(canvas, 'DUR', (1430, 230, 65, 50), self.stop, True)
        self.painter.put(canvas, f'Hız: {self.speed}', 1360, 440, (21, 17, 13))
        self.button(canvas, '−', (1360, 480, 95, 44), lambda: self.change_speed(-10))
        self.button(canvas, '+', (1470, 480, 95, 44), lambda: self.change_speed(10))
        self.button(canvas, 'PTZ ayarları', (1360, 560, 205, 48), self.panel.show)
        self.painter.put(canvas, 'Boşluk: DUR', 1360, 635, (21, 17, 13))
        self.painter.put(canvas, 'Optik zoom: donanım', 1360, 668, (21, 17, 13))
        self.painter.put(canvas, 'desteği gerekir', 1360, 697, (21, 17, 13))
        controller = self.panel.controller
        if controller and controller.future and controller.future.done() and controller.future is not self.seen:
            self.seen = controller.future
            self.status = 'Komut tamamlandı' if controller.future.exception() is None else 'PTZ hatası • protokol / kanal / yetkiyi kontrol edin'
        self.painter.put(canvas, self.status, 24, 815, (21, 17, 13))
        self.painter.put(canvas, self.analysis_text, 24, 785, (21, 17, 13))
        self.painter.put(canvas, 'C: bağlantı ayarları   F: tam ekran   S: görüntüyü kaydet   P: PTZ ayarları   Q: çıkış',
                         24, 858, (21, 17, 13))
        return canvas

    def change_speed(self, delta):
        self.speed = max(10, min(100, self.speed+delta))

    def move(self, direction):
        accepted = self.panel.pulse(direction, self.speed)
        self.status = 'Hareket gönderiliyor…' if accepted else 'Komut sürüyor • hareket sıraya alınmadı'

    def stop(self):
        if self.panel.controller:
            self.panel.controller.stop()
        self.status = 'Durdurma istendi'

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for (left, top, width, height), action in self.buttons:
                if left <= x < left+width and top <= y < top+height:
                    action()
                    return
        if self.traffic and self.optical_mapping:
            left, top, width, height, fw, fh = self.optical_mapping
            if left <= x < left+width and top <= y < top+height:
                self.traffic.click(event, int((x-left)*fw/width), int((y-top)*fh/height), flags, param)
