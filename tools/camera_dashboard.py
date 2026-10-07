"""Operator dashboard: independent optical/thermal video and visible controls."""
import cv2
import numpy as np

# Corporate light theme (BGR).
PAGE = (247, 245, 243)
CARD = (255, 255, 255)
BORDER = (226, 222, 218)
VIDEO_BG = (236, 233, 230)
NAVY = (117, 58, 11)
BLUE = (196, 99, 21)
TEXT = (55, 41, 31)
MUTED = (138, 114, 100)
GREEN = (74, 148, 30)
AMBER = (20, 120, 200)
RED = (52, 52, 200)


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
        self._background = self._chrome()
        self._tile_cache = {}

    @staticmethod
    def card(canvas, rect, fill=CARD):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x+w, y+h), fill, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), BORDER, 1)

    def _chrome(self):
        canvas = np.empty((900, 1600, 3), np.uint8)
        canvas[:] = PAGE
        cv2.rectangle(canvas, (0, 0), (1600, 84), CARD, -1)
        cv2.line(canvas, (0, 84), (1600, 84), BORDER, 1)
        cv2.rectangle(canvas, (0, 0), (6, 84), NAVY, -1)
        self.card(canvas, (1352, 100, 232, 690))
        cv2.rectangle(canvas, (0, 804), (1600, 900), CARD, -1)
        cv2.line(canvas, (0, 804), (1600, 804), BORDER, 1)
        return canvas

    def button(self, canvas, label, rect, action, active=False, danger=False):
        x, y, w, h = rect
        fill = RED if danger else BLUE if active else CARD
        fg = CARD if danger or active else NAVY
        cv2.rectangle(canvas, (x, y), (x+w, y+h), fill, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), fill if danger or active else BORDER, 1)
        ph, pw = self.painter.patch(label, fill, fg).shape[:2]
        self.painter.put(canvas, label, x+(w-pw)//2, y+(h-ph)//2, fill, fg)
        self.buttons.append((rect, action))

    def tile(self, canvas, frame, rect, label, live):
        x, y, w, h = rect
        self.card(canvas, rect)
        cv2.line(canvas, (x, y+48), (x+w, y+48), BORDER, 1)
        cv2.rectangle(canvas, (x+1, y+49), (x+w-1, y+h-1), VIDEO_BG, -1)
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
            self.painter.put(canvas, 'Görüntü bekleniyor • otomatik yeniden bağlantı', x+16, y+h//2, VIDEO_BG, MUTED)
        self.painter.put(canvas, label, x+14, y+11, CARD, NAVY)
        pill = 'CANLI' if live else 'BAĞLANTI BEKLENİYOR'
        pill_bg = GREEN if live else AMBER
        pw = self.painter.patch(pill, pill_bg, CARD).shape[1]
        self.painter.put(canvas, pill, x+w-pw-14, y+11, pill_bg, CARD)
        return mapping

    def render(self, optical, thermal=None, optical_live=True, thermal_live=False):
        canvas = self._background.copy()
        self.buttons = []
        self.painter.put(canvas, 'CAMERA / OPERATÖR MERKEZİ', 28, 16, CARD, NAVY)
        self.painter.put(canvas, f'{self.host} • Bağımsız kanal bağlantıları', 28, 48, CARD, MUTED)
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
        self.painter.put(canvas, 'PTZ KONTROL', 1366, 116, CARD, NAVY)
        cv2.line(canvas, (1366, 148), (1570, 148), BORDER, 1)
        for label, direction, x, y in [('↑', 'up', 1430, 170), ('←', 'left', 1360, 230),
                ('→', 'right', 1500, 230), ('↓', 'down', 1430, 290),
                ('Zoom +', 'zoom_in', 1360, 365), ('Zoom −', 'zoom_out', 1470, 365)]:
            self.button(canvas, label, (x, y, 65 if y < 350 else 100, 50), lambda d=direction: self.move(d))
        self.button(canvas, 'DUR', (1430, 230, 65, 50), self.stop, danger=True)
        self.painter.put(canvas, f'Hız: {self.speed}', 1366, 444, CARD, TEXT)
        self.button(canvas, '−', (1360, 480, 95, 44), lambda: self.change_speed(-10))
        self.button(canvas, '+', (1470, 480, 95, 44), lambda: self.change_speed(10))
        self.button(canvas, 'PTZ ayarları', (1360, 560, 205, 48), self.panel.show, True)
        self.painter.put(canvas, 'Boşluk: DUR', 1366, 635, CARD, MUTED)
        self.painter.put(canvas, 'Optik zoom: donanım', 1366, 668, CARD, MUTED)
        self.painter.put(canvas, 'desteği gerekir', 1366, 697, CARD, MUTED)
        controller = self.panel.controller
        if controller and controller.future and controller.future.done() and controller.future is not self.seen:
            self.seen = controller.future
            self.status = 'Komut tamamlandı' if controller.future.exception() is None else 'PTZ hatası • protokol / kanal / yetkiyi kontrol edin'
        self.painter.put(canvas, self.analysis_text, 28, 814, CARD, TEXT)
        self.painter.put(canvas, self.status, 28, 843, CARD, NAVY)
        self.painter.put(canvas, 'C: bağlantı ayarları   F: tam ekran   S: görüntüyü kaydet   P: PTZ ayarları   Q: çıkış',
                         28, 871, CARD, MUTED)
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
