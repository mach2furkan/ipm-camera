"""Interactive camera launcher; passwords stay within this process."""
import sys
from tools.live_detect import main


def startup_error_message(exc):
    if isinstance(exc, MemoryError) or "outofmemory" in type(exc).__name__.lower() or "out of memory" in str(exc).lower():
        return "GPU/RAM belleği yetersiz. Diğer kamera pencerelerini kapatın veya trafik seçeneğini kapatıp yeniden deneyin."
    if isinstance(exc, (ImportError, FileNotFoundError)):
        return "Gerekli yazılım bileşeni veya model dosyası bulunamadı. Python ortamını ve model dosyasını kontrol edin."
    return f"Uygulama başlatılamadı ({type(exc).__name__}). Bağlantı ve görüntü çözme ayarlarını kontrol edin."


def run(argv=None):
    defaults = {}  # Only this process; no password and no persistence.
    args = ["--setup", "--windowed", *(sys.argv[1:] if argv is None else argv)]
    while True:
        try:
            result = main(args, connection_defaults=defaults)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            try:
                retry = messagebox.askretrycancel("Kamera uygulaması", startup_error_message(exc), parent=root)
            finally:
                root.destroy()
            if retry:
                continue
            return 1
        if result == 75:
            continue
        return result

if __name__ == "__main__":
    raise SystemExit(run())
