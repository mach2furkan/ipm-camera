"""Interactive camera launcher; passwords stay within this process."""
import sys
from tools.live_detect import main

if __name__ == "__main__":
    raise SystemExit(main(["--setup", "--windowed", *sys.argv[1:]]))
