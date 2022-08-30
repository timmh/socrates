import os


class LedManager:
    def __init__(self):
        pass

    def enable_leds(self):
        for path in [
            "/sys/class/leds/pwr/brightness",
            "/sys/class/leds/mmc0::/brightness",
        ]:
            os.system(f"sudo sh -c 'echo 255 > \"{path}\"'")

    def disable_leds(self):
        for path in [
            "/sys/class/leds/pwr/brightness",
            "/sys/class/leds/mmc0::/brightness",
        ]:
            os.system(f"sudo sh -c 'echo 0 > \"{path}\"'")