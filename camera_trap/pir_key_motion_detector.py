import asyncio
import evdev
import time


class PirKeyMotionDetector:
    def __init__(self, keycode="KEY_POWER", detection_duration=30., cooldown_duration=5 * 60.):
        self.device = evdev.InputDevice("/dev/input/by-path/platform-gpio-keys-event")
        self.keycode = keycode
        self.detection_duration = detection_duration
        self.cooldown_duration = cooldown_duration
        self.last_detection_time = 0
        self.scheduled_wakeup_time = None

    def set_scheduled_wakeup_time(self, scheduled_wakeup_time):
        self.scheduled_wakeup_time = scheduled_wakeup_time

    async def motion_detected(self) -> bool:
        async for event in self.device.async_read_loop():
            if event.type == evdev.ecodes.EV_KEY:
                event = evdev.KeyEvent(event)
                if event.keystate == evdev.KeyEvent.key_up and event.keycode == self.keycode:
                    now = time.time()
                    if (self.last_detection_time + self.detection_duration + self.cooldown_duration < now) and (self.scheduled_wakeup_time is None or self.scheduled_wakeup_time > now):
                        self.last_detection_time = now
                        yield True
                        await asyncio.sleep(self.detection_duration)
                        yield False