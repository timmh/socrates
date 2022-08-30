from typing import Optional
import os
import threading
import time
import datetime
import logging
from .utils import get_uptime


import dbus
from gi.repository import GObject
from dbus.mainloop.glib import DBusGMainLoop
DBusGMainLoop(set_as_default=True)


class SleepManager:
    def __init__(self, lamp, sleep_interval=60*60*3, wake_interval=60, uptime_backoff=5*60):
        self.sleep_interval = sleep_interval
        self.wake_interval = wake_interval
        self.uptime_backoff = uptime_backoff
        self.sleep_until_epoch = 0
        self.allow_sleep = False
        self.optional_devices_enabled = True
        self.dbus_bus = dbus.SystemBus()
        self.gobject_loop = GObject.MainLoop()
        self.dbus_bus.add_signal_receiver(
            self._resumed,
            'PrepareForSleep',
            'org.freedesktop.login1.Manager',
            'org.freedesktop.login1',
        )
        self.loop_lock = threading.Lock()
        self.loop_stop_event = threading.Event()
        self.loop_thread: Optional[threading.Thread] = None
        self.motion_detector = None
        self.debug_disable_sleep = False
        self.lamp = lamp

    def set_motion_detector(self, motion_detector):
        self.motion_detector = motion_detector

    def is_sleep_allowed(self):
        return (
            self.allow_sleep and
            self.sleep_until_epoch + self.wake_interval < time.time() and
            get_uptime() > self.uptime_backoff and 
            not os.path.exists("/tmp/nosleep") and
            not self.debug_disable_sleep
        )

    def _loop(self):
        assert not self.loop_stop_event.is_set()
        while True:
            while not self.is_sleep_allowed():
                if self.loop_stop_event.wait(1):
                    return
                if self.sleep_until_epoch <= time.time() <= self.sleep_until_epoch + self.wake_interval:
                    self._enable_optional_devices(True)

            with self.loop_lock:

                # make sure lamp is turned off
                self.lamp.turn_off()

                # set wakeup time
                self.sleep_until_epoch = int(round(time.time() + self.sleep_interval))
                if self.motion_detector is not None:
                    self.motion_detector.set_scheduled_wakeup_time(self.sleep_until_epoch)
                logging.info(f"sleeping until {datetime.datetime.fromtimestamp(self.sleep_until_epoch)}")
                if os.system('sudo sh -c "echo 0 > /sys/class/rtc/rtc0/wakealarm"') != 0:
                    raise RuntimeError("failed to remove old wake time")
                if os.system(f'sudo sh -c "echo {self.sleep_until_epoch} > /sys/class/rtc/rtc0/wakealarm"') != 0:
                    raise RuntimeError("failed to set wake time")

                # save power by disabling optional devices
                self._enable_optional_devices(False)

                # sleep
                if os.system("sudo systemctl suspend") != 0:
                    raise RuntimeError("failed to suspend")

            # wait until system is woken up
            self.gobject_loop.run()

            # stay awake for at most wake_interval seconds
            if self.loop_stop_event.wait(self.wake_interval):
                return

    def _start_loop(self):
        with self.loop_lock:
            assert self.loop_thread is None or not self.loop_thread.is_alive()
            self.loop_stop_event.clear()
            self.loop_thread = threading.Thread(target=self._loop)
            self.loop_thread.start()

    def _stop_loop(self):
        with self.loop_lock:
            self.loop_stop_event.set()
            if self.loop_thread is not None:
                self.loop_thread.join()
            self.loop_stop_event.clear()

    def set_allow_sleep(self, allow_sleep: bool):
        with self.loop_lock:
            if allow_sleep:
                logging.info("allowing sleep")
            else:
                logging.info("disallowing sleep")
            self.allow_sleep = allow_sleep
        if not self.debug_disable_sleep:
            self._stop_loop()
            if self.allow_sleep:
                self._start_loop()

    def _resumed(self, will_sleep):
        if not will_sleep:
            if time.time() >= self.sleep_until_epoch:
                logging.info(f"woke up after timer at {datetime.datetime.now()}")

                # woke up from timer, enable optional devices to facilitate maintenance
                self._enable_optional_devices(True)
            else:
                logging.info(f"woke up before timer at {datetime.datetime.now()}")
            self.gobject_loop.quit()
        else:
            logging.info(f"system is going to sleep at {datetime.datetime.now()}")

    def _enable_optional_devices(self, enable: bool):
        if self.optional_devices_enabled == enable:
            return
        if enable:
            logging.info("enabling optional devices")

            # enable wifi
            if os.system("nmcli radio wifi on") != 0:
                logging.warn(f"failed to enable wifi")
        else:
            logging.info("disabling optional devices")

            # disable wifi
            if os.system("nmcli radio wifi off") != 0:
                logging.warn(f"failed to disable wifi")
        self.optional_devices_enabled = enable

    def __del__(self):
        with self.loop_lock:
            self.allow_sleep = False
        self._stop_loop()
