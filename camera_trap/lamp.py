import importlib
import logging


class Lamp:
    def __init__(self):
        try:
            self.GPIO = importlib.import_module("Jetson.GPIO")
            self.GPIO.setmode(self.GPIO.BOARD)
            self.GPIO.setwarnings(False)
            self.channels = [16]
            self.GPIO.setup(self.channels, self.GPIO.OUT, initial=self.GPIO.LOW)
            self.initialized = True
            self.turn_off()
        except ModuleNotFoundError:
            logging.warning("Lamp could not be initialized")
            self.initialized = False

    def turn_on(self):
        if not self.initialized:
            return
        try:
            self.GPIO.output(self.channels[0], self.GPIO.HIGH)
        except:
            pass

    def turn_off(self):
        if not self.initialized:
            return
        try:
            self.GPIO.output(self.channels[0], self.GPIO.LOW)
        except:
            pass

    def __del__(self):
        if not self.initialized:
            return
        try:
            self.turn_off()
            self.GPIO.cleanup()
        except:
            pass
