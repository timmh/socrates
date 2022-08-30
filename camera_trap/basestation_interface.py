import os
import sys
import datetime
import threading
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ammod-sensor-coap-api", "build", "python"))
from CoAP_Python_API import SensorAPI


class BasestationInterface:
    def __init__(self, sample_name="stereocamerafeed", telemetry={}):
        self.io_thread = None
        self.sample_name = sample_name
        self.api = SensorAPI("5683")
        self.api.AddSample(self.sample_name, False, SensorAPI.DataType.DATATYPE_RAW)
        self.stop_flag = threading.Event()
        self.io_thread = threading.Thread(target=self.io_process)
        self.io_thread.start()

    def add_sample(self, filename: str, filepath: str, metadata: dict, telemetry=None):
        self.api.UpdateMetadata(metadata, os.path.splitext(filename)[0] + ".json")
        self.api.UpdateSample(self.sample_name, filename, filepath)

        if telemetry is not None:
            self.api.UpdateTelemetry(telemetry, "telemetry.json")

    def io_process(self):
        io_wait_time = datetime.timedelta(seconds=0.1)
        while not self.stop_flag.wait(0.001):
            if self.api.IoProcess(io_wait_time) == -1:
                logging.error("an error occured during api io process")

    def __del__(self):
        if self.io_thread is not None:
            self.stop_flag.set()
            self.io_thread.join()
