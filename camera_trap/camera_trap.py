import os
import shutil
import logging
import datetime
from enum import Enum, auto
from .pir_key_motion_detector import PirKeyMotionDetector
from .lamp import Lamp
from .sleep_manager import SleepManager
from .csi_subprocess import CSI
from .basestation_interface import BasestationInterface
from .led_manager import LedManager
from .utils import checksum, get_temp


class CameraTrapState(Enum):
    INACTIVE = auto()
    WATCHING = auto()
    RECORDING = auto()


class CameraTrap:
    def __init__(self):
        self.state = CameraTrapState.INACTIVE
        self.motion_detector = PirKeyMotionDetector()
        self.lamp = Lamp()
        self.sleep_manager = SleepManager(self.lamp)
        self.sleep_manager.set_motion_detector(self.motion_detector)
        self.csi = CSI()
        self.csi_args = dict(
            framerate=30,
            width=1920,
            height=1080,
        )
        self.basestation_interface = BasestationInterface(telemetry=self.get_telemetry())
        self.led_manager = LedManager()
        self.led_manager.disable_leds()

    def get_free_space_ratio(self):
        total, _, free = shutil.disk_usage(os.environ["CAPTURE_DIR"])
        return free / total

    def get_capture_path(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return os.path.abspath(os.path.join(os.environ["CAPTURE_DIR"], timestamp))

    def get_metadata(
        self,
        filepath: str,
        filename: str,
        detection_start_time: datetime.datetime,
        detection_end_time: datetime.datetime,
    ):
        return {
            "deviceID": os.environ["DEVICE_ID"],
            "serialNumber": os.environ["SERIAL_NUMBER"],
            "timestamp": {
                "start": detection_start_time.astimezone().replace(microsecond=0).isoformat(),
                "stop": detection_end_time.astimezone().replace(microsecond=0).isoformat(),
            },
            "location": {
                "latitude": os.environ["LOCATION_LATITUDE"],
                "longitude": os.environ["LOCATION_LONGITUDE"],
                "geometry": {
                    "type": "Point",
                    "coordinates": [os.environ["LOCATION_LONGITUDE"], os.environ["LOCATION_LATITUDE"]],
                }
            },
            "files": [
                {
                    "fileName": filename,  # make sure there are not collisions with other files
                    "fileSize": os.path.getsize(filepath) * 1e-6,  # file size in megabytes (MB)
                    "md5Checksum": checksum(filepath).hexdigest(),  # compute md5
                }
            ],
            "sourceFiles": [],  # raw data, does not have prior processing steps
        }

    def get_telemetry(self):
        return {
            "deviceID": os.environ["DEVICE_ID"],
            "serialNumber": os.environ["SERIAL_NUMBER"],
            "timestamp": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "location": {
                "latitude": os.environ["LOCATION_LATITUDE"],
                "longitude": os.environ["LOCATION_LONGITUDE"],
                "geometry": {
                    "type": "Point",
                    "coordinates": [os.environ["LOCATION_LONGITUDE"], os.environ["LOCATION_LATITUDE"]],
                }
            },
            "computingUnit": {
                "cpuTemperature": get_temp(),
            },
            "status": {
                "error": False,
                "message": "fully operational",
            }
        }

    async def loop(self):
        assert self.state == CameraTrapState.INACTIVE
        logging.info("watching")
        self.state = CameraTrapState.WATCHING
        self.sleep_manager.set_allow_sleep(True)

        current_output_path = None
        detection_start_time = None
        detection_end_time = None

        async for motion_detected in self.motion_detector.motion_detected():
            if motion_detected:
                logging.info("motion detected via PIR")
                assert self.state == CameraTrapState.WATCHING
                detection_start_time = datetime.datetime.now()
                self.state = CameraTrapState.RECORDING
                self.sleep_manager.set_allow_sleep(False)
                self.lamp.turn_on()
                current_output_path = self.get_capture_path() + ".mkv"
                self.csi.start_capture(**self.csi_args, output_path=current_output_path)
            else:
                logging.info("motion stopped")
                assert self.state == CameraTrapState.RECORDING
                detection_end_time = datetime.datetime.now()
                self.state = CameraTrapState.WATCHING
                self.csi.stop_capture()
                self.lamp.turn_off()

                self.basestation_interface.add_sample(
                    os.path.basename(current_output_path),
                    current_output_path,
                    self.get_metadata(
                        current_output_path,
                        os.path.basename(current_output_path),
                        detection_start_time,
                        detection_end_time,
                    ),
                    self.get_telemetry(),
                )

                self.sleep_manager.set_allow_sleep(True)

    def __del__(self):
        if self.state != CameraTrapState.INACTIVE:
            self.csi.stop_capture()
        self.lamp.turn_off()
        self.sleep_manager.set_allow_sleep(False)
        self.basestation_interface.__del__()
        self.led_manager.enable_leds()
        self.state = CameraTrapState.INACTIVE
        logging.info("shutting down")
