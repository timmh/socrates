import os
import logging
import gi


gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)


# TODO: this does not work with the current JetPack SDK version due to segfaults. Use csi_subprocess.py instead
class CSI:
    def __init__(self, debug_print_method=None):
        self.pipeline = None
        self.debug_print_method = debug_print_method

    def start_capture(self, framerate: int, width: int, height: int, output_path: str):
        if self.pipeline is not None:
            raise RuntimeError("CSI: pipeline is already running")
        use_color = os.path.exists("/dev/video2")
        pipeline_description = f"""
            nvarguscamerasrc sensor_id=0 name=left \
            nvarguscamerasrc sensor_id=1 name=right \
            {'v4l2src device=/dev/video2 name=color' if use_color else ''} \
            nvcompositor name=mix sink_0::xpos=0 sink_0::ypos=0 sink_0::width={width} sink_0::height={height} sink_1::xpos={width} sink_1::ypos=0 sink_1::width={width} sink_1::height={height} {f'sink_2::xpos=0 sink_2::ypos={height} sink_2::width={width} sink_2::height={height}' if use_color else ''} \
            left.  ! queue ! video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1,format=NV12 ! nvvidconv flip-method=2 ! video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1,format=RGBA ! mix.sink_1 \
            right. ! queue ! video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1,format=NV12 ! nvvidconv flip-method=2 ! video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1,format=RGBA ! mix.sink_0 \
            {f'color. ! queue ! aspectratiocrop aspect-ratio={width}/{height} ! videoscale add-borders=false ! video/x-raw,width={width},height={height} ! videoconvert ! nvvidconv flip-method=0 ! video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1,format=RGBA ! mix.sink_2' if use_color else ''} \
            mix. ! nvvidconv ! omxh265enc control-rate=2 bitrate=8000000 ! matroskamux ! filesink location="{os.path.realpath(output_path)}"
        """
        self.pipeline = Gst.parse_launch(pipeline_description)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        if self.debug_print_method is not None:
            bus.connect("message", self.debug_print_method)
        self.pipeline.set_state(Gst.State.PLAYING)
        bus.connect("message", self.on_message)
        logging.info("gstreamer pipeline started")

    def on_message(self, _, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.warn(f"gstreamer error: {err}\n{debug}")
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")

    def stop_capture(self):
        if self.pipeline is not None:
            self.pipeline.send_event(Gst.Event.new_eos())
            self.pipeline.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS)
            self.pipeline.set_state(Gst.State.NULL)
            logging.info("gstreamer pipeline stopped")
            self.pipeline = None

    def __del__(self):
        self.stop_capture()
