import subprocess
import signal
import os


class CSI():
    def __init__(self):
        self.process = None

    def start_capture(self, framerate: int, width: int, height: int, output_path: str):
        if self.process is not None:
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
        self.process = subprocess.Popen("/usr/bin/gst-launch-1.0 " + pipeline_description.replace("\n", " ").replace("(", r"\(").replace(")", r"\)"), shell=True, preexec_fn=os.setsid)

    def stop_capture(self):
        if self.process is not None:
            # self.process.send_signal(signal.SIGINT)
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("KILLING")
                self.process.kill()
            self.process = None

    def __del__(self):
        self.stop_capture()
