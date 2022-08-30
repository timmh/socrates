#!/usr/bin/env bash

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [calibration.mkv] [...optional]"
fi

SCRIPT_DIR="$(realpath $(dirname $0))"

# convert side-by-side video to bagfile
docker run --rm -v "$SCRIPT_DIR:/calibration" -v "$(realpath $1):/calibration.mkv" osrf/ros:noetic-desktop-full python3 /calibration/csi_to_rosbag.py /calibration.mkv /calibration/data/calibration.bag

# run kalibr
xhost +local:root;
docker run --rm -it -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$SCRIPT_DIR:/calibration" --entrypoint /calibration/run_calibr.sh thaucke/kalibr
xhost -local:root;