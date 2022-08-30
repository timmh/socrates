#!/usr/bin/env bash

set -e

# setup ros environment
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
source /catkin_ws/devel/setup.bash

# run calibration
pushd /calibration/data
rosrun kalibr kalibr_calibrate_cameras --show-extraction --bag calibration.bag --models pinhole-radtan pinhole-radtan --topics camera/left_raw camera/right_raw --target ../target.yaml
popd