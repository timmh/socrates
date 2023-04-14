#!/bin/bash

wget --content-disposition -P "$(dirname $0)" https://github.com/timmh/ONNX-CREStereo-Depth-Estimation/releases/download/v1.0.0/crestereo_combined_iter5_720x1280.onnx

echo Download finished.