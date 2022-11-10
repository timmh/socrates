import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
import argparse
import numpy as np
import pandas as pd
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import matplotlib.pyplot as plt
from rectify import rectify
from crestereo import CREStereo


def main():
    parser = argparse.ArgumentParser(description="evaluate disparities obtained with stereo vision using ground truth measurements")
    parser.add_argument("in_file", type=str)
    parser.add_argument("--file_type_intensity", type=str, default="png")
    parser.add_argument("--file_type_disparity", type=str, default="exr")
    parser.add_argument("--force_overwrite", action="store_true", help="set to re-extract existing frames (destructive)")
    args = parser.parse_args()

    camera_focal_length = 0.006  # meters
    camera_baseline = 0.25  # meters
    camera_horizontal_pixels = 1920  # pixels
    camera_horizontal_sensor_size = 0.007564  # meters
    camera_disparity_to_depth_factor = camera_focal_length * camera_baseline * camera_horizontal_pixels * camera_horizontal_sensor_size ** -1
    
    processing_size = (1280, 720)
    crestereo = CREStereo(
        f'weights/crestereo/crestereo_combined_iter5_{processing_size[1]}x{processing_size[0]}.onnx',
    )

    gt_measurements = pd.read_csv(os.path.join(os.path.dirname(__file__), "gt_measurements.csv"))
    gt_measurements_mask = cv2.imread(os.path.join(os.path.dirname(__file__), "gt_measurements_mask.png"), cv2.IMREAD_GRAYSCALE)
    
    x = []
    y = []
    for _, left, right in rectify(input_file=args.in_file, recalibrate=False, recalibrate_iterative=False):
        original_size = (left.shape[1], left.shape[0])
        disp_crestereo = crestereo(
            cv2.resize(left, processing_size, interpolation=cv2.INTER_LINEAR),
            cv2.resize(right, processing_size, interpolation=cv2.INTER_LINEAR),
        )
        disparity_scale = original_size[0] / processing_size[0]
        disp_crestereo = cv2.resize(disp_crestereo * disparity_scale, original_size, interpolation=cv2.INTER_NEAREST)
        
        for gt_measurement, mask_value in zip(gt_measurements["Ground Truth [m]"], np.unique(gt_measurements_mask)[1:]):
            disparity_measurement = np.median(disp_crestereo[gt_measurements_mask == mask_value])
            disparity_true = camera_disparity_to_depth_factor / gt_measurement
            x.append(disparity_measurement)
            y.append(disparity_true)

        break

    x = np.array(x)
    y = np.array(y)

    rmse = np.sqrt(np.mean((x - y) ** 2))
    epe = np.mean(np.sqrt((x - y) ** 2))
    rel = np.mean(np.abs(1 - x / y))

    print(f"RMSE: {rmse:.2f} px")
    print(f"EPE: {epe:.2f} px")
    print(f"REL: {rel:.2f}")

    plt.scatter(x, y)
    plt.xlabel("Estimated Disparity [px]")
    plt.ylabel("True Disparity [px]")
    plt.show()


if __name__ == "__main__":
    main()
