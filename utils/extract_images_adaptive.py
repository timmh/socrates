import argparse
from configparser import Interpolation
import os
import glob
from tqdm.auto import tqdm
from rectify import rectify, RectificationException
import cv2
import numpy as np
from crestereo import CREStereo


def main():
    parser = argparse.ArgumentParser(description="sample images from side-by-side video based on an accumulated motion threshold")
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--foreground_ratio_threshold", type=int, default=10)
    parser.add_argument("--min_sampling_interval", type=int, default=300, help="set to zero to disable adaptive extraction")
    parser.add_argument("--file_type_intensity", type=str, default="png")
    parser.add_argument("--file_type_disparity", type=str, default="exr")
    parser.add_argument("--force_overwrite", action="store_true", help="set to re-extract existing frames (destructive)")
    args = parser.parse_args()
    
    assert os.path.isdir(args.in_dir) and os.path.isdir(args.out_dir)

    processing_size = (1280, 720)
    crestereo = CREStereo(
        f'weights/crestereo/crestereo_combined_iter5_{processing_size[1]}x{processing_size[0]}.onnx',
    )

    for in_file in tqdm(sorted(glob.glob(os.path.join(args.in_dir, "*.mkv")))):
        file_out_dir = os.path.join(args.out_dir, os.path.splitext(os.path.basename(in_file))[0])
        reextract = False
        if os.path.exists(file_out_dir):
            reextract = True
            if not args.force_overwrite:
                continue

        file_out_dir_left = os.path.join(file_out_dir, "left")
        file_out_dir_right = os.path.join(file_out_dir, "right")
        file_out_dir_disp_crestereo = os.path.join(file_out_dir, "disp_crestereo")
        os.makedirs(file_out_dir_left, exist_ok=True)
        os.makedirs(file_out_dir_right, exist_ok=True)
        os.makedirs(file_out_dir_disp_crestereo, exist_ok=True)

        gmm = cv2.createBackgroundSubtractorMOG2()
        fg_acc = 0
        last_extracted_idx = -float("inf")

        frames_to_reextract = set([
            int(os.path.splitext(os.path.basename(f))[0]) for f in
            glob.glob(os.path.join(file_out_dir_left, f"*.{args.file_type_intensity}")) +
            glob.glob(os.path.join(file_out_dir_right, f"*.{args.file_type_intensity}"))
        ])

        try:
            for i, left, right in rectify(input_file=in_file, recalibrate=False, recalibrate_iterative=False):
                if reextract and i not in frames_to_reextract:
                    continue
                fg_ratio = (np.sum(gmm.apply(cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)) > 127) / np.prod(left.shape[0:2])) if not reextract else 0
                fg_acc += fg_ratio
                if fg_acc >= args.foreground_ratio_threshold / 100 or i >= last_extracted_idx + args.min_sampling_interval or reextract:
                    fg_acc = 0
                    last_extracted_idx = i

                    original_size = (left.shape[1], left.shape[0])
                    disp_crestereo = crestereo(
                        cv2.resize(left, processing_size, interpolation=cv2.INTER_LINEAR),
                        cv2.resize(right, processing_size, interpolation=cv2.INTER_LINEAR),
                    )
                    disparity_scale = original_size[0] / processing_size[0]
                    disp_crestereo = cv2.resize(disp_crestereo * disparity_scale, original_size, interpolation=cv2.INTER_NEAREST)
                    
                    cv2.imwrite(os.path.join(file_out_dir_left, f"{i:06d}.{args.file_type_intensity}"), left)
                    cv2.imwrite(os.path.join(file_out_dir_right, f"{i:06d}.{args.file_type_intensity}"), right)
                    cv2.imwrite(os.path.join(file_out_dir_disp_crestereo, f"{i:06d}.{args.file_type_disparity}"), disp_crestereo)
        except RectificationException:
            tqdm.write(f"failed to rectify file: {in_file}")


if __name__ == "__main__":
    main()
