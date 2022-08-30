import argparse
import os
import glob
from tqdm.auto import tqdm
import cv2
import av
from crestereo import CREStereo
from rectify import rectify


def main(video_dir, output_dir, sampling_interval):
    assert os.path.isdir(output_dir)
    processing_size = (1280, 720)
    crestereo = CREStereo(
        f'weights/crestereo/crestereo_combined_iter5_{processing_size[1]}x{processing_size[0]}.onnx',
    )
    camera_focal_length = 0.006  # meters
    camera_baseline = 0.25  # meters
    camera_horizontal_pixels = 1920  # pixels
    camera_horizontal_sensor_size = 0.007564  # meters
    camera_disparity_to_depth_factor = camera_focal_length * camera_baseline * camera_horizontal_pixels * camera_horizontal_sensor_size ** -1

    for detection_video in tqdm(sorted(glob.glob(os.path.join(video_dir, "*.mkv")))):
        detection_id = os.path.splitext(os.path.basename(detection_video))[0]
        os.makedirs(os.path.join(output_dir, "detection_frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "detection_frames_depth"), exist_ok=True)

        with av.open(detection_video) as container:
            fps = container.streams.video[0].average_rate

        last_snapshot = -sampling_interval

        for i, left, right in rectify(input_file=detection_video, recalibrate=False, recalibrate_iterative=False):
            current_time = (i / fps)
            if current_time >= last_snapshot + sampling_interval:
                last_snapshot = last_snapshot + sampling_interval

                original_size = (left.shape[1], left.shape[0])
                disp_crestereo = crestereo(
                    cv2.resize(left, processing_size, interpolation=cv2.INTER_LINEAR),
                    cv2.resize(right, processing_size, interpolation=cv2.INTER_LINEAR),
                )
                disparity_scale = original_size[0] / processing_size[0]
                disp_crestereo = cv2.resize(disp_crestereo * disparity_scale, original_size, interpolation=cv2.INTER_NEAREST)
                depth_crestereo = camera_disparity_to_depth_factor / disp_crestereo

                cv2.imwrite(os.path.join(output_dir, "detection_frames", f"{detection_id}_{last_snapshot:06d}.png"), left)
                cv2.imwrite(os.path.join(output_dir, "detection_frames_depth", f"{detection_id}_{last_snapshot:06d}.exr"), depth_crestereo)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="sample frames from side-by-side video in a regular interval for camera trap distance sampling")
    argparser.add_argument("video_dir", type=str, help="path to directory containing side-by-side videos")
    argparser.add_argument("output_dir", type=str, help="path to directory where the output frames should be placed")
    argparser.add_argument("--sampling_interval", type=float, default=2., help="sampling interval in seconds")
    args = argparser.parse_args()
    main(args.video_dir, args.output_dir, args.sampling_interval)