import collections
from functools import lru_cache
import math
import os
import argparse
import cv2
import numpy as np
import av


class RectificationException(Exception):
    pass


default_args = dict(
    width=1920,
    height=1080,
    resize_factor=1,
    make_dimensions_divisible_by=8,
    calibration_frame_idx_dist=30,
    autoexposure_frames=90,
    recalibrate=True,
    recalibrate_iterative=True,
    use_deepmatching=False,
    use_cotr=True,
    show_matches=False,
    show_rectification=False,
    save_rectification=False,
    draw_epipolar_lines=False,
    grayscale=True,
    denoise=False,
    denoise_num_frames=5,
    denoise_noise_std=5,
    output_calibration_file=None,
    invert=False,
)


def rectify(**args):
    args = {**default_args, **args}
    input_file = args["input_file"]
    image_size = (args["width"], args["height"])
    resize_factor = args["resize_factor"]
    make_dimensions_divisible_by = args["make_dimensions_divisible_by"]
    calibration_frame_idx_dist = args["calibration_frame_idx_dist"]
    autoexposure_frames = args["autoexposure_frames"]
    recalibrate = args["recalibrate"]
    recalibrate_iterative = args["recalibrate"] and args["recalibrate_iterative"]
    use_deepmatching = args["use_deepmatching"]
    use_cotr = args["use_cotr"]
    show_matches = args["show_matches"]
    show_rectification = args["show_rectification"]
    draw_epipolar_lines = args["draw_epipolar_lines"]
    grayscale = args["grayscale"]
    denoise = args["denoise"]
    denoise_num_frames = args["denoise_num_frames"]
    denoise_noise_std = args["denoise_noise_std"]
    output_calibration_file = args["output_calibration_file"]
    invert = args["invert"]

    assert not (use_deepmatching and use_cotr)

    if autoexposure_frames < denoise_num_frames:
        raise ValueError("argument autoexposure_frames must be greater or equal to denoise_num_frames")

    if not use_deepmatching:
        sift = cv2.SIFT_create()
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})

    init_cam_mats_left = np.array([
        [1998.548313762794, 0.00000000e+00, 899.1958547906069],
        [0.00000000e+00, 1995.552255534815, 598.8278928199517],
        [0.00000000e+00, 0.00000000e+00, 1],
    ])
    init_cam_mats_right = np.array([
        [2007.1648145600445, 0.00000000e+00, 859.7953980050097],
        [0.00000000e+00, 2009.1782660647034, 613.4969001852309],
        [0.00000000e+00, 0.00000000e+00, 1],
    ])

    init_dist_coefs_left =  np.array([-0.4595421970863985, 0.2019752346541169, -0.0019327353310391964, 0.001957806937187838])
    init_dist_coefs_right = np.array([-0.473438679479971, 0.1967383195268051, -0.0032260744184233843, 0.005053066550145099])

    init_rot_mat = np.array([
        [ 0.99982747,  0.00547327,  0.01775028, -0.25437443 ],
        [-0.00613865,  0.99927213,  0.03765017,  0.0070481  ],
        [-0.01753129, -0.03775264,  0.99913332,  -0.01743795],
        [ 0.        ,  0.        ,  0.        ,  1.        ],
    ])

    init_rot_mat = np.array([
        [0.9997373615098061, 0.004595532919614159, 0.022451928170746975, -0.2509914052172423],
        [-0.005437145244579489, 0.9992792439499563, 0.037569004011037285, 0.003944074622135025],
        [-0.022263096212993057, -0.03768121133903117, 0.9990417813380091, 0.017989804386538424],
        [0.0, 0.0, 0.0, 1.0],
    ])

    init_trans_vec = init_rot_mat[:-1, -1]
    init_rot_mat = init_rot_mat[:-1, :-1]

    if recalibrate_iterative:
        rect_trans_left, rect_trans_right, proj_mats_left, proj_mats_right, disp_to_depth, valid_boxes_left, valid_boxes_right = cv2.stereoRectify(
            init_cam_mats_left,
            init_dist_coefs_left,
            init_cam_mats_right,
            init_dist_coefs_right,
            image_size,
            init_rot_mat,
            init_trans_vec,
            flags=0
        )

        undistort_map_left, rectify_map_left = cv2.initUndistortRectifyMap(
            init_cam_mats_left,
            init_dist_coefs_left,
            rect_trans_left,
            proj_mats_left,
            image_size,
            cv2.CV_32FC1
        )

        undistort_map_right, rectify_map_right = cv2.initUndistortRectifyMap(
            init_cam_mats_right,
            init_dist_coefs_right,
            rect_trans_right,
            proj_mats_right,
            image_size,
            cv2.CV_32FC1
        )

        min_x = max(valid_boxes_left[0], valid_boxes_right[0])
        min_y = max(valid_boxes_left[1], valid_boxes_right[1])
        max_x = min(valid_boxes_left[2], valid_boxes_right[2])
        max_y = min(valid_boxes_left[3], valid_boxes_right[3])
        valid_width = max_x - min_x
        valid_height = max_y - min_y

        if valid_width < image_size[0] / 2 or valid_height < image_size[1] / 2:
            raise RectificationException()

    if denoise:
        frame_buffer_left = collections.deque(maxlen=denoise_num_frames)
        frame_buffer_right = collections.deque(maxlen=denoise_num_frames)
        
        import torch
        from fastdvdnet.fastdvdnet import denoise_seq_fastdvdnet_custom
        from fastdvdnet.models import FastDVDnet

        fastdvdnet_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fastdvdnet_model = FastDVDnet(num_input_frames=denoise_num_frames)
        if torch.cuda.is_available():
            fastdvdnet_model = torch.nn.DataParallel(fastdvdnet_model, device_ids=[0]).cuda()
        else:
            raise RuntimeError("not implemented")
        fastdvdnet_model.load_state_dict(torch.load(os.path.join("fastdvdnet", "model.pth"), map_location=fastdvdnet_device))
        fastdvdnet_model.eval()


    def denoise_frame(left, right):
        if not denoise:
            return left, right

        for frame, frame_buffer in [(left, frame_buffer_left), (right, frame_buffer_right)]:
            frame_buffer.append(frame.transpose(2, 0, 1).astype(np.float32) / 255)

        if len(frame_buffer_left) < denoise_num_frames:
            return left, right

        with torch.no_grad():
            left = denoise_seq_fastdvdnet_custom(frame_buffer_left, denoise_noise_std / 255, fastdvdnet_model, fastdvdnet_device)
            right = denoise_seq_fastdvdnet_custom(frame_buffer_right, denoise_noise_std / 255, fastdvdnet_model, fastdvdnet_device)

        return left, right


    if recalibrate:
        container = av.open(input_file)
        pts1 = []
        pts2 = []
        for frame_i, frame in enumerate(container.decode(video=0)):

            array = frame.to_ndarray(format="rgb24")
            left = array[:, : array.shape[1] // 2, :]
            right = array[:, array.shape[1] // 2 :, :]
            if invert:
                left, right = right, left
            if resize_factor != 1:
                left = cv2.resize(left, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            if left.shape[0:2] != image_size[::-1] or right.shape[0:2] != image_size[::-1]:
                raise RectificationException()

            left, right = denoise_frame(left, right)

            if frame_i < autoexposure_frames or frame_i % calibration_frame_idx_dist != 0:
                continue

            if recalibrate_iterative:
                left = cv2.remap(left, undistort_map_left, rectify_map_left, cv2.INTER_CUBIC)
                right = cv2.remap(right, undistort_map_right, rectify_map_right, cv2.INTER_CUBIC)

            if use_deepmatching:
                from deepmatching_utils import deepmatching
                pts1, pts2, dm_score, dm_idx = deepmatching(
                    cv2.cvtColor(left, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(right, cv2.COLOR_RGB2GRAY),
                    resize_factor=1/2,
                )
                if show_matches:
                    show_correspondences(left, right, np.stack([pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1], dm_score, dm_idx], axis=0).transpose(1, 0))
                break
            elif use_cotr:
                from cotr_utils import get_cotr
                pts1, pts2 = get_cotr()(left, right)
                if show_matches:
                    from deepmatching_utils import show_correspondences
                    show_correspondences(left, right, np.stack([pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1], dm_score, dm_idx], axis=0).transpose(1, 0))
                break
            else:
                kp1, desc1 = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), None)
                kp2, desc2 = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), None)

                if desc1 is None or desc2 is None:
                    continue

                def match_is_valid(kp1, kp2, m, max_y_deviation=100, p=10):
                    (x1, y1) = kp1[m.queryIdx].pt
                    (x2, y2) = kp2[m.trainIdx].pt

                    return (
                        abs(y1 - y2) <= max_y_deviation and
                        x1 >= x2 and
                        (
                            not recalibrate_iterative or (
                                min_x + p < x1 < max_x - p and
                                min_x + p < x2 < max_x - p and
                                min_y + p < y1 < max_y - p and
                                min_y + p < y2 < max_y - p
                            )
                        )
                    )

                matches = flann.knnMatch(desc1, desc2, k=2)
                matchesMask = [[0,0] for i in range(len(matches))]
                for i, (m1, m2) in enumerate(matches):
                    if m1.distance < 0.7 * m2.distance and match_is_valid(kp1, kp2, m1):
                        matchesMask[i] = [1,0]
                        pt1 = kp1[m1.queryIdx].pt
                        pt2 = kp2[m1.trainIdx].pt
                        pts1.append(pt1)
                        pts2.append(pt2)

                
                if show_matches:
                    vis = cv2.drawMatchesKnn(left,kp1,right,kp2,matches,None, matchColor=(255, 0, 0), singlePointColor=(0,0,255), matchesMask=matchesMask, flags=0)
                    cv2.imshow("Matches", vis)
                    cv2.waitKey(1000//10)


        container.close()

        if len(pts1) < 8:
            raise RectificationException()

        _pts1 = np.array(pts1, dtype=np.float32)
        _pts2 = np.array(pts2, dtype=np.float32)

        f = np.mean([init_cam_mats_left[0, 0], init_cam_mats_left[1, 1], init_cam_mats_right[0, 0], init_cam_mats_right[1, 1]])
        if recalibrate_iterative:
            px = image_size[0] / 2
            py = image_size[1] / 2
            cam_mats_simple = np.array([
                [f, 0, px],
                [0, f, py],
                [0, 0, 1],
            ])
        else:
            px = np.mean([init_cam_mats_left[0, 2], init_cam_mats_right[0, 2]])
            py = np.mean([init_cam_mats_left[1, 2], init_cam_mats_right[1, 2]])
            cam_mats_simple = np.mean([init_cam_mats_left, init_cam_mats_right], axis=0)
        E, _ = cv2.findEssentialMat(_pts1,_pts2, f, (px, py), method=cv2.FM_RANSAC)
        _, R, t, _ = cv2.recoverPose(E, _pts1, _pts2, cam_mats_simple)
        t = t * init_trans_vec[0] / t[0]

        trans_vec = t

    if recalibrate and recalibrate_iterative:
        rot_mat = R @ init_rot_mat
        cam_mats_left = init_cam_mats_left
        cam_mats_right = init_cam_mats_right
    else:
        rot_mat = init_rot_mat
        trans_vec = init_trans_vec
        cam_mats_left = init_cam_mats_left if not recalibrate or recalibrate_iterative else cam_mats_simple
        cam_mats_right = init_cam_mats_right if not recalibrate or recalibrate_iterative else cam_mats_simple

    rect_trans_left, rect_trans_right, proj_mats_left, proj_mats_right, disp_to_depth, valid_boxes_left, valid_boxes_right = cv2.stereoRectify(
        cam_mats_left,
        init_dist_coefs_left,
        cam_mats_right,
        init_dist_coefs_right,
        image_size,
        rot_mat,
        trans_vec,
        flags=0
    )

    undistort_map_left, rectify_map_left = cv2.initUndistortRectifyMap(
        cam_mats_left,
        init_dist_coefs_left,
        rect_trans_left,
        proj_mats_left,
        image_size,
        cv2.CV_32FC1
    )

    undistort_map_right, rectify_map_right = cv2.initUndistortRectifyMap(
        cam_mats_right,
        init_dist_coefs_right,
        rect_trans_right,
        proj_mats_right,
        image_size,
        cv2.CV_32FC1
    )

    min_x = max(valid_boxes_left[0], valid_boxes_right[0])
    min_y = max(valid_boxes_left[1], valid_boxes_right[1])
    max_x = min(valid_boxes_left[2], valid_boxes_right[2])
    max_y = min(valid_boxes_left[3], valid_boxes_right[3])

    crop_width = (max_x - min_x) % make_dimensions_divisible_by
    crop_height = (max_y - min_y) % make_dimensions_divisible_by

    min_x += math.floor(crop_width / 2)
    max_x -= math.ceil(crop_width / 2)
    min_y += math.floor(crop_height / 2)
    max_y -= math.ceil(crop_height / 2)

    valid_width = max_x - min_x
    valid_height = max_y - min_y

    if valid_width < image_size[0] / 2 or valid_height < image_size[1] / 2:
        raise RectificationException()

    if output_calibration_file is not None:
        np.savez(
            output_calibration_file,

            rot_mat=rot_mat,
            init_rot_mat=init_rot_mat,
            trans_vec=trans_vec,
            init_trans_vec=init_trans_vec,
            cam_mats_left=cam_mats_left,
            init_cam_mats_left=init_cam_mats_left,
            cam_mats_right=cam_mats_right,
            init_cam_mats_right=init_cam_mats_right,
            undistort_map_left=undistort_map_left,
            rectify_map_left=rectify_map_left,
            undistort_map_right=undistort_map_right,
            rectify_map_right=rectify_map_right,
            init_dist_coefs_left=init_dist_coefs_left,
            init_dist_coefs_right=init_dist_coefs_right,
            image_size=image_size,
            disp_to_depth=disp_to_depth,
            valid_boxes_left=valid_boxes_left,
            valid_boxes_right=valid_boxes_right,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

    container = av.open(input_file)
    for frame_i, frame in enumerate(container.decode(video=0)):

        array = frame.to_ndarray(format="rgb24")
        left = array[:, : array.shape[1] // 2, :]
        right = array[:, array.shape[1] // 2 :, :]
        if invert:
            left, right = right, left
        if resize_factor != 1:
            left = cv2.resize(left, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        if left.shape[0:2] != image_size[::-1] or right.shape[0:2] != image_size[::-1]:
            raise RectificationException()

        left, right = denoise_frame(left, right)

        if frame_i < autoexposure_frames:
            continue

        if grayscale:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        left = cv2.remap(left, undistort_map_left, rectify_map_left, cv2.INTER_CUBIC)
        left = left[min_y:max_y, min_x:max_x]

        right = cv2.remap(right, undistort_map_right, rectify_map_right, cv2.INTER_CUBIC)
        right = right[min_y:max_y, min_x:max_x]

        if draw_epipolar_lines:
            for image in [left, right]:
                for y in np.linspace(0, left.shape[0], 20):
                    cv2.line(image, (0, int(y)), (left.shape[1], int(y)), (0, 255, 0), thickness=2)

        if show_rectification:
            cv2.imshow("Rectified", np.hstack([left, right]))
            cv2.waitKey(1000//30)

        yield (frame_i, left, right)

    container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rectify stereo image sequences from CSI cameras")
    parser.add_argument("--width", type=int, help="original (single image) width")
    parser.add_argument("--height", type=int, help="original (single image) height")
    parser.add_argument("--resize_factor", type=float, help="resize factor at which the original calibration was performed")
    parser.add_argument("--make_dimensions_divisible_by", type=int, help="make output dimensions divisible by this constant (needed for some video codecs")
    parser.add_argument("--calibration_frame_idx_dist", type=int, help="interval at which to take iterative recalibration frames")
    parser.add_argument("--autoexposure_frames", type=int, help="number of frames to skip due to autoexposure")
    parser.add_argument("--recalibrate", dest="recalibrate", action="store_true")
    parser.add_argument("--no-recalibrate", dest="recalibrate", action="store_false")
    parser.add_argument("--recalibrate_iterative", dest="recalibrate_iterative", action="store_true")
    parser.add_argument("--no-recalibrate_iterative", dest="recalibrate_iterative", action="store_false")
    parser.add_argument("--use_deepmatching", dest="use_deepmatching", action="store_true")
    parser.add_argument("--no-use_deepmatching", dest="use_deepmatching", action="store_false")
    parser.add_argument("--use_cotr", dest="use_cotr", action="store_true")
    parser.add_argument("--no-use_cotr", dest="use_cotr", action="store_false")
    parser.add_argument("--show_matches", dest="show_matches", action="store_true")
    parser.add_argument("--no-show_matches", dest="show_matches", action="store_false")
    parser.add_argument("--show_rectification", dest="show_rectification", action="store_true")
    parser.add_argument("--no-show_rectification", dest="show_rectification", action="store_false")
    parser.add_argument("--save_rectification", dest="save_rectification", action="store_true")
    parser.add_argument("--no-save_rectification", dest="save_rectification", action="store_false")
    parser.add_argument("--draw_epipolar_lines", dest="draw_epipolar_lines", action="store_true")
    parser.add_argument("--no-draw_epipolar_lines", dest="draw_epipolar_lines", action="store_false")
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    parser.add_argument("--denoise", dest="denoise", action="store_true")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument("--denoise_num_frames", type=int, help="number of frames to use at once for denoising")
    parser.add_argument("--denoise_noise_std", type=int, help="standard deviation of expeced noise")
    parser.add_argument("--output_calibration_file", type=str, help="write calibration parameters to this .npz file")
    parser.add_argument("input_file", type=str, help="input mkv file")
    parser.add_argument("output_dir", type=str, help="output directory containing left and right subdirectories")
    parser.add_argument("--invert", dest="invert", action="store_true")
    parser.add_argument("--no-invert", dest="invert", action="store_false")
    parser.set_defaults(**default_args)
    args = parser.parse_args()

    if args.save_rectification:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "right"), exist_ok=True)

    for frame_i, (left, right) in enumerate(rectify(**vars(args))):
        if args.save_rectification:
            cv2.imwrite(os.path.join(args.output_dir, "left", f"{frame_i:06d}.png"), left)
            cv2.imwrite(os.path.join(args.output_dir, "right", f"{frame_i:06d}.png"), right)
