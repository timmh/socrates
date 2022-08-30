import argparse
import rospy
import rosbag
from cv_bridge import CvBridge
import cv2


def video_iterator(video_paths, sampling_interval):
    timestamp_offset = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        prop_fps = cap.get(cv2.CAP_PROP_FPS)
        if prop_fps != prop_fps or prop_fps <= 1e-2:
            print("Warning: can't get FPS. Assuming 30.")
            prop_fps = 30
        ret = True
        frame_id = 0
        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = 100 * float(frame_id) / prop_fps

            if timestamp_offset != 0:
                timestamp += timestamp_offset

            if frame_id % sampling_interval == 0:
                yield frame, timestamp

            frame_id += 1
        
        timestamp_offset = timestamp + 1

        cap.release()


def main(csi_input, bag_output, resize_factor, flip_horizontal, flip_vertical, sampling_interval):
    bag = rosbag.Bag(bag_output, "w")
    cv_bridge = CvBridge()
    
    for frame, timestamp in video_iterator([csi_input], sampling_interval):
        stamp = rospy.rostime.Time.from_sec(timestamp)
        frames = {
            "left": frame[:, :frame.shape[1] // 2, :],
            "right": frame[:, frame.shape[1] // 2 :, :],
        } if not flip_horizontal else {
            "left": frame[:, frame.shape[1] // 2 :, :],
            "right": frame[:, : frame.shape[1] // 2, :],
        }

        if flip_vertical:
            for k in frames.keys():
                frames[k] = frames[k][::-1, ...].copy()

        for k in frames.keys():
            frame = frames[k]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize_factor != 1:
                frame = cv2.resize(
                    frame,
                    dsize=None,
                    fx=resize_factor,
                    fy=resize_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
            image = cv_bridge.cv2_to_imgmsg(frame)
            image.header.stamp = stamp
            image.header.frame_id = "camera"
            topics = {"left": "camera/left_raw", "right": "camera/right_raw"}
            bag.write(topics[k], image, stamp)
    bag.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("csi_input", type=str, help="the input CSI side-by-side video file")
    argparser.add_argument("bag_output", type=str, help="where to write the output bag file to")
    argparser.add_argument("--resize_factor", type=float, default=1., help="by which factor the image file size should be reduced")
    argparser.add_argument("--flip_horizontal", action="store_true", help="whether to flip the input video horizontally")
    argparser.add_argument("--flip_vertical", action="store_true", help="whether to flip the input video vertically")
    argparser.add_argument("--sampling_interval", type=int, default=15, help="at which interval the input video should be sampled")
    args = argparser.parse_args()

    main(args.csi_input, args.bag_output, args.resize_factor, args.flip_horizontal, args.flip_vertical, args.sampling_interval)