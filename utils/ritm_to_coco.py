import argparse
import json
import os
import glob
import shutil
import random
from math import floor, ceil
import numpy as np
import cv2
from tqdm.auto import tqdm


def mask2bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_dir", type=str)
    argparser.add_argument("output_dir", type=str)
    args = argparser.parse_args()

    assert os.path.isdir(args.input_dir)
    assert not os.path.exists(args.output_dir)
    
    intensity_image_extension_list = [".jpg", ".png"]
    output_dir_images = os.path.join(args.output_dir, "images")
    os.makedirs(output_dir_images)

    random.seed(42)

    video_paths = [
        video_path for video_path in sorted(glob.glob(os.path.join(args.input_dir, "*/")))
        if len(glob.glob(os.path.join(video_path, "masks", "*.png"))) > 0
    ]
    random.shuffle(video_paths)

    splits = {"train": 0.8, "val": 0.2}
    assert sum(splits.values()) == 1

    splits_video_paths = {}
    cummulative_ratio = 0
    for split_name, split_ratio in splits.items():
        splits_video_paths[split_name] = video_paths[
            floor(cummulative_ratio * len(video_paths)):
            floor((cummulative_ratio + split_ratio) * len(video_paths))
        ]
        cummulative_ratio += split_ratio

    assert sum([len(e) for e in splits_video_paths.values()]) == len(video_paths)

    data = {
        split_name: dict(
            info=dict(),
            licences=dict(),
            images=[],
            annotations=[],
            categories=[
                dict(
                    name="animal",
                    id=1,
                    supercategory=None,
                )
            ],
        )
        for split_name in splits.keys()
    }

    for split_name in tqdm(splits.keys()):
        for video_path in splits_video_paths[split_name]:
            video_id = os.path.basename(os.path.normpath(video_path))
            for mask_path in sorted(glob.glob(os.path.join(video_path, "masks", "*.png"))):
                mask_img = cv2.imread(mask_path)
                image_id = os.path.splitext(os.path.basename(mask_path))[0]

                left_image_source_paths = [
                    os.path.join(video_path, "left", f"{image_id}{ext}")
                    for ext in intensity_image_extension_list
                    if os.path.exists(os.path.join(video_path, "left", f"{image_id}{ext}"))
                ]
                assert len(left_image_source_paths) == 1
                left_image_source_path = left_image_source_paths[0]
                disp_image_source_path = os.path.join(video_path, "disp_crestereo", f"{image_id}.exr")

                left_image_destination_path = os.path.join(output_dir_images, f"{video_id}_{os.path.basename(left_image_source_path)}")
                disp_image_destination_path = os.path.join(output_dir_images, f"{video_id}_{image_id}.exr")

                data[split_name]["images"].append(dict(
                    id=len(data[split_name]["images"]) + 1,
                    file_name=os.path.relpath(left_image_destination_path, output_dir_images),
                    height=int(mask_img.shape[0]),
                    width=int(mask_img.shape[1]),
                ))

                assert not os.path.exists(left_image_destination_path) and not os.path.exists(disp_image_destination_path)
                shutil.copyfile(left_image_source_path, left_image_destination_path)
                shutil.copyfile(disp_image_source_path, disp_image_destination_path)

                for color in np.unique(mask_img.reshape(-1, mask_img.shape[2]), axis=0):
                    if (color == [0, 0, 0]).all():
                        continue
                    mask = (mask_img == color[None, None, :]).all(axis=-1)

                    ret = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = ret[1] if len(ret) == 3 else ret[0]
                    segmentation = []
                    for contour in contours:
                        contour = contour.flatten().tolist()
                        if len(contour) > 4:
                            segmentation.append(contour)
                    if len(segmentation) == 0:
                        continue

                    # TODO: this would be more efficient but somehow does not work with mmdetection
                    # import pycocotools.mask as cocomask
                    # rles = cocomask.encode(np.asfortranarray(mask))
                    # for rle in rles if type(rles) == list else [rles]:
                    #     rle["counts"] = rle["counts"].decode("ascii")
                    # segmentation = rles

                    y_min, y_max, x_min, x_max = mask2bbox(mask)

                    data[split_name]["annotations"].append(dict(
                        segmentation=segmentation,
                        area=int((x_max - x_min) * (y_max - y_min)),
                        bbox=[int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                        iscrowd=0,
                        image_id=data[split_name]["images"][-1]["id"],
                        category_id=1,
                        id=len(data[split_name]["annotations"]) + 1,
                    ))

    for split_name in splits.keys():
        with open(os.path.join(args.output_dir, f"{split_name}.json"), "w") as f:
            json.dump(data[split_name], f)

if __name__ == "__main__":
    main()