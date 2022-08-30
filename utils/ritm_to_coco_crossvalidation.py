import argparse
import copy
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

def list_split_random(list_to_split, ratio):
    elements = len(list_to_split)
    list_to_split = random.sample(list_to_split, elements)
    middle = max(1, int(elements * ratio))
    return [list_to_split[:middle], list_to_split[middle:]]

def partition_list(list_to_split, k):
    elements = len(list_to_split)
    list_to_split = random.sample(list_to_split, elements)
    chunksize = elements // k
    result = []
    for i in range(k):
        result += [list_to_split[i * chunksize:((i + 1) * chunksize) if i != (k - 1) else elements]]
    return result

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

    splits = {"train": 0.8, "test": 0.2}
    folds = {"train": 10, "test": 0}
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

    data = {}
    default_data_entry = dict(
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

    for split_name in tqdm(splits.keys()):

        video_paths = splits_video_paths[split_name]

        if folds[split_name] > 0:
            fold_splits = partition_list(video_paths, folds[split_name])

        for fold in range(max(1, folds[split_name])):

            if folds[split_name] > 0:
                fold_train_video_paths, fold_test_video_paths = [], []
                for i in range(folds[split_name]):
                    if i == fold:
                        fold_test_video_paths += fold_splits[i]
                    else:
                        fold_train_video_paths += fold_splits[i]
                video_path_collections = [fold_train_video_paths, fold_test_video_paths]
                video_path_collections_names = ["train", "val"]
            else:
                video_path_collections = [video_paths]
                video_path_collections_names = [split_name]

            for video_path_collection, video_path_collections_name in zip(video_path_collections, video_path_collections_names):
                current_data_key = f"{split_name}_fold_{fold:03d}_{video_path_collections_name}" if folds[split_name] > 0 else split_name
                data[current_data_key] = copy.deepcopy(default_data_entry)
                current_data = data[current_data_key]
                for video_path in video_path_collection:
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

                        current_data["images"].append(dict(
                            id=len(current_data["images"]) + 1,
                            file_name=os.path.relpath(left_image_destination_path, output_dir_images),
                            height=int(mask_img.shape[0]),
                            width=int(mask_img.shape[1]),
                        ))

                        if not os.path.exists(left_image_destination_path) or not os.path.exists(disp_image_destination_path):
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

                            current_data["annotations"].append(dict(
                                segmentation=segmentation,
                                area=int((x_max - x_min) * (y_max - y_min)),
                                bbox=[int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                                iscrowd=0,
                                image_id=current_data["images"][-1]["id"],
                                category_id=1,
                                id=len(current_data["annotations"]) + 1,
                            ))

    for key in data.keys():
        with open(os.path.join(args.output_dir, f"{key}.json"), "w") as f:
            json.dump(data[key], f)

if __name__ == "__main__":
    main()