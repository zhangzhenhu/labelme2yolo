"""
Created on Aug 18, 2021

@author: xiaosonh
@author: GreatV(Wang Xin)
"""
from typing import List, Optional
import base64
import glob
import io
import json
import math
import os
import random
import shutil
import uuid
import logging

import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import cv2
import numpy as np
import tqdm
from pathlib import Path

# set seed
random.seed(12345678)
random.Random().seed(12345678)
np.random.seed(12345678)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("labelme2yolo")


def train_test_split(dataset_index, test_size=0.2):
    """Split dataset into train set and test set with test_size"""
    test_size = min(max(0.0, test_size), 1.0)
    total_size = len(dataset_index)
    train_size = int(math.ceil(total_size * (1.0 - test_size)))
    random.shuffle(dataset_index)
    train_index = dataset_index[:train_size]
    test_index = dataset_index[train_size:]

    return train_index, test_index


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_pil(img_data):
    """Convert img_data(byte) to PIL.Image"""
    file = io.BytesIO()
    file.write(img_data)
    img_pil = PIL.Image.open(file)
    return img_pil


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_arr(img_data):
    """Convert img_data(byte) to numpy.ndarray"""
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_b64_to_arr(img_b64):
    """Convert img_b64(str) to numpy.ndarray"""
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_pil_to_data(img_pil):
    """Convert PIL.Image to img_data(byte)"""
    file = io.BytesIO()
    img_pil.save(file, format="PNG")
    img_data = file.getvalue()
    return img_data


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_arr_to_b64(img_arr):
    """Convert numpy.ndarray to img_b64(str)"""
    img_pil = PIL.Image.fromarray(img_arr)
    file = io.BytesIO()
    img_pil.save(file, format="PNG")
    img_bin = file.getvalue()
    img_b64 = base64.encodebytes(img_bin)
    return img_b64


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_png_data(img_data):
    """Convert img_data(byte) to png_data(byte)"""
    with io.BytesIO() as f_out:
        f_out.write(img_data)
        img = PIL.Image.open(f_out)

        with io.BytesIO() as f_in:
            img.save(f_in, "PNG")
            f_in.seek(0)
            return f_in.read()


def extend_point_list(point_list, out_format="polygon"):
    """Extend point list to polygon or bbox"""
    x_min = min(float(point) for point in point_list[::2])
    x_max = max(float(point) for point in point_list[::2])
    y_min = min(float(point) for point in point_list[1::2])
    y_max = max(float(point) for point in point_list[1::2])

    if out_format == "bbox":
        x_i = x_min
        y_i = y_min
        w_i = x_max - x_min
        h_i = y_max - y_min
        x_i = x_i + w_i / 2
        y_i = y_i + h_i / 2
        return np.array([x_i, y_i, w_i, h_i])

    return np.array([x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max])


def save_yolo_label(obj_list, text_file_path):
    """Save yolo label to txt file"""
    # txt_path = os.path.join(label_dir, target_dir, target_name)

    with open(text_file_path, "w+", encoding="utf-8") as file:
        for label, points in obj_list:
            points = [str(item) for item in points]
            line = f"{label} {' '.join(points)}\n"
            file.write(line)


def save_yolo_image(json_data, json_file, save_path, copy=True, rename=False):
    """Save yolo image to image_dir_path/target_dir"""
    json_dir = Path(json_file).parent
    image_save_path = Path(save_path)

    # image_file
    if json_data["imageData"] is None:
        # src_file = Path()
        if "\\" in json_data["imagePath"]:
            json_data["imagePath"] = json_data["imagePath"].replace("\\", "/")
        src_path = json_dir.joinpath(json_data["imagePath"])
        # 复制一份图片
        if copy:
            if rename:
                filename: str = uuid.UUID(int=random.Random().getrandbits(128)).hex
                # 重新命名
                image_file_path = image_save_path / (filename + src_path.suffix)  # Path(image_save_path)
            else:
                # 保留原始文件名
                image_file_path = image_save_path / src_path.name
            shutil.copy(src_path, image_file_path)
            return image_file_path
        else:
            return src_path

        # src_image = cv2.imread(src_image_name)
        # cv2.imwrite(image_save_path, src_image)
    else:
        img = img_b64_to_arr(json_data["imageData"])
        filename: str = uuid.UUID(int=random.Random().getrandbits(128)).hex
        image_file_path = image_save_path / (filename + ".png")
        PIL.Image.fromarray(img).save(image_save_path, formate="png")
        return image_file_path


class Labelme2YOLO:
    """Labelme to YOLO format converter"""

    def __init__(self, json_dirs: List,
                 output_format,
                 save_dir,
                 copy_image=True, rename=False,
                 include_labels: Optional[List] = None,
                 exclude_labels: Optional[List] = None,
                 ):
        self.json_dirs = json_dirs
        # self._json_dir = os.path.expanduser(json_dir)
        self._output_format = output_format
        self._label_list = []
        self._label_id_map = {}
        self._label_dir_path = ""
        self.copy_image = copy_image
        self.rename = rename
        self.save_dir = save_dir
        self._image_dir_path = os.path.join(self.save_dir, "images")
        if include_labels:
            self._label_list = include_labels
            self._label_id_map = {
                label: label_id for label_id, label in enumerate(include_labels)
            }
            self.frozen_labels = True
        else:
            self._label_list = []
            self._label_id_map = {}
            self.frozen_labels = False

        self.exclude_labels = set(exclude_labels) if exclude_labels else {}

    # def _update_id_map(self, label: str):
    #     if label not in self._label_list:
    #         self._label_list.append(label)
    #         self._label_id_map[label] = len(self._label_id_map)
    def get_label_id(self, label: str):
        if label in self.exclude_labels:
            return None
        if label not in self._label_id_map and self.frozen_labels:
            return None
        if label in self._label_id_map:
            return self._label_id_map[label]

        label_id = len(self._label_id_map)
        self._label_id_map[label] = label_id
        self._label_list.append(label)

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self.save_dir, "labels")
        self._image_dir_path = os.path.join(self.save_dir, "images")

        for yolo_path in (
                os.path.join(self._label_dir_path, "train"),
                os.path.join(self._label_dir_path, "val"),
                os.path.join(self._label_dir_path, "test"),
                os.path.join(self._image_dir_path, "train"),
                os.path.join(self._image_dir_path, "val"),
                os.path.join(self._image_dir_path, "test"),
        ):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)

            os.makedirs(yolo_path)

    # def _get_dataset_part_json_names(self, dataset_part: str):
    #     """Get json names in dataset_part folder"""
    #     set_folder = os.path.join(self._json_dir, dataset_part)
    #     json_names = []
    #     for sample_name in os.listdir(set_folder):
    #         set_dir = os.path.join(set_folder, sample_name)
    #         if os.path.isdir(set_dir):
    #             json_names.append(sample_name + ".json")
    #     return json_names

    def _train_test_split(self, json_names, val_size, test_size):
        """Split json names to train, val, test"""
        total_size = len(json_names)
        dataset_index = list(range(total_size))
        train_ids, val_ids = train_test_split(dataset_index, test_size=val_size)
        test_ids = []
        if test_size is None:
            test_size = 0.0
        if test_size > 1e-8:
            train_ids, test_ids = train_test_split(
                train_ids, test_size=test_size / (1 - val_size)
            )
        train_json_names = [json_names[train_idx] for train_idx in train_ids]
        val_json_names = [json_names[val_idx] for val_idx in val_ids]
        test_json_names = [json_names[test_idx] for test_idx in test_ids]

        return train_json_names, val_json_names, test_json_names

    # def conver(self):
    def convert(self, val_size=None, test_size=None):
        """Convert labelme format to yolo format"""
        json_files = []
        for json_dir in self.json_dirs:
            if Path(json_dir).is_file():
                json_files.append(json_dir)
                continue
            _json_names = glob.glob(
                os.path.join(json_dir, "**", "*.json"), recursive=True
            )
            json_files.extend(_json_names)
        # json_names = sorted(json_names)
        if val_size or test_size:
            train_json_names, val_json_names, test_json_names = self._train_test_split(
                json_files, val_size, test_size
            )
            groups = ("train", "val", "test")
            names = (train_json_names, val_json_names, test_json_names)
        else:
            groups = ("train",)
            names = (json_files,)

        self._label_dir_path = os.path.join(self.save_dir, "labels")
        self._image_dir_path = os.path.join(self.save_dir, "images")

        # self._make_train_val_dir()
        for group_name in groups:
            img_save_path = os.path.join(self._image_dir_path, group_name)
            label_save_path = os.path.join(self._label_dir_path, group_name)
            os.makedirs(img_save_path, exist_ok=True)
            os.makedirs(label_save_path, exist_ok=True)
        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder

        for group_name, json_files in zip(groups, names):
            # target_part = target_dir.replace("/", "")
            # img_save_path = os.path.join(self._image_dir_path, group_name)
            # label_save_path = os.path.join(self._label_dir_path, group_name)
            # logger.info("Converting %s set ...", target_dir)
            for json_file in tqdm.tqdm(json_files, total=len(json_files)):
                self.covert_json_to_text(group_name, json_file)

        self._save_dataset_yaml()

    def covert_json_to_text(self, group_name, json_file):
        """Convert json file to yolo format text file and save them to files"""
        # json_file = Path(json_file)
        with open(json_file, encoding="utf-8") as file:
            json_data = json.load(file)

        # filename: str = uuid.UUID(int=random.Random().getrandbits(128)).hex
        # image_name = f"{filename}.png"
        # label_name = f"{filename}.txt"
        img_save_path = os.path.join(self._image_dir_path, group_name)
        label_save_path = os.path.join(self._label_dir_path, group_name)

        img_path = save_yolo_image(
            json_data,
            json_file=json_file,
            save_path=img_save_path,
            copy=self.copy_image, rename=self.rename
        )

        yolo_obj_list = self._get_yolo_object_list(json_data, str(img_path))
        label_file_path = os.path.join(label_save_path, img_path.stem + ".txt")
        save_yolo_label(yolo_obj_list, label_file_path)

    # def convert_one(self, json_name):
    #     """Convert one json file to yolo format text file and save them to files"""
    #     json_path = os.path.join(self._json_dir, json_name)
    #     with open(json_path, encoding="utf-8") as file:
    #         json_data = json.load(file)
    #
    #     # image_name = json_name.replace(".json", ".png")
    #     # label_name = json_name.replace(".json", ".txt")
    #     # img_path = os.path.join(self._image_dir_path, image_name)
    #     # label_path = os.path.join(self._image_dir_path, label_name)
    #
    #     img_path = save_yolo_image(
    #         json_data,
    #         json_dir=self._json_dir,
    #         save_path=self._image_dir_path,
    #         copy=self.copy_image,
    #         rename=self.rename
    #     )
    #     yolo_obj_list = self._get_yolo_object_list(json_data, str(img_path))
    #     label_file_path = os.path.join(self._label_dir_path, img_path.stem + ".txt")
    #     save_yolo_label(yolo_obj_list, label_file_path)
    #
    #     # yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
    #     # save_yolo_label(yolo_obj_list, text_save_path=label_path)

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        img_h, img_w = json_data.get("imageHeight"), json_data.get("imageWidth")
        if not img_h or not img_w:
            img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data["shapes"]:
            if not shape["label"]:
                continue
            label = shape["label"]
            label_id = self.get_label_id(label)
            if not label_id:
                continue
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape["shape_type"] == "circle":
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            elif shape["shape_type"] == "rectangle":
                yolo_obj = self._get_rectangle_shape_yolo_object(shape, img_h, img_w)
            else:
                logger.warning(f"Not support object shape {shape['shape_type']}")
                continue
            if yolo_obj:
                yolo_obj_list.append((label_id, list(yolo_obj)))

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):

        obj_center_x, obj_center_y = shape["points"][0]

        radius = math.sqrt(
            (obj_center_x - shape["points"][1][0]) ** 2
            + (obj_center_y - shape["points"][1][1]) ** 2
        )
        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        # if shape["label"]:
        #     label = shape["label"]
        #     # if label not in self._label_list:
        #     #     self._update_id_map(label)
        #     # label_id = self._label_id_map[shape["label"]]
        #     label_id = self.get_label_id(label)
        return yolo_center_x, yolo_center_y, yolo_w, yolo_h

        # return None

    def _get_rectangle_shape_yolo_object(self, shape, img_h, img_w):
        point_list = shape["points"]
        points = np.zeros(2 * len(point_list))
        # 负值修正
        points[::2] = [min(float(point[0]), 0) / img_w for point in point_list]
        points[1::2] = [min(float(point[1]), 0) / img_h for point in point_list]

        if len(points) == 4:
            if self._output_format == "polygon":
                points = extend_point_list(points)
            if self._output_format == "bbox":
                points = extend_point_list(points, "bbox")
        else:
            return None
        # if shape["label"]:
        #     label = shape["label"]
        #     if label not in self._label_list:
        #         self._update_id_map(label)
        #     label_id = self._label_id_map[shape["label"]]

        return points.tolist()

        # return None

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self.save_dir, "dataset.yaml")
        save_path = Path(self.save_dir).absolute()
        with open(yaml_path, "w+", encoding="utf-8") as yaml_file:
            train_dir = os.path.join(self.save_dir, "images", "train")
            val_dir = os.path.join(self.save_dir, "images", "val")
            test_dir = os.path.join(self.save_dir, "images", "test")

            names_str = ""
            for label, _ in self._label_id_map.items():
                names_str += f'"{label}", '
            names_str = names_str.rstrip(", ")

            content = (

                f"train: {train_dir}\nval: {val_dir}\ntest: {test_dir}\n"
                f"nc: {len(self._label_id_map)}\n"
                f"names: [{names_str}]"
            )

            yaml_file.write(content)
