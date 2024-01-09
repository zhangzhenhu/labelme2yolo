# Labelme2YOLO

**Forked from [GreatV/Labelme2YOLO](https://github.com/GreatV/labelme2yolo)**


Labelme2YOLO is a powerful tool for converting LabelMe's JSON format to [YOLOv5](https://github.com/ultralytics/yolov5) dataset format. 
This tool can also be used for YOLOv5/YOLOv8 segmentation datasets, 
if you have already made your segmentation dataset with LabelMe, 
it is easy to use this tool to help convert to YOLO format dataset.



## New Features

本工具用于把 labelme 输出的 Json 格式标注样本转换成 YOLO 可用 TXT 格式样本。

从 GreatV/Labelme2YOLO Fork 而来，并在原版基础上做了一些改进和优化

* `--json_dir` 参数可以输入多个路径（支持通配符）
* 新增 `--output` 参数，可以指定输出路径
* 新增 `--rename` 参数，可以控制是否用uid重新命名文件。
* 新增 `--labels` 参数，可以选择 **只保留** 哪些标签。
* 新增 `--exclude_labels` 参数，可以选择 **去掉** 哪些标签。
* 去掉了某些情况下不必要的文件读取和拷贝，加快处理速度。
* 其它一些小的改进。


## Installation

```shell
pip install git+https://github.com/zhangzhenhu/labelme2yolo.git
```

## 使用方法

```shell
 labelme2yolo --json_dir all_jsons/  ../tee/*/*/jsons/   --val_size 0.15  --output mydataset/
 
```

## Arguments




```text

 -h, --help            show this help message and exit
  --json_dir JSON_DIR [JSON_DIR ...]
                        Please input the path of the labelme json files.
  --output OUTPUT       The output path.
  --val_size [VAL_SIZE]
                        Please input the validation dataset size, for example 0.1.
  --test_size [TEST_SIZE]
                        Please input the test dataset size, for example 0.1.
  --rename              是否用uuid重新命名文件名称。
  --output_format {bbox,polygon}
                        The default output format for labelme2yolo is "polygon". However, you can choose to output in bbox format by specifying the "bbox"
                        option.
  --labels LABELS [LABELS ...]
                        The labels you want to include, for example --labels cat dog
  --exclude_labels EXCLUDE_LABELS [EXCLUDE_LABELS ...]
                        Excluding labels.


```



## How to Use


## How to build package/wheel

1. [install hatch](https://hatch.pypa.io/latest/install/)
2. Run the following command:

```shell
hatch build
```

## 数据说明

### labelme 输出的json文件格式
```json5

{
  "version": "5.4.0.post1",
  "flags": {},
  "shapes": [
    {
      "label": "cat",  // 标签名称
      "points": [
        [   // 对于矩形框(rectangle) 这是矩形左上角的坐标
          153.0487804878049,  // 第1个点的x坐标
          141.46341463414637  // 第1个点的y坐标
        ],
        [  // 对于矩形框(rectangle) 这是矩形右下角的坐标
          749.0,  // 第2个点的x坐标
          780.0696294079506  // 第2个点的y坐标
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",  // 框的类型，有 circle、polygon 等
      "flags": {},
      "mask": null
    }
  ],
  "imagePath": "0a16af6c15815a7deb95a8ad4b787c13.jpg",
  "imageData": "图片本身的base64编码数据，可以依次还原成图片。",
  "imageHeight": 1000,
  "imageWidth": 750
}
```


### yolo 

## License

`labelme2yolo` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
