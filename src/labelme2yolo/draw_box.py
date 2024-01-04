import cv2
import argparse
import requests
from pathlib import Path

colors = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (0, 199, 140),
    (192, 14, 235),
    (255, 255, 255),
    (0, 0, 128),
    (173, 255, 47),
    (8, 249, 73),
    (255, 0, 255),
    (255, 255, 0),
    (100, 100, 0),
    (0, 100, 255),
]


def read_image(image_url: str) -> bytearray:
    def download(url):
        response = requests.get(url)
        return response.content

    if image_url.startswith("http://") or image_url.startswith("https://"):
        image = download(image_url)
    else:
        with open(image_url, "rb") as fh:
            image = fh.read()
    return bytearray(image)


def _draw(image_file, labels: list):
    image = cv2.imread(image_file)
    print(image.shape)
    height,width = image.shape[0:2]
    # image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    for i, label in enumerate(labels):
        name = label["label"]

        print(label["location"])
        x0, y0, w, h = label["location"]
        x1 = x0 - w / 2
        x2 = x0 + w / 2
        y1 = y0 - h / 2
        y2 = y0 + h / 2
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        print(x1, y1, x2, y2)
        # 根据标签选择颜色
        color = colors[i % len(colors)]  # 默认为黑色

        # 绘制矩形框和标签
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 使用 color 作为颜色参数
        cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # 使用 color 作为颜色参数
    return image


def read_labels(file_path: str):
    labels = []
    with open(file_path) as fh:
        for line in fh:
            line = line.split()

            labels.append({"label": line.pop(0),
                           "location": [float(x) for x in line]
                           })
    return labels


def main(opts):
    labels = read_labels(opts.label_file)
    # image = read_image(opts.img_file)
    image = _draw(opts.img_file, labels)
    src_image_file = Path(opts.img_file)
    save_image_file = src_image_file.parent / (src_image_file.stem + "_bbox" + src_image_file.suffix)
    # save_image_file += "_bbox"
    cv2.imwrite(str(save_image_file), image)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', dest="img_file", type=str, help='输入文件')
    parser.add_argument('--label', dest="label_file", type=str, help='label txt file')

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_opt()
    opts.img_file = "/Users/zhangzhenhu/Downloads/public/0a8f1b19cd906c6404de6556ceffc13b.jpg"
    opts.label_file = "/Users/zhangzhenhu/Downloads/public/0a8f1b19cd906c6404de6556ceffc13b.txt"
    main(opts=opts)
