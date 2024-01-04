# SPDX-FileCopyrightText: 2022-present Wang Xin <xinwang614@gmail.com>
#
# SPDX-License-Identifier: MIT
"""
cli init
"""
import argparse

from labelme2yolo.l2y import Labelme2YOLO


def run():
    '''
    run cli
    '''
    parser = argparse.ArgumentParser("labelme2yolo")
    parser.add_argument(
        "--json_dir",
        type=str,
        nargs="+",
        help="Please input the path of the labelme json files.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str, help="The output path.",
        required=True,

    )
    parser.add_argument(
        "--val_size",
        type=float,
        nargs="?",
        default=0.2,
        help="Please input the validation dataset size, for example 0.1.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        nargs="?",
        default=0.0,
        help="Please input the test dataset size, for example 0.1.",
    )
    # parser.add_argument(
    #     "--json_name",
    #     type=str,
    #     nargs="?",
    #     default=None,
    #     help="If you put json name, it would convert only one json file to YOLO.",
    # )
    parser.add_argument(
        "--link",
        # type=str,
        action="store_true",
        default=False,
        help="Use a soft link for the image file to connect to the image source file instead of making a copy,"
             " which speeds up processing and saves disk space."
    )
    parser.add_argument(
        "--rename",
        # type=str,
        action="store_true",
        default=False,
        help="Rename the file name with uuid."
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["bbox", "polygon"],
        default="bbox",
        help='The default output format for labelme2yolo is "bbox[center_x,center_y,width,height]".'
             ' However, you can choose to output in polygon format [points list] by specifying the "polygon" option.',
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="The labels you want to include, for example, --labels cat dog",
        required=False,
    )
    parser.add_argument(
        "--exclude_labels",
        nargs="+",
        # action="store_true",
        # default=False,
        help="Excluding labels. for example, --exclude_labels cat dog"
    )

    args = parser.parse_args()

    if not args.json_dir:
        parser.print_help()
        return 0
    # print(args.json_dir)
    convertor = Labelme2YOLO(
        args.json_dir, args.output_format,
        include_labels=args.labels,
        save_dir=args.output,
        rename=args.rename,
        copy_image=not args.link,

    )

    # if args.json_name is None:
    convertor.convert(val_size=args.val_size, test_size=args.test_size)
    # else:
    #     convertor.convert_one(args.json_name)

    return 0
