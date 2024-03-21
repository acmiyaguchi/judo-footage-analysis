import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class RefereeExtraction(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def run(self):
        model = YOLO(self.checkpoint)
        # img = cv2.imread(self.input_path)
        output_root = ensure_parent(self.output().path).parent

        for p in Path(self.input_path).glob("*"):
            img = cv2.imread(p.as_posix())

            results = model.predict(
                img,
                save=False,
                conf=0.2,
                iou=0.5,
                verbose=False,
                stream=True,
                batch=6,
                device="cpu",
            )

            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cls = box.cls[0]
                    # cvzone.cornerRect(img, (x1, y1, w, h), colorR=colors[int(cls)])
                    start_pt = (x1, y1)
                    end_pt = (x2, y2)

                    if int(cls) == 2:
                        referee = img[y1 : y1 + h, x1 : x1 + w]
                        name = f"{p.stem}_{i:02d}.png"
                        cv2.imwrite((Path(self.output_path) / name).as_posix(), referee)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-root-path",
        type=str,
        default="/mnt/cs-share/pradalier/tmp/judo/frames/",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        default="/mnt/cs-share/pradalier/tmp/judo/referee/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/cs-share/pradalier/tmp/judo/models/entity_detection/v2/weights/best.pt",
    )
    parser.add_argument("--num-workers", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_root_path).glob("*/*"))
    # for p in image_batch_root:
    #     print(p)

    luigi.build(
        [
            RefereeExtraction(
                input_path=p.as_posix(),
                output_path=(
                    Path(args.output_root_path)
                    / p.relative_to(Path(args.input_root_path))
                ).as_posix(),
                checkpoint=args.checkpoint,
            )
            for p in image_batch_root
        ],
        workers=args.num_workers,
    )
