import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import luigi
from ultralytics import YOLO

model_name = "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/scripts/runs/detect/train2/weights/best.pt"
yolo_original = (
    "/home/GTL/tsutar/intro_to_ress8813-judo-footage-analysis/user/tsutar/yolov8n.pt"
)
model = YOLO(model_name)

input_path_shared = "/cs-share/pradalier/tmp/judo/frames/mat_01/0001/0001.jpg"
input_path_local = "/home/GTL/tsutar/intro_to_res/judo/frames/mat_01/0000/0000.jpg"

results = model(input_path_shared, stream=False, verbose=False)
print(results)


# for r in result_dict:
#     print(r)
