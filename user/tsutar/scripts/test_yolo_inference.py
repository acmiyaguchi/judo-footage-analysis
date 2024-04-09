import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import luigi
from ultralytics import YOLO

model_name = "/cs-share/pradalier/tmp/judo/models/referee_pose/v1/weights/03-04-2024-referee-best.pt"
yolo_original = (
    "/home/GTL/tsutar/intro_to_ress8813-judo-footage-analysis/user/tsutar/yolov8n.pt"
)
model = YOLO(model_name)

input_path = "/cs-share/pradalier/tmp/judo/data/referee_v2/mat_01/0000/0398_00.png"

results = model(input_path, verbose=False)

result_dict = []
# Process results list
for result in results:
    print(result.probs)
