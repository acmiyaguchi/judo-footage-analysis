import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import luigi
from ultralytics import YOLO

model_name = "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/runs/classify/train10/weights/last.pt"
yolo_original = (
    "/home/GTL/tsutar/intro_to_ress8813-judo-footage-analysis/user/tsutar/yolov8n.pt"
)
model = YOLO(model_name)

input_path = "/mnt/cs-share/pradalier/tmp/judo/frames/mat_01/0001/0001.jpg"

results = model(input_path, stream=False, verbose=False)

result_dict = []
# Process results list
for result in results:
    path = result.path
    names = result.names
    probs = result.probs.cpu().numpy()
    class_probs = probs.data  # Probs object for classification outputs
    class_list = class_probs.tolist()
    print(type(class_list))
    names["path"] = path
    names["prob"] = class_list
    result_dict.append(names)

with open("output.json", "w") as outfile:
    for r in result_dict:
        json.dump(r, outfile)

# for r in result_dict:
#     print(r)
