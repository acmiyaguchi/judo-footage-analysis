from ultralytics import YOLO

model = YOLO(
    "/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/scripts/runs/detect/train2/weights/best.pt"
)

model.export(format="onnx", device="cpu")
