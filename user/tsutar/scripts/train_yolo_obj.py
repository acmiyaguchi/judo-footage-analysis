from ultralytics import YOLO

# import torch

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
# print(torch.cuda.devices())

# Train the model
results = model.train(
    data="/home/GPU/tsutar/home_gtl/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/scripts/yolo_train_config.yaml",
    epochs=100,
    imgsz=640,
    device=0,
)
