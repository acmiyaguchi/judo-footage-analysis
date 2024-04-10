from ultralytics import YOLO

# import torch

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
# print(torch.cuda.devices())

# Train the model
results = model.train(
    data="/home/GTL/tsutar/intro_to_res/referee_classification_dataset/",
    epochs=100,
    imgsz=640,
    device="cpu",
    patience=10,
)
