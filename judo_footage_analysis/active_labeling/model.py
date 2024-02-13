from label_studio_ml.model import LabelStudioMLBase
import requests
from ultralytics import YOLO
from PIL import Image
from io import BytesIO


class YOLOv8Model(LabelStudioMLBase):
    def __init__(
        self,
        base_url="http://localhost:8080",
        api_token="",
        model_name="yolov8n.pt",
        **kwargs,
    ):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = ["Edible", "Inedible", "Visual defects"]
        self.model = YOLO(model_name)
        self.base_url = base_url
        self.api_token = api_token

    def predict(self, tasks, **kwargs):
        """This is where inference happens: model returns
        the list of predictions based on input list of tasks
        """
        task = tasks[0]

        predictions = []
        score = 0

        header = {"Authorization": f"Token {self.api_token}"}
        image = Image.open(
            BytesIO(
                requests.get(
                    self.base_url + task["data"]["image"], headers=header
                ).content
            )
        )
        original_width, original_height = image.size
        results = self.model.predict(image)

        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append(
                    {
                        "id": str(i),
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": prediction.conf.item(),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": xyxy[0] / original_width * 100,
                            "y": xyxy[1] / original_height * 100,
                            "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                            "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                            "rectanglelabels": [
                                self.labels[int(prediction.cls.item())]
                            ],
                        },
                    }
                )
                score += prediction.conf.item()

        return [
            {
                "result": predictions,
                "score": score / (i + 1),
                # all predictions will be differentiated by model version
                "model_version": "v8n",
            }
        ]
