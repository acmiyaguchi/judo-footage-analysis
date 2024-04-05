import io

import numpy as np
from imageio.v3 import imread
from pyspark.ml import Transformer
from pyspark.ml.functions import array_to_vector, predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType


class WrappedYOLOv8DetectEmbedding(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        checkpoint="yolov8n.pt",
        batch_size=8,
    ):
        super().__init__()
        self._setDefault(inputCol=input_col, outputCol=output_col)
        self.checkpoint = checkpoint
        self.batch_size = batch_size

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""
        from ultralytics import YOLO

        model = YOLO(self.checkpoint)

        def predict(inputs: np.ndarray) -> np.ndarray:
            activations = []

            def _hook(model, input, output):
                activations.append(output.detach().cpu().numpy())

            handle = (
                model.model.model[-1]
                ._modules["cv3"]
                ._modules["2"]
                .register_forward_hook(_hook)
            )

            try:
                images = [imread(io.BytesIO(input)) for input in inputs]
                list(
                    model.predict(
                        images,
                        device="cpu",
                        stream=True,
                        save=False,
                        verbose=False,
                    )
                )
                # stack the activations together
                res = np.stack(activations).reshape(len(images), -1)
            finally:
                handle.remove()
            return res

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            array_to_vector(
                predict_batch_udf(
                    make_predict_fn=self._make_predict_fn,
                    return_type=ArrayType(FloatType()),
                    batch_size=self.batch_size,
                )(self.getInputCol())
            ),
        )
