"""A module to evaluate the performance of the overall pipeline."""

from pathlib import Path

import luigi
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F

from judo_footage_analysis.transforms import WrappedYOLOv8DetectEmbedding
from judo_footage_analysis.utils import spark_resource

from .sample_frames import FrameSampler


class ImageParquet(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    tmp_path = luigi.Parameter(default="/tmp/judo")
    num_partitions = luigi.IntParameter(default=32)

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def _consolidate(self, spark, tmp_path, output_path):
        df = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.jpg")
            .option("recursiveFileLookup", "true")
            .load(str(tmp_path))
            .withColumn(
                "path",
                F.udf(
                    lambda path: Path(path.replace("file:", ""))
                    .relative_to(tmp_path)
                    .as_posix()
                )("path"),
            )
        )
        df.printSchema()
        df.show()
        df.repartition(self.num_partitions).write.parquet(output_path, mode="overwrite")

    def run(self):
        input_root = Path(self.input_path)
        yield [
            FrameSampler(
                input_path=p.as_posix(),
                output_root_path=(
                    Path(self.tmp_path)
                    / p.relative_to(input_root).as_posix().replace(".mp4", "")
                ).as_posix(),
            )
            for p in input_root.glob("data/clips/**/*.mp4")
        ]

        with spark_resource() as spark:
            self._consolidate(spark, self.tmp_path, self.output_path)


class GenerateEmbeddings(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    feature_name = luigi.Parameter(default="yolov8n_emb")
    primary_key = luigi.Parameter(default="path")
    checkpoint = luigi.Parameter(default="yolov8n.pt")

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/data/_SUCCESS")

    def run(self):
        with spark_resource(cores=4, memory="4g") as spark:
            df = spark.read.parquet(self.input_path)
            df.printSchema()
            df.show()

            pipeline = Pipeline(
                stages=[
                    WrappedYOLOv8DetectEmbedding(
                        input_col="content",
                        output_col=self.feature_name,
                        checkpoint=self.checkpoint,
                    )
                ]
            )
            # save the pipeline
            pipeline.fit(df).write().overwrite().save(f"{self.output_path}/pipeline")
            model = PipelineModel.load(f"{self.output_path}/pipeline")

            # run inference
            model.transform(df).select(
                self.primary_key, self.feature_name
            ).write.parquet(f"{self.output_path}/data", mode="overwrite")
            transformed = spark.read.parquet(f"{self.output_path}/data")
            transformed.printSchema()
            transformed.show()


class EvaluationWorkflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        yield [
            GenerateEmbeddings(
                input_path=f"{self.input_path}/data/evaluation_frames/v1",
                output_path=f"{self.output_path}/data/evaluation_embeddings_entity_detection_v2/v1",
                checkpoint=f"{self.input_path}/models/entity_detection/v2/weights/best.pt",
                feature_name="entity_detection_v2_emb",
            ),
            GenerateEmbeddings(
                input_path=f"{self.input_path}/data/evaluation_frames/v1",
                output_path=f"{self.output_path}/data/evaluation_embeddings_vanilla_yolov8n_emb/v1",
                checkpoint=f"yolov8n.pt",
                feature_name="vanilla_yolov8n_emb",
            ),
        ]


if __name__ == "__main__":
    # first extract all the frames from the evaluation videos

    data_root = Path("/cs-share/pradalier/tmp/judo")
    luigi.build(
        [
            ImageParquet(
                input_path=f"{data_root}",
                output_path=f"{data_root}/data/evaluation_frames/v1",
            )
        ],
        workers=4,
        log_level="INFO",
    )

    luigi.build(
        [
            EvaluationWorkflow(
                input_path=f"{data_root}",
                output_path=f"{data_root}",
            )
        ],
        workers=1,
    )
