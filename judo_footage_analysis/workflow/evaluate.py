"""A module to evaluate the performance of the overall pipeline."""

from pathlib import Path

import luigi
import numpy as np
from contexttimer import Timer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    MultilabelClassificationEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
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


class ConsolidateEmbeddings(luigi.Task):
    label_path = luigi.Parameter()
    input_paths = luigi.ListParameter()
    feature_names = luigi.ListParameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:

            @F.udf(returnType="struct<file: string, time: long>")
            def extract_key_udf(path: str) -> dict:
                parts = Path(path).parts
                filename = "/".join(parts[:-1]) + ".mp4"
                time = int(Path(path).stem)
                return {"file": filename, "time": time}

            df = spark.read.json(self.label_path, multiLine=True)
            df.printSchema()
            df.show(truncate=100)
            for name, feature_df in zip(
                self.feature_names,
                [
                    spark.read.parquet(f"{input_path}/data")
                    for input_path in self.input_paths
                ],
            ):
                tmp = feature_df.withColumn("key", extract_key_udf("path")).select(
                    F.col("key.file").alias("file"),
                    F.col("key.time").alias("time"),
                    name,
                )
                df = df.join(tmp, on=["file", "time"])
            df.printSchema()
            df.write.parquet(self.output_path, mode="overwrite")

            # show the result
            df = spark.read.parquet(self.output_path, truncate=100)
            df.show()
            df.describe().show()


class FitLogisticModelBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    features = luigi.ListParameter(default=["features"])
    label = luigi.Parameter(default="label")

    num_folds = luigi.IntParameter(default=3)
    max_iter = luigi.ListParameter(default=[100])
    reg_param = luigi.ListParameter(default=[0.0])
    elastic_net_param = luigi.ListParameter(default=[0.0])
    seed = luigi.IntParameter(default=42)

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def _transform(self, df):
        for col in self.features:
            # check if array
            if df.schema[col].dataType.typeName() == "array":
                df = df.withColumn(col, array_to_vector(col).alias(col))
        return df

    def _load(self, spark):
        return self._transform(spark.read.parquet(self.input_path))

    def _train_test_split(self, df, train_ratio=0.8):
        sample_id = df.withColumn("sample_id", F.crc32(F.col("file")) % 100)
        train = (
            sample_id.filter(F.col("sample_id") < train_ratio * 100)
            .repartition(16)
            .cache()
        )
        test = (
            sample_id.filter(F.col("sample_id") >= train_ratio * 100)
            .repartition(16)
            .cache()
        )
        return train, test

    def _pipeline(self):
        raise NotImplementedError()

    def _evaluator(self):
        return MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label,
            metricName="f1",
        )

    def _param_grid(self, lr):
        return (
            ParamGridBuilder()
            .addGrid(lr.maxIter, self.max_iter)
            .addGrid(lr.regParam, self.reg_param)
            .addGrid(lr.elasticNetParam, self.elastic_net_param)
            .build()
        )

    def run(self):
        with spark_resource() as spark:
            train, test = self._train_test_split(self._load(spark))

            # write the model to disk
            pipeline = self._pipeline()
            lr = [
                stage
                for stage in pipeline.getStages()
                if isinstance(stage, LogisticRegression)
            ][-1]
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=self._param_grid(lr),
                evaluator=self._evaluator(),
                numFolds=self.num_folds,
            )

            with Timer() as train_timer:
                cv.fit(train).write().overwrite().save(f"{self.output_path}/model")

            model = CrossValidatorModel.load(f"{self.output_path}/model")

            # evaluate against the test set
            with Timer() as eval_timer:
                predictions = model.transform(test)
                f1 = self._evaluator().evaluate(predictions)

            # write the results to disk
            perf = spark.createDataFrame(
                [
                    {
                        "train_time": train_timer.elapsed,
                        "avg_metrics": np.array(model.avgMetrics).tolist(),
                        "std_metrics": np.array(model.avgMetrics).tolist(),
                        "metric_name": model.getEvaluator().getMetricName(),
                        "test_time": eval_timer.elapsed,
                        "test_metric": f1,
                        "label": self.label,
                    }
                ],
                schema="train_time double, avg_metrics array<double>, std_metrics array<double>, metric_name string",
            )
            perf.write.json(f"{self.output_path}/perf", mode="overwrite")
            perf.show()

        # write the output
        with self.output().open("w") as f:
            f.write("")


class FitLogisticModel(FitLogisticModelBase):
    def _pipeline(self):
        return Pipeline(
            stages=[
                VectorAssembler(
                    inputCols=self.features,
                    outputCol="features",
                ),
                StandardScaler(inputCol="features", outputCol="scaled_features"),
                LogisticRegression(featuresCol="scaled_features", labelCol=self.label),
            ]
        )


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
        yield ConsolidateEmbeddings(
            label_path=f"{self.input_path}/data/combat_phase/discrete_v2/labels.json",
            input_paths=[
                f"{self.output_path}/data/evaluation_embeddings_entity_detection_v2/v1",
                f"{self.output_path}/data/evaluation_embeddings_vanilla_yolov8n_emb/v1",
            ],
            feature_names=["entity_detection_v2_emb", "vanilla_yolov8n_emb"],
            output_path=f"{self.output_path}/data/evaluation_embeddings/v1",
        )

        # now let's fit the simplest possible model
        yield [
            FitLogisticModel(
                input_path=f"{self.output_path}/data/evaluation_embeddings/v1",
                output_path=f"{self.output_path}/models/evaluation_embeddings_logistic_binary/v1/{col}/{label}",
                features=[col],
                label=label,
            )
            for col in ["entity_detection_v2_emb", "vanilla_yolov8n_emb"]
            for label in ["is_match", "is_active", "is_standing"]
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
