"""A module to evaluate the performance of the overall pipeline."""

from pathlib import Path

import luigi
from pyspark.sql import functions as F

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


class EvaluationWorkflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        pass


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

    # luigi.build(
    #     [EvaluationWorkflow()],
    #     workers=1,
    # )
