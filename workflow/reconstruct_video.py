"""Script for sampling frames from livestream judo videos.

In this particular script, we are generating frames that we will use for whole
scene classification. We will sample at 1hz, and place the resulting frames into
a directory structure that should be relatively easy to retrieve for our
labeling tasks.
"""

from pathlib import Path
from argparse import ArgumentParser
import luigi
import ffmpeg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from judo_footage_analysis.utils import ensure_parent


class ReconstructVideoClassificationInference(luigi.Task):
    input_frames_path = luigi.Parameter()
    input_inference_path = luigi.Parameter()
    prefix = luigi.Parameter()
    output_path = luigi.Parameter()

    def _preprocess(self, df):
        # a bunch of pre-processing
        df["mat"] = df["path"].apply(lambda x: int(x.split("/")[-3].split("_")[-1]))
        df["batch_id"] = df["path"].apply(lambda x: int(x.split("/")[-2]))
        df["frame_id"] = df["path"].apply(lambda x: int(x.split("/")[-1].split(".")[0]))
        df["timestamp"] = df.batch_id * 600 + df.frame_id
        df["predicted_index"] = df.prob.apply(lambda x: np.argmax(x))
        labels = df.iloc[0].labels
        df["predicted_label"] = df.predicted_index.apply(lambda x: labels[x])
        for label in labels:
            df[label] = df.prob.apply(lambda x: x[labels.index(label)])
        df = df.set_index("timestamp").sort_index()
        return df

    def _plot(self, df):
        labels = df.iloc[0].labels
        mat_id = df.iloc[0].mat
        batch_id = df.iloc[0].batch_id
        for label in labels:
            df[label].plot(label=label, figsize=(12, 4))
        plt.gcf().set_facecolor("white")
        plt.legend()
        plt.title(f"Probabilities over time for mat {mat_id} batch {batch_id}")

    def output(self):
        return luigi.LocalTarget(
            (Path(self.output_path) / self.prefix / "_SUCCESS").as_posix()
        )

    def run(self):
        df = pd.read_json(Path(self.input_inference_path) / self.prefix / "output.json")
        df = self._preprocess(df)
        # pad the dataframe so our plots are consistent
        min_timestamp = df.index.min()
        max_timestamp = df.index.max()
        for i in range(min_timestamp - 60, min_timestamp):
            df.loc[i] = df.iloc[0]
        for i in range(max_timestamp + 1, max_timestamp + 60):
            df.loc[i] = df.iloc[-1]

        output_path = ensure_parent(self.output().path).parent

        # now plot windows of 60 seconds, centered on the current timestamp
        print(df.index.min(), df.index.max())
        for i in range(min_timestamp, max_timestamp):
            window = df.loc[i - 30 : i + 30]
            self._plot(window)
            plt.savefig(output_path / f"{i}.png")
            plt.title(f"Probabilities over time (t={df.index[i]})")
            plt.close()

        with self.output().open("w") as f:
            f.write("")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-frames-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/frames/",
    )
    parser.add_argument(
        "--input-inference-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/fullframe_inference/",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/reconstructed/",
    )
    parser.add_argument("--num-workers", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_frames_path).glob("*/*"))

    luigi.build(
        [
            ReconstructVideoClassificationInference(
                input_frames_path=args.input_frames_path,
                input_inference_path=args.input_inference_path,
                prefix=p.relative_to(Path(args.input_frames_path)).as_posix(),
                output_path=args.output_root_path,
            )
            for p in image_batch_root[-1:]
        ],
        workers=args.num_workers,
    )
