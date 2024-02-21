import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import luigi
from ultralytics import YOLO


class Inference(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path + "/output.json")

    def run(self):
        model = YOLO(self.checkpoint)

        print(self.input_path)
        results = model(self.input_path, save=False, conf=0.2, iou=0.5, verbose=False)

        result_dict = []

        for result in results:
            res = {}
            probs = result.probs.cpu().numpy()
            res["labels"] = list(result.names.values())
            res["path"] = result.path
            res["prob"] = probs.data.tolist()
            result_dict.append(res)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output().path, "w") as outfile:
            json.dump(result_dict, outfile)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-root-path",
        type=str,
        default="/mnt/cs-share/pradalier/tmp/judo/frames/",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        default="/home/GTL/tsutar/intro_to_res/fullframe_inference/",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/runs/classify/train10/weights/best.pt",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_root_path).glob("*/*"))

    luigi.build(
        [
            Inference(
                input_path=p.as_posix(),
                output_path=args.output_root_path + p.parents[0].name + "/" + p.name,
                checkpoint=args.checkpoint,
            )
            for i, p in enumerate(image_batch_root)
        ],
        workers=args.num_workers,
    )
