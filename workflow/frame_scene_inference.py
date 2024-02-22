import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import luigi
from ultralytics import YOLO


class SceneClassificationInference(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    checkpoint = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.output_path + "/output.json")

    def run(self):
        model = YOLO(self.checkpoint)
        model_prediction = model(self.input_path, save=False, conf=0.2, iou=0.5, verbose=False)

        results = []
        for pred in model_prediction:
            probs = pred.probs.cpu().numpy()
            result_dict.append(dict(
                labels=list(result.names.values())
                path=result.path,
                prob=probs.data.tolist()
            ))

        output_path = Path(self.output().path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as outfile:
            json.dump(result_dict, outfile)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-root-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/frames/",
    )
    parser.add_argument(
        "--output-root-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/fullframe_inference/",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/cs-share/pradalier/tmp/yolo_segmentation_runs/classify/train10/weights/best.pt",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_batch_root = sorted(Path(args.input_root_path).glob("*/*"))

    luigi.build(
        [
            SceneClassificationInference(
                input_path=p.as_posix(),
                output_path=args.output_root_path + p.parents[0].name + "/" + p.name,
                checkpoint=args.checkpoint,
            )
            for p in image_batch_root
        ],
        workers=args.num_workers,
    )
