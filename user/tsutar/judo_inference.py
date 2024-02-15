from ultralytics import YOLO
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
import luigi



class Inference(luigi.Task):
    input_path = luigi.Parameter(default='/cs-share/pradalier/tmp/judo/frames/')
    output_root_path = luigi.Parameter(default = '/home/GTL/tsutar/intro_to_res/' )
    # output_prefix = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        # self.frames_path = 
        # self.output_path = '/home/GTL/tsutar/intro_to_res/'
        # self.model_name = '/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/runs/classify/train10/weights/best.pt'
        pass
        

    def output(self):
        """We check for the existence of the output directory, and a success sempahore."""
        return [
            luigi.LocalTarget(self.output_path),
            luigi.LocalTarget(self.output_path / "_SUCCESS"),
        ]

    def run(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        model_name = '/home/GTL/tsutar/intro_to_res/cs8813-judo-footage-analysis/user/tsutar/runs/classify/train10/weights/best.pt'
        model = YOLO(model_name)

        for d in self.input_path:
            print(d)

        # write a success sempahore
        with self.output()[1].open("w") as f:
            f.write("")



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images = sorted(Path(args.input_root_path).glob("*/*"))
    print(images)
    # luigi.build(
    #     [
    #         Inference(
    #             input_path=p.as_posix(),
    #             output_root_path=args.output_root_path,
    #         )

    #         for i,p in enumerate(images)
    #     ],
    #     workers=args.num_workers,
    # )
