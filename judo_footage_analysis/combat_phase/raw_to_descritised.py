import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import ffmpeg
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CombatPhaseRawToDescritised(luigi.Task):
    input_json_path = luigi.Parameter()
    output_json_path = luigi.Parameter()
    interval_duration = luigi.IntParameter(default=5)

    def output(self):
        return luigi.LocalTarget(self.output_json_path)

    def run(self):
        df = pd.read_json(self.input_json_path)
        file = df["file"]
        annotations = df["annotations"]

        descritised_annotations = []
        for file, annotation in zip(file, annotations):
            descritised_annotation = []

            # Get the min start time and max end time
            min_start_time = min([a["start"] for a in annotation])
            max_end_time = max([a["end"] for a in annotation])

            # Create the recording points
            recording_points = np.arange(
                min_start_time, max_end_time, self.interval_duration
            )
            recording_points = np.append(recording_points, max_end_time)

            for i, recording_point in enumerate(recording_points[:-1]):
                is_match = False
                is_active = False
                is_standing = False

                for a in annotation:
                    if a["labels"] == "Match" and not is_match:
                        if (
                            a["start"] <= recording_point
                            and a["end"] >= recording_points[i + 1]
                        ):
                            is_match = True
                    if a["labels"] == "Active" and not is_active:
                        if (
                            a["start"] <= recording_point
                            and a["end"] >= recording_points[i + 1]
                        ):
                            is_active = True
                    if a["labels"] == "Standing" and not is_standing:
                        if (
                            a["start"] <= recording_point
                            and a["end"] >= recording_points[i + 1]
                        ):
                            is_standing = True

                descritised_annotation.append(
                    {
                        "file": file,
                        "time": recording_point,
                        "is_match": is_match,
                        "is_active": is_active,
                        "is_standing": is_standing,
                    }
                )

            descritised_annotations.extend(descritised_annotation)

        # Log the list of descritised annotations
        print(descritised_annotations)

        # Convert the list of descritised annotations to a DataFrame
        descritised_annotations = pd.DataFrame(descritised_annotations)

        # Save the descritised annotations
        descritised_annotations.to_json(self.output_json_path)


class Workflow(luigi.Task):
    input_json_path = luigi.Parameter()
    output_json_path = luigi.Parameter()
    interval_duration = luigi.IntParameter(default=5)

    def requires(self):
        return CombatPhaseRawToDescritised(
            input_json_path=self.input_json_path,
            output_json_path=self.output_json_path,
            interval_duration=self.interval_duration,
        )

    def run(self):
        pass


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-json-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/combat_phase/filtered_annotations.json",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default="/cs-share/pradalier/tmp/judo/data/combat_phase/descritised_annotations.json",
        help="Path to save the descritised annotations.",
    )
    parser.add_argument(
        "--interval-duration",
        type=int,
        default=5,
        help="Duration of each interval in seconds.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of workers for Luigi."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [
            Workflow(
                input_json_path=args.input_json_path,
                output_json_path=args.output_json_path,
                interval_duration=args.interval_duration,
            )
        ]
    )
