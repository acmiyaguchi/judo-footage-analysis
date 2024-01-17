#!/usr/bin/env python3
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer


def parse_args():
    parser = ArgumentParser(description="run detectron2")
    parser.add_argument(
        "--config-file",
        type=str,
        default="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    )
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--model_device", type=str, default="cuda")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def get_detectron_model(args):
    """Initialize the detectron2 predictor."""
    cfg = get_cfg()
    # disable cuda at the moment
    cfg.MODEL.DEVICE = args.model_device
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)
    # only keep the top 5 instances
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    model = build_model(cfg)
    model.train(False)
    return model, cfg


@lru_cache
def probe_video_dim(input):
    """Probe the video dimensions."""
    probe = ffmpeg.probe(input)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    return width, height


def get_ffmpeg_frames_array(
    args, start_time: int, duration: int, frame_rate: int = 15
) -> np.ndarray:
    """Initialize the ffmpeg stream."""
    width, height = probe_video_dim(args.input)

    # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#convert-video-to-numpy-array
    # https://stackoverflow.com/questions/63623398/read-one-16bits-video-frame-at-a-time-with-ffmpeg-python
    out, _ = (
        ffmpeg.input(args.input)
        .trim(start_frame=start_time * frame_rate, duration=duration)
        .filter("fps", fps=frame_rate, round="up")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True)
    )
    video = (
        np.frombuffer(out, np.uint8).reshape([-1, height, width, 3]).astype(np.float32)
    )
    return video


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video: np.ndarray):
        self.video = video

    def __getitem__(self, index):
        # https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format
        # https://github.com/facebookresearch/detectron2/issues/282
        return {"image": torch.from_numpy(self.video[index].transpose(2, 0, 1))}

    def __len__(self):
        return len(self.video)


def main():
    args = parse_args()
    model, cfg = get_detectron_model(args)
    video = get_ffmpeg_frames_array(args, start_time=0, duration=30, frame_rate=1)

    dataset = VideoDataset(video)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=15, collate_fn=lambda x: x
    )
    for batch in dataloader:
        predictions = model(batch)
        # now write out an image to test
        for i, prediction in enumerate(predictions):
            v = Visualizer(
                batch[i]["image"].permute(1, 2, 0).numpy(),
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=1.2,
            )
            # print(prediction["instances"])
            out = v.draw_instance_predictions(prediction["instances"].to("cpu"))
            out.save(f"data/interim/test/output_{i}.jpg")
        break


if __name__ == "__main__":
    main()
