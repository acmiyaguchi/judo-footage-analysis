import os
import argparse
import logging
import logging.config
from label_studio_ml.api import init_app
from judo_footage_analysis.active_labeling.model import YOLOv8Model
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description="Label studio")
    parser.add_argument("-p", "--port", type=int, default=9090, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("-d", "--debug", action="store_true", help="Switch debug mode")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where models are stored (relative to the project directory)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate model instance before launching server",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8080",
        help="Base URL for the API",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        required=True,
        help="API token for the API",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig()
    if args.log_level:
        logging.root.setLevel(args.log_level)

    if args.check:
        YOLOv8Model()

    app = init_app(
        model_class=YOLOv8Model,
        base_url=args.base_url,
        api_token=args.api_token,
        model_dir=os.environ.get("MODEL_DIR", args.model_dir),
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
