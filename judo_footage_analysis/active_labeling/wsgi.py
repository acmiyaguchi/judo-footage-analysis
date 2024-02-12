import os
import argparse
import logging
import logging.config
from label_studio_ml.api import init_app
from model import YOLOv8Model


def parse_args():
    parser = argparse.ArgumentParser(description="Label studio")
    parser.add_argument(
        "-p", "--port", dest="port", type=int, default=9090, help="Server port"
    )
    parser.add_argument(
        "--host", dest="host", type=str, default="0.0.0.0", help="Server host"
    )
    parser.add_argument(
        "-d", "--debug", dest="debug", action="store_true", help="Switch debug mode"
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=os.path.dirname(__file__),
        help="Directory where models are stored (relative to the project directory)",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Validate model instance before launching server",
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
        model_dir=os.environ.get("MODEL_DIR", args.model_dir),
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
