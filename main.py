"""Точка входа.

    python main.py train
    python main.py train --set model.name=rf --set cv.n_splits=10
    python main.py predict
"""

import argparse

import yaml

from src.training import predict, train


def load_config(path, overrides):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for override in overrides or []:
        key, _, value = override.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)  # "5" -> 5, "true" -> True
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="переопределить поле конфига, например --set model.name=xgb")
    parser.add_argument("--run-dir", help="для predict: папка запуска (по умолчанию последний)")
    parser.add_argument("--input", help="для predict: csv с данными (по умолчанию test из конфига)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    if args.command == "train":
        train(cfg)
    else:
        predict(cfg, run_dir=args.run_dir, input_path=args.input)


if __name__ == "__main__":
    main()
