import logging
import argparse

from greetings import cfg, Dataset, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(prog="Preprocess SQuAD 1 data")
    parser.add_argument(
        "--cfg", dest="cfg", type=str, help="Config YAML filepath",
        required=False, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    logging.basicConfig(
        level=cfg.LOG_LVL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _ = Dataset(prepare=True)


if __name__ == "__main__":
    main()
