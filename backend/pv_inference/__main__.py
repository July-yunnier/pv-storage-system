"""python -m pv_inference --model state_grid_1 --input request.json"""
import argparse
import json
from pathlib import Path

from .infer import ROOT, load_model


def main():
    parser = argparse.ArgumentParser(description="Frozen PV inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release-root", type=Path, default=ROOT)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    request = json.loads(args.input.read_text(encoding="utf-8"))
    model = load_model(args.model, args.release_root, args.threads)
    result = model.predict(**request)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text+"\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
