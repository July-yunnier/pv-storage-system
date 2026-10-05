from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.baselines import prediction_rows
from experiments.data import StateGridDataset
from experiments.models import official_ampdnet
from experiments.protocol import ROOT, code_version, dump_json, package_run, sample_indices, seed_everything


def run(dataset, epochs=10, ablation=False):
    seed_everything()
    directory = ROOT / "data/processed" / dataset
    frame = pd.read_pickle(directory / "frame.pkl")
    manifest = json.loads((directory / "manifest.json").read_text())
    split = json.loads((directory / "split.json").read_text())
    indices, sparse_future = sample_indices(frame, split, 64, 15)
    cap = manifest["cap_kw"]
    values = frame[["power_kw", "tsi", "dni", "ghi"]].to_numpy(dtype=np.float32)
    train_rows = frame.ts < pd.Timestamp(split["train_end"])
    means, stds = values[train_rows].mean(axis=0), values[train_rows].std(axis=0)
    stds = np.maximum(stds, 1e-6)
    means[0], stds[0] = 0, cap
    values = (values - means) / stds
    future = np.minimum(np.arange(len(frame))[:, None] + np.arange(1, 17)[None, :], len(frame)-1)
    tick = pd.DatetimeIndex(frame.ts).asi8
    good = (tick[future] == tick[:, None] + np.arange(1,17)[None,:] * 15 * 60 * 1_000_000_000).all(1)
    indices = {k: v[good[v]] for k,v in indices.items()}
    loaders = {k: DataLoader(StateGridDataset(values, frame.power_kw.to_numpy(), v, future, cap),
                             batch_size=128, shuffle=k=="train", num_workers=0, generator=torch.Generator().manual_seed(42))
               for k,v in indices.items()}
    model = official_ampdnet(ablation=ablation)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    name = "ampdnet_no_amsconv" if ablation else "ampdnet_official"
    out = ROOT / "runs" / dataset / name
    out.mkdir(parents=True, exist_ok=True)
    config = {"model": name, "dataset_manifest": manifest, "split": split, "epochs": epochs,
              "batch_size": 128, "learning_rate": .001, "seed": 42, "fit_on": "train",
              "select_checkpoint": "validation MSE", "predict_steps": 16,
              "feature_mean": means.tolist(), "feature_std": stds.tolist(),
              "indices": {k:len(v) for k,v in indices.items()}, "device": "cpu",
              "architecture": "author ConvPVNet.py; M=4,L=64,T=16,D=64,P=4,S=2,r=2,dp=0,layers=2",
              "upstream_commit": __import__("subprocess").check_output(["git","-C",str(ROOT/"research/official-ampdnet"),"rev-parse","HEAD"]).decode().strip(),
              "parameters": sum(p.numel() for p in model.parameters()), "protocol": "unified chronological; differs from author 60/20/20 and daylight-compressed timeline"}
    dump_json(out / "config.json", dict(config, code=code_version()))
    best = float("inf")
    log = []
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for x,y in loaders["train"]:
            optimizer.zero_grad()
            pred = model(x)
            loss = (pred-y).square().mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        model.eval()
        val, n = 0., 0
        with torch.no_grad():
            for x,y in loaders["val"]:
                val += float((model(x)-y).square().sum())
                n += y.numel()
        val /= n
        log.append({"epoch":epoch+1,"train_mse":total/count,"validation_mse":val,"elapsed_seconds":time.monotonic()-started})
        if val < best:
            best = val
            torch.save(model.state_dict(), out / "weights.pt")
        dump_json(out / "training.json", log)
        print(dataset,name,log[-1],flush=True)
    model.load_state_dict(torch.load(out / "weights.pt", weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for x,y in loaders["test"]:
            preds.append(model(x).numpy()[:, [0,3,15]])
    predictions = np.clip(np.concatenate(preds) * cap, 0, cap)
    rows = prediction_rows(frame, indices["test"], sparse_future, predictions)
    package_run(out, config, rows, {"weights_file":"weights.pt", "checkpoint_selection":"validation MSE", "best_validation_mse":best}, cap)
    print("completed",dataset,name,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="state_grid_1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    for dataset in ([f"state_grid_{i}" for i in range(1,9)] if args.dataset=="all" else [args.dataset]):
        run(dataset,args.epochs,args.ablation)
