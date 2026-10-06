"""Shared forecasting algorithms; each experiment owns its fitted instance.

Lazy imports keep the policy/statistical backend usable without training deps.
NeuralForecast architectures use their official forward implementation with
deterministic chronological minibatches rather than random window sampling.
"""
from __future__ import annotations

import inspect
import random
from dataclasses import dataclass

import numpy as np

TRADITIONAL = ("SmartPersistence", "HistoryMean", "ARIMA", "SARIMA", "Prophet", "SVR", "LightGBM", "XGBoost")
DEEP = ("LSTM", "GRU", "BiLSTM", "TFT", "Informer", "Autoformer", "PatchTST", "DLinear", "RLinear", "NHiTS")
MODEL_NAMES = TRADITIONAL + DEEP


def set_seed(seed: int = 42, torch_required: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_required:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)


@dataclass
class TrainSettings:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 256
    learning_rate: float = .001
    hidden_size: int = 32
    trees: int = 200


def make_tree(name: str, settings: TrainSettings):
    if name == "SVR":
        from sklearn.svm import SVR
        return SVR(C=10, epsilon=.03, cache_size=512)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=settings.trees, num_leaves=31, learning_rate=.05,
                             random_state=settings.seed, n_jobs=2, verbosity=-1,
                             deterministic=True, force_col_wise=True)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=settings.trees, max_depth=5, learning_rate=.05,
                            random_state=settings.seed, n_jobs=2, tree_method="hist")
    raise ValueError(f"unknown tree/regression model: {name}")


def make_neural(name: str, history_size: int, horizon: int, exog_names: list[str],
                calendar_names: list[str], settings: TrainSettings):
    import torch
    from torch import nn
    if name in {"LSTM", "GRU", "BiLSTM", "RLinear"}:
        class WindowNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.kind = name
                self.horizon = horizon
                if name == "RLinear":
                    self.linear = nn.Linear(history_size, horizon)
                    self.exog_head = nn.Linear(len(exog_names) + len(calendar_names), horizon) if exog_names or calendar_names else None
                else:
                    cell = nn.GRU if name == "GRU" else nn.LSTM
                    bidirectional = name == "BiLSTM"
                    self.encoder = cell(1 + len(exog_names) + len(calendar_names), settings.hidden_size,
                                        batch_first=True, bidirectional=bidirectional)
                    self.head = nn.Linear(settings.hidden_size * (2 if bidirectional else 1) + horizon * len(calendar_names), horizon)

            def forward(self, batch):
                values = batch["insample_y"]
                hist = batch["hist_exog"]
                cal = batch["futr_exog"]
                if self.kind == "RLinear":
                    # Reversible instance normalization, linear projection,
                    # then denormalization. No Transformer surrogate.
                    mean = values.mean(dim=1)
                    std = values.std(dim=1, unbiased=False).clamp_min(1e-4)
                    output = self.linear(((values - mean[:, None]) / std[:, None]).squeeze(-1))
                    output = output * std + mean
                    if self.exog_head is not None:
                        ex = torch.cat([hist[:, -1], cal[:, -1]], dim=-1)
                        output = output + self.exog_head(ex)
                    return output
                x = torch.cat([values, hist, cal[:, :history_size]], dim=-1)
                _, state = self.encoder(x)
                hidden = state if self.kind == "GRU" else state[0]
                final = hidden[-2:].transpose(0, 1).reshape(len(values), -1) if name == "BiLSTM" else hidden[-1]
                return self.head(torch.cat([final, cal[:, history_size:].flatten(1)], dim=-1))
        return WindowNet()
    from neuralforecast import models as nf_models
    cls = getattr(nf_models, "NHITS" if name == "NHiTS" else name)
    kwargs = {"h": horizon, "input_size": history_size, "random_seed": settings.seed,
              "scaler_type": "identity", "max_steps": 1, "enable_progress_bar": False,
              "logger": False, "accelerator": "cpu"}
    sig = inspect.signature(cls).parameters
    if "hidden_size" in sig:
        kwargs["hidden_size"] = settings.hidden_size
    if "d_model" in sig:
        kwargs["d_model"] = settings.hidden_size
    if "n_heads" in sig:
        kwargs["n_heads"] = 4
    if "n_head" in sig:
        kwargs["n_head"] = 4
    if "encoder_layers" in sig:
        kwargs["encoder_layers"] = 1
    if "e_layers" in sig:
        kwargs["e_layers"] = 1
    if "d_layers" in sig:
        kwargs["d_layers"] = 1
    if "hist_exog_list" in sig and getattr(cls, "EXOGENOUS_HIST", False):
        kwargs["hist_exog_list"] = exog_names or None
    if "futr_exog_list" in sig and getattr(cls, "EXOGENOUS_FUTR", False):
        kwargs["futr_exog_list"] = calendar_names or None
    if "mlp_units" in sig:
        kwargs["mlp_units"] = [[settings.hidden_size, settings.hidden_size]] * 3
    if name == "PatchTST":
        kwargs.update(patch_len=8, stride=4, encoder_layers=1)
    if name == "Informer":
        # At this short input length, score every query. ProbSparse's
        # random ranking then cannot change eval outputs or batch results.
        kwargs["factor"] = history_size
    if name == "DLinear":
        kwargs["moving_avg_window"] = min(25, history_size if history_size % 2 else history_size - 1)
    return cls(**kwargs)


def package_for(name: str) -> str | None:
    return {
        "ARIMA": "statsmodels", "SARIMA": "statsmodels", "Prophet": "prophet",
        "SVR": "sklearn", "LightGBM": "lightgbm", "XGBoost": "xgboost",
        **{n: "torch" for n in ("LSTM", "GRU", "BiLSTM", "RLinear")},
        **{n: "neuralforecast" for n in ("TFT", "Informer", "Autoformer", "PatchTST", "DLinear", "NHiTS")},
    }.get(name)
