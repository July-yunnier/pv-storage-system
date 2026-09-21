# -*- coding: utf-8 -*-
"""
05_train_sunset.py — SUNSET 模型训练（PyTorch 实现）

严格还原官方 TensorFlow 版的结构与训练策略：
  · 网络  : Conv(24,3x3)-BN-MaxPool -> Conv(48,3x3)-BN-MaxPool -> Flatten
            -> [forecast 额外拼接 16 维功率历史]
            -> Dense(1024)-Dropout(0.4) -> Dense(1024)-Dropout(0.4) -> Dense(1)
  · 损失  : MSE
  · 优化器: Adam(3e-6)          ← 注意学习率极小
  · 划分  : day-block shuffle + 10 折交叉验证   ← 绝不能随机打乱样本
  · 早停  : val_loss 连续 5 轮不降
  · 推理  : 10 个模型的集成均值

为什么用 PyTorch 版：
  老师的方案三（fusion_model.py）本来就是 PyTorch，
  统一技术栈便于队友接手、也便于把本模型融进现有代码。

用法：
    # 先跑 1 折验证流程（强烈建议）
    python 05_train_sunset.py --task nowcast --data_dir ./data_expanded --folds 1 --epochs 30
    # 完整复现
    python 05_train_sunset.py --task nowcast --data_dir ./data_expanded --folds 10 --epochs 200
    python 05_train_sunset.py --task forecast --data_dir ./data_expanded --folds 10 --epochs 200
"""

import argparse
import datetime as dt
import itertools
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("需要 PyTorch：pip install torch")
    sys.exit(1)

# ---------------------------------------------------------------- 超参数
NUM_FILTERS = 24
KERNEL = 3
DENSE = 1024
DROP = 0.4
LR = 3e-06
BATCH = 256
PATIENCE = 5

# DataLoader 工作进程数，由命令行 --num_workers 覆盖。
# ⚠️ macOS 上请保持 0（默认）。
#    实测对比（Apple M5，349,372 样本，1 epoch）：
#      num_workers=0  ->  1 分 42 秒
#      num_workers=4  ->  > 2 分 52 秒（更慢，且未跑完）
#    原因：macOS 用 spawn 启动子进程，每个 worker 会各持一份 4 GB 图像数组，
#    内存与进程间通信开销远超并行收益。
#    Linux 上可以试 4~8。
NUM_WORKERS = 0


# ---------------------------------------------------------------- 模型
class SUNSET(nn.Module):
    """
    task='nowcast' : 输入 [B,3,S,S]                      -> 当前功率
    task='forecast': 输入 [B,48,S,S] 与 [B,16]（功率历史）-> 15 分钟后功率
    """

    def __init__(self, task, img_channels=3, log_terms=0):
        super().__init__()
        self.task = task
        self.log_terms = log_terms

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, KERNEL, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.conv = nn.Sequential(block(img_channels, NUM_FILTERS),
                                  block(NUM_FILTERS, NUM_FILTERS * 2))

        # 两次 2x2 池化后，特征图为 S/4 x S/4
        side = 64 // 4
        feat = NUM_FILTERS * 2 * side * side

        head_in = feat + (log_terms if task == "forecast" else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, DENSE), nn.ReLU(inplace=True), nn.Dropout(DROP),
            nn.Linear(DENSE, DENSE), nn.ReLU(inplace=True), nn.Dropout(DROP),
            nn.Linear(DENSE, 1),
        )

    def forward(self, x, pv_log=None):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if self.task == "forecast":
            x = torch.cat([x, pv_log], dim=1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------- 数据
class ArrayDataset(torch.utils.data.Dataset):
    """
    idx 不为 None 时用索引访问全量数组，而不是每次切一片副本。
    官方基准数据是 349,372×64×64×3 ≈ 3.8 GB，若每折都做 img[idx] 复制一份，
    10 折下来内存峰值会翻倍。用索引可把额外开销压到几 MB。
    """

    def __init__(self, images, targets, pv_log=None, idx=None, augment=False):
        self.images = images
        self.targets = targets
        self.pv_log = pv_log
        self.idx = idx
        self.augment = augment

    def __len__(self):
        return len(self.idx) if self.idx is not None else len(self.targets)

    def __getitem__(self, i):
        j = int(self.idx[i]) if self.idx is not None else i
        img = self.images[j].astype(np.float32) / 255.0        # 归一到 [0,1]
        img = np.transpose(img, (2, 0, 1))                     # HWC -> CHW
        if self.augment and np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()                       # 水平翻转
        t = torch.from_numpy(img)
        y = torch.tensor(float(self.targets[j]))
        if self.pv_log is not None:
            return t, torch.from_numpy(self.pv_log[j].astype(np.float32)), y
        return t, y


def day_block_shuffle(times):
    """
    按「天」分块打乱。
    ⚠️ 绝不能直接 shuffle 样本：相邻分钟的图像几乎一样，
       随机打乱会让验证集泄漏训练集信息，分数虚高。
    """
    dates = np.asarray([t.date() for t in times])
    blocks = [np.where(dates == d)[0] for d in np.unique(dates)]
    rng = np.random.RandomState(1)
    rng.shuffle(blocks)
    return np.asarray(list(itertools.chain.from_iterable(blocks)))


def cv_split(order, fold, n_fold):
    """
    交叉验证切分。
    n_fold >= 2 时按折切分；n_fold == 1 时退化为「前 10% 作验证」的快速模式
    （用于先跑通流程，不能用于最终结果）。
    """
    n = len(order)
    if n_fold <= 1:
        hi = max(1, int(0.1 * n))
        train_idx, val_idx = order[hi:], order[:hi]
    else:
        lo, hi = int(fold / n_fold * n), int((fold + 1) / n_fold * n)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[lo:hi] = True
        val_idx = order[val_mask]
        train_idx = order[~val_mask]
    rng = np.random.RandomState(fold)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ---------------------------------------------------------------- 训练
def run_fold(task, imgs, tgt, log, idx_tr, idx_va, epochs, device, tag):
    ch = imgs.shape[-1]
    log_terms = log.shape[1] if log is not None else 0
    model = SUNSET(task, img_channels=ch, log_terms=log_terms).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()

    ds_tr = ArrayDataset(imgs, tgt, log, idx=idx_tr, augment=True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                                        num_workers=NUM_WORKERS, drop_last=True,
                                        persistent_workers=NUM_WORKERS > 0)
    ds_va = ArrayDataset(imgs, tgt, log, idx=idx_va)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=500, shuffle=False,
                                        num_workers=NUM_WORKERS,
                                        persistent_workers=NUM_WORKERS > 0)

    def evaluate():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                if task == "forecast":
                    x, pl, y = batch
                    pl = pl.to(device)
                else:
                    x, y = batch
                    pl = None
                x, y = x.to(device), y.to(device)
                pred = model(x, pl)
                tot += float(((pred - y) ** 2).sum())
                cnt += len(y)
        return tot / max(cnt, 1)

    best_val = float("inf")
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        run, cnt = 0.0, 0
        for batch in dl_tr:
            if task == "forecast":
                x, pl, y = batch
                pl = pl.to(device)
            else:
                x, y = batch
                pl = None
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            pred = model(x, pl)
            loss = lossf(pred, y)
            loss.backward()
            opt.step()

            run += float(((pred - y) ** 2).sum())
            cnt += len(y)

        train_mse = run / max(cnt, 1)
        val_mse = evaluate()

        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep % 5 == 0 or ep == 1:
            print(f"    [{tag}] ep{ep:>4}  train_RMSE={np.sqrt(train_mse):.4f}"
                  f"  val_RMSE={np.sqrt(val_mse):.4f}  best={np.sqrt(best_val):.4f}")

        if wait >= PATIENCE:
            print(f"    [{tag}] 早停于第 {ep} 轮（验证损失连续 {PATIENCE} 轮未降）")
            break

    model.load_state_dict(best_state)
    return model


def predict(task, model, imgs, log, idx, device):
    ds = ArrayDataset(imgs, np.zeros(len(imgs), dtype=np.float32), log, idx=idx)
    dl = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False, num_workers=0)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            if task == "forecast":
                x, pl, _ = batch
                pl = pl.to(device)
            else:
                x, _ = batch
                pl = None
            out.append(model(x.to(device), pl).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="训练 SUNSET 模型")
    ap.add_argument("--task", choices=["nowcast", "forecast"], required=True)
    ap.add_argument("--data_dir", default="",
                    help="03/04 脚本产出的目录（用 --hdf5 时不需要）")
    ap.add_argument("--hdf5", default="",
                    help="直读官方基准 hdf5（仅 nowcast；times_*.npy 需在同目录）")
    ap.add_argument("--out_dir", default="./model_output")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--fold_ids", default="",
                    help="只跑指定折，逗号分隔（如 1,2,3），默认全部。"
                         "用途：10 折 x 200 轮耗时长，macOS 内存压力可能中途发 SIGKILL，"
                         "分批跑并让每折结果立即落盘，被杀也不白跑；"
                         "下次接着跑剩下的折，最后自动合并所有已完成的折")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--max_samples", type=int, default=0,
                    help="只用开发集前 N 个样本（调试用，0=全量）")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader 进程数。⚠️ macOS 保持 0（实测 4 反而更慢）；"
                         "Linux 可试 4~8")
    args = ap.parse_args()

    global NUM_WORKERS
    NUM_WORKERS = max(0, args.num_workers)

    if not args.hdf5 and not args.data_dir:
        print("[错误] 需要指定数据来源：--hdf5（官方基准）或 --data_dir（自建预处理结果）")
        return 1

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available() else "cpu")
    else:
        dev = args.device
    device = torch.device(dev)

    print("=" * 78)
    print(f"SUNSET {args.task} 训练")
    print("=" * 78)
    print(f"设备 {dev}  |  折数 {args.folds}  |  最大轮数 {args.epochs}")
    print(f"超参数: filters={NUM_FILTERS} dense={DENSE} drop={DROP} lr={LR} batch={BATCH}")
    if args.folds < 2:
        print("\n[注意] 折数 < 2 时为「快速跑通模式」：前 10% 作验证集，"
              "仅用于验证流程是否通顺，不能作为最终结果。")
        print("       正式复现请用 --folds 10。")
    print()

    if args.hdf5:
        # ---------- 直读官方基准 hdf5（跳过 03/04 预处理）----------
        if args.task != "nowcast":
            print("[错误] --hdf5 目前只支持 nowcast 任务。")
            print("       forecast 样本需要先用 04_build_forecast.py 生成。")
            return 1
        import h5py
        h5path = os.path.abspath(os.path.expanduser(args.hdf5))
        if not os.path.exists(h5path):
            print(f"[错误] 找不到 hdf5：{h5path}")
            return 1
        h5dir = os.path.dirname(h5path)

        tv_p = os.path.join(h5dir, "times_trainval.npy")
        te_p = os.path.join(h5dir, "times_test.npy")
        missing = [p for p in (tv_p, te_p) if not os.path.exists(p)]
        if missing:
            print("[错误] 缺少训练必需的时间戳文件：")
            for p in missing:
                print(f"        {p}")
            print("       请把 times_trainval.npy 和 times_test.npy 放到 hdf5 同一目录。")
            return 1

        print("数据来源: 官方基准 hdf5（跳过预处理）")
        print(f"  文件: {h5path}")
        with h5py.File(h5path, "r") as f:
            img_tr = f["trainval"]["images_log"][...]
            target_tr = f["trainval"]["pv_log"][...].astype(np.float32)
            img_te = f["test"]["images_log"][...]
            y_te = f["test"]["pv_log"][...].astype(np.float32)
        t_tr = np.load(tv_p, allow_pickle=True)
        t_te = np.load(te_p, allow_pickle=True)
        pv_tr = None
        log_tr = log_te = None
        print(f"  已载入内存约 {(img_tr.nbytes + img_te.nbytes)/1024**3:.2f} GB（图像占大头）")
    elif args.task == "nowcast":
        sub = "data_nowcast"
        img_tr = np.load(os.path.join(args.data_dir, sub, "image_log_trainval.npy"))
        pv_tr = np.load(os.path.join(args.data_dir, sub, "pv_log_trainval.npy"))
        t_tr = np.load(os.path.join(args.data_dir, sub, "times_trainval.npy"), allow_pickle=True)
        img_te = np.load(os.path.join(args.data_dir, sub, "image_log_test.npy"))
        pv_te = np.load(os.path.join(args.data_dir, sub, "pv_log_test.npy"))
        t_te = np.load(os.path.join(args.data_dir, sub, "times_test.npy"), allow_pickle=True)
        log_tr = log_te = None
        y_te = pv_te
    else:
        sub = os.path.join("data_forecast", "frequency_2")
        img_tr = np.load(os.path.join(args.data_dir, sub, "image_log_trainval.npy"))
        log_tr = np.load(os.path.join(args.data_dir, sub, "pv_log_trainval.npy"))
        y_tr = np.load(os.path.join(args.data_dir, sub, "pv_pred_trainval.npy"))
        t_tr = np.load(os.path.join(args.data_dir, sub, "times_trainval.npy"), allow_pickle=True)
        img_te = np.load(os.path.join(args.data_dir, sub, "image_log_test.npy"))
        log_te = np.load(os.path.join(args.data_dir, sub, "pv_log_test.npy"))
        y_te = np.load(os.path.join(args.data_dir, sub, "pv_pred_test.npy"))
        t_te = np.load(os.path.join(args.data_dir, sub, "times_test.npy"), allow_pickle=True)
        pv_tr = None

    print(f"开发集 {img_tr.shape}  测试集 {img_te.shape}")

    if args.max_samples and len(img_tr) > args.max_samples:
        n = args.max_samples
        img_tr = np.ascontiguousarray(img_tr[:n])
        target_tr = target_tr[:n]
        t_tr = t_tr[:n]
        if log_tr is not None:
            log_tr = log_tr[:n]
        print(f"[调试] 开发集已限制为前 {n} 个样本")

    if args.task == "forecast":
        # (N,16,S,S,3) -> (N,S,S,48)：把时间维折进通道维
        img_tr = np.transpose(img_tr, (0, 2, 3, 1, 4)).reshape(
            img_tr.shape[0], img_tr.shape[2], img_tr.shape[3], -1)
        img_te = np.transpose(img_te, (0, 2, 3, 1, 4)).reshape(
            img_te.shape[0], img_te.shape[2], img_te.shape[3], -1)
        target_tr = y_tr
        print(f"图像折叠后 {img_tr.shape}（16 帧 x 3 通道 = 48）")
    elif not args.hdf5:
        target_tr = pv_tr
    # 注：hdf5 分支已在加载时设好 target_tr，此处不要覆盖

    print("\n按天分块打乱（防止相邻样本泄漏）")
    order = day_block_shuffle(t_tr)

    out_dir = os.path.join(args.out_dir, f"SUNSET_{args.task}")
    os.makedirs(out_dir, exist_ok=True)

    # 本次要跑哪些折（支持分批，见 --fold_ids）
    if args.fold_ids:
        try:
            fold_list = [int(x) - 1 for x in str(args.fold_ids).split(",") if x.strip()]
        except ValueError:
            print("[错误] --fold_ids 格式应为逗号分隔的折号，例如 1,2,3")
            return 1
        bad = [f + 1 for f in fold_list if f < 0 or f >= args.folds]
        if bad:
            print(f"[错误] 折号超出范围 1~{args.folds}：{bad}")
            return 1
    else:
        fold_list = list(range(args.folds))

    print(f"\n本次要跑的折: {[f + 1 for f in fold_list]}   （总折数 {args.folds}）")

    for f in fold_list:
        print(f"\n{'─'*78}\n第 {f+1}/{args.folds} 折")
        tr_idx, va_idx = cv_split(order, f, args.folds)

        model = run_fold(args.task, img_tr, target_tr, log_tr,
                         tr_idx, va_idx, args.epochs, device, f"fold{f+1}")
        p = predict(args.task, model, img_te, log_te,
                    np.arange(len(y_te)), device)

        torch.save(model.state_dict(), os.path.join(out_dir, f"fold_{f+1}.pt"))
        # ⭐ 每折立即落盘。macOS 内存压力会对长任务发 SIGKILL，
        #    落盘后即使被杀，这一折也不算白跑，下次用 --fold_ids 接着跑剩下的。
        np.save(os.path.join(out_dir, f"fold_{f+1}_pred.npy"), p)

        vr = float(np.sqrt(np.mean((p - y_te) ** 2)))
        print(f"  折 {f+1} 测试 RMSE = {vr:.4f}   （已存 fold_{f+1}_pred.npy）")

    # 集成：合并【所有已完成】的折（不限于本次跑的），与官方 10 折集成一致
    files = sorted(
        [os.path.join(out_dir, x) for x in os.listdir(out_dir)
         if x.startswith("fold_") and x.endswith("_pred.npy")],
        key=lambda p: int(os.path.basename(p).split("_")[1]))
    if not files:
        print("[错误] 没有任何折的预测结果")
        return 1

    allp = np.stack([np.load(p) for p in files])
    ens = allp.mean(axis=0)
    rmse = float(np.sqrt(np.mean((ens - y_te) ** 2)))
    mae = float(np.mean(np.abs(ens - y_te)))

    np.save(os.path.join(out_dir, "test_predictions.npy"), allp)
    np.save(os.path.join(out_dir, "prediction_ensemble.npy"), ens)

    done = ", ".join(f"折{int(os.path.basename(p).split('_')[1])}" for p in files)
    print("\n" + "=" * 78)
    print(f"集成结果（{len(files)} 折均值：{done}）")
    print(f"  RMSE = {rmse:.4f} kW")
    print(f"  MAE  = {mae:.4f} kW")
    print(f"  1 - RMSE/Cap(30.1) = {1 - rmse/30.1:.4f}")
    print(f"  单折 RMSE 均值 = {np.mean([float(np.sqrt(np.mean((np.load(p) - y_te) ** 2))) for p in files]):.4f}")
    if len(files) < args.folds:
        rest = [f + 1 for f in range(args.folds)
                if not os.path.exists(os.path.join(out_dir, f"fold_{f+1}_pred.npy"))]
        print(f"\n  [未完成] 还缺折 {rest}，接着跑：")
        print(f"      python 05_train_sunset.py --task {args.task} --hdf5 <path> "
              f"--fold_ids {','.join(map(str, rest))}")
    print("=" * 78)
    print(f"模型与预测已保存到 {out_dir}")
    print("下一步： python 06_evaluate.py --pred_dir %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
