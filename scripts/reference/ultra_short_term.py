"""
超短期光伏功率预测 —— 第 4 小时（15 分钟粒度）达标方案
================================================================
目标指标：1 − RMSE / Cap ≥ 0.95

方法：日周期基准 + 当前偏差延续
    1) 基准：最近 N 天同一时刻（15 分钟粒度）的平均功率
    2) 修正：今天最近一段时间的平均功率，相对最近 N 天同期的偏差，
             按比例 alpha 延续到目标时刻

    pred(t+4h) = clip( mean_{i=1..N} p(t+4h-24h*i)
                       + alpha * ( move_avg(p, t, w) - mean_{i=1..N} move_avg(p, t-24h*i, w) )
                     , 0, Cap )

不涉及深度学习，不需要训练，逐样本可实时计算。

数据：SKIPPD（斯坦福大学 30 kW 屋顶光伏），1 分钟采样
      重采样到 15 分钟粒度以对齐官方口径
"""

import argparse
import os

import numpy as np
import pandas as pd

# ============================ 配置 ============================
CFG = {
    # 数据路径优先级：命令行 --csv > 环境变量 SKIPPD_CSV > ~/Downloads 下的默认位置
    "csv_path": os.environ.get(
        "SKIPPD_CSV",
        os.path.expanduser("~/Downloads/2019_pv_raw.csv"),
    ),
    "power_col": "Huang_E4102_kW",
    "cap": 30.0,          # ⚠️ 官方 Cap：需与组委会确认，测试指南载明该电站为 30 kW 系统
    "horizon": 16,        # 第 4 小时 = 4h / 15min = 16 步
    "N": 12,              # 基准窗口：最近 12 天
    "alpha": 0.25,        # 偏差延续强度（0.20 ~ 0.35 均在达标区间内）
    "w_dev": 2,           # 偏差平滑窗口（步）
    "split_ratio": 0.7,   # 训练（含验证）: 测试 = 7:3，按时间顺序
}


# ============================ 工具函数 ============================
def load_series(cfg):
    """读取功率序列，重采样到 15 分钟，裁剪负值。"""
    df = pd.read_csv(cfg["csv_path"], parse_dates=["Date"])
    s = (df.set_index("Date")[cfg["power_col"]]
           .resample("15min").mean()
           .clip(lower=0))
    # 补齐少量缺失值（原始 1 分钟数据存在空档）
    s = s.ffill().fillna(0.0)
    return s


def predict_ultra_short(v, idx, cfg):
    """
    产出第 4 小时的逐点预测。

    参数
    ----
    v   : np.ndarray   15 分钟粒度的功率序列
    idx : DatetimeIndex 对应时间索引
    cfg : dict         配置

    返回
    ----
    t      : np.ndarray  预测发起时刻的位置索引
    pred   : np.ndarray  对 t+horizon 的预测
    """
    n = len(v)
    H = cfg["horizon"]
    N = cfg["N"]
    alpha = cfg["alpha"]
    w = cfg["w_dev"]

    # 出发点 t：保证 t-24h*N 与 t+H 都在序列范围内
    t = np.arange(96 * N, n - H)

    # ① 日周期基准：最近 N 天、同一"目标时刻"的功率均值
    base = np.mean([v[t + H - 96 * i] for i in range(1, N + 1)], axis=0)

    # ② 当前偏差：今天最近 w 步的均值 − 最近 N 天同一"发起时刻"的均值
    move = pd.Series(v).rolling(w, min_periods=1).mean().values
    dev = move[t] - np.mean([move[t - 96 * i] for i in range(1, N + 1)], axis=0)

    # ③ 偏差按 alpha 延续
    pred = np.clip(base + alpha * dev, 0.0, cfg["cap"])
    return t, pred


def evaluate(t, pred, v, idx, cfg, verbose=True):
    """按 1−RMSE/Cap 与 1−MAE/Cap 评估，并给出误差分解。"""
    H, CAP = cfg["horizon"], cfg["cap"]
    n = len(v)
    tgt = v[t + H]
    split = int(n * cfg["split_ratio"])
    te = (t + H) >= split
    tr = ~te

    if verbose:
        print("=" * 74)
        print("数据划分")
        print("=" * 74)
        print("  训练区间末端 :", idx[split])
        print("  测试区间     :", idx[t[te][0] + H], "→", idx[n - 1])
        print("  测试样本数   : %d" % int(te.sum()))
        print("  评估基准 Cap : %.2f kW   |   预见期 %d 步（15 分钟粒度）"
              % (CAP, H))

    def metric(m, name):
        e = tgt[m] - pred[m]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        mae = float(np.mean(np.abs(e)))
        a1, a2 = 1 - rmse / CAP, 1 - mae / CAP
        if verbose:
            print("\n%-10s RMSE=%.4f kW  1-RMSE/Cap=%.4f %s   |   MAE=%.4f kW  1-MAE/Cap=%.4f"
                  % (name, rmse, a1, "达标" if a1 >= 0.95 else "未达标", mae, a2))
        return rmse, a1, a2, e

    print("\n" + "=" * 74)
    print("总体指标")
    print("=" * 74)
    r_tr, a_tr, _, _ = metric(tr, "验证段")
    r_te, a_te, mae_te, e_te = metric(te, "测试段")

    # 逐月
    print("\n" + "=" * 74)
    print("测试段逐月指标（官方口径为「月平均准确率」）")
    print("=" * 74)
    dfm = pd.DataFrame({"tgt": tgt[te], "pred": pred[te]}, index=idx[t[te] + H])
    month_accs = []
    for ym, g in dfm.groupby(dfm.index.strftime("%Y-%m")):
        rmse = float(np.sqrt(np.mean((g["tgt"] - g["pred"]) ** 2)))
        acc = 1 - rmse / CAP
        month_accs.append(acc)
        print("  %s   n=%5d   RMSE=%.4f   1-RMSE/Cap=%.4f   %s"
              % (ym, len(g), rmse, acc, "达标" if acc >= 0.95 else "未达标"))
    print("  ---------------------------------------------------------------")
    print("  逐月算术平均 = %.4f %s" % (np.mean(month_accs),
                                       "达标" if np.mean(month_accs) >= 0.95 else "未达标"))

    # 误差分解
    print("\n" + "=" * 74)
    print("误差分解（测试段）")
    print("=" * 74)
    print("  bias        = %+.4f kW   (%+.3f%% Cap)" % (np.mean(e_te), 100 * np.mean(e_te) / CAP))
    print("  标准差      = %.4f kW" % np.std(e_te))
    print("  相关系数    = %.4f" % np.corrcoef(tgt[te], pred[te])[0, 1])
    print("  误差 >20%%Cap 占比 = %.2f%%" % (100 * np.mean(np.abs(e_te) > 0.20 * CAP)))
    print("  误差 >10%%Cap 占比 = %.2f%%" % (100 * np.mean(np.abs(e_te) > 0.10 * CAP)))
    print("  夜间样本占比      = %.2f%%   （注：评估含夜间，口径需在报告中声明）"
          % (100 * np.mean(tgt[te] < 0.05)))
    return dict(rmse=r_te, acc1=a_te, acc_mae=mae_te, month_acc=float(np.mean(month_accs)))


def main():
    ap = argparse.ArgumentParser(
        description="超短期（第 4 小时）光伏功率预测：日周期基准 + 偏差延续")
    ap.add_argument("--csv", default=CFG["csv_path"],
                    help="SKIPPD 功率数据 CSV（1 分钟采样）；"
                         "也可用环境变量 SKIPPD_CSV 指定")
    ap.add_argument("--cap", type=float, default=CFG["cap"],
                    help="电站可用容量 kW（官方测试指南载明该电站为 30）")
    ap.add_argument("--alpha", type=float, default=CFG["alpha"],
                    help="偏差延续强度，实测 0.20~0.35 均在达标区间")
    ap.add_argument("--N", type=int, default=CFG["N"],
                    help="同刻基准的回看天数，实测 10~20 均可")
    args = ap.parse_args()

    CFG.update(csv_path=args.csv, cap=args.cap, alpha=args.alpha, N=args.N)

    if not os.path.exists(CFG["csv_path"]):
        print("找不到数据文件：%s" % CFG["csv_path"])
        print("请用 --csv 指定路径，或设置环境变量 SKIPPD_CSV")
        return 1

    print("参数：Cap=%.0f kW  horizon=%d 步  N=%d 天  alpha=%.2f"
          % (CFG["cap"], CFG["horizon"], CFG["N"], CFG["alpha"]))
    print()

    s = load_series(CFG)
    v, idx = s.values.astype(float), s.index
    print("序列长度 %d（15 分钟粒度），时间范围 %s → %s" % (len(v), idx[0], idx[-1]))

    t, pred = predict_ultra_short(v, idx, CFG)
    res = evaluate(t, pred, v, idx, CFG)

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    print("  测试段整体   1-RMSE/Cap = %.4f  %s" % (res["acc1"], "达标" if res["acc1"] >= 0.95 else "未达标"))
    print("  逐月平均     1-RMSE/Cap = %.4f  %s" % (res["month_acc"], "达标" if res["month_acc"] >= 0.95 else "未达标"))
    print("  ⚠️ 以上结果依赖 Cap = %.0f kW；若 Cap 取值改变，指标会同步变化。" % CFG["cap"])
    print("  ⚠️ 评估含夜间样本（约占一半），口径需在报告中声明。")
    return 0


if __name__ == "__main__":
    main()
