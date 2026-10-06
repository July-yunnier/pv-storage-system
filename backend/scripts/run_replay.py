"""跑一次历史滚动回放并导出结果。

用法（在 backend/ 目录下）：
    python scripts/run_replay.py --site S-DONGGUAN-DEMO --days 30

physics/blend 方法需要 Open-Meteo archive 气象代理（历史再分析），
网络受限环境下先设代理：set HTTPS_PROXY=http://127.0.0.1:7897
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.application.replay_service import PV_METHODS, STRATEGIES, run_replay  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402

DB_PATH = BACKEND_DIR / "data" / "storage.db"
EXPORT_DIR = BACKEND_DIR / "data" / "replay"


def main() -> None:
    parser = argparse.ArgumentParser(description="历史滚动回放实验")
    parser.add_argument("--site", default="S-DONGGUAN-DEMO")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--pv-methods", nargs="*", default=None,
                        choices=PV_METHODS, help="默认全部 6 种")
    parser.add_argument("--strategies", nargs="*", default=None,
                        choices=STRATEGIES, help="默认全部 4 种")
    args = parser.parse_args()

    db = Database(str(DB_PATH))
    print(f"回放站点 {args.site}，测试 {args.days} 天 …")
    result = run_replay(
        db, args.site, test_days=args.days,
        pv_methods=args.pv_methods, strategies=args.strategies,
        export_dir=EXPORT_DIR,
    )
    print(f"run_id: {result['run_id']}")
    print(f"结果导出: {EXPORT_DIR / result['run_id']}")
    print("\n== 预测精度汇总（30 天均值）==")
    for method, m in result["summary"]["forecasts"].items():
        print(f"  {method:18s} skill(全天)={m['skill_all_day']:.4f} "
              f"skill(白天)={m['skill_daylight']:.4f} RMSE={m['rmse']:.1f} bias={m['bias']:+.1f}")
    print("\n== 调度收益汇总（30 天合计，元）==")
    for combo, s in result["summary"]["strategies"].items():
        print(f"  {combo:36s} 电费={s['total_cost_yuan']:>10,.0f} "
              f"节省={s['total_saving_yuan']:>8,.0f} 峰值={s['max_peak_import_kw']:.0f} kW")


if __name__ == "__main__":
    main()
