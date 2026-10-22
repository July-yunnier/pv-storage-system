# 示例：T4 命令行工程冒烟

**全部价格和事件为合成示例，不是真实广东市场数据，不作效果证据。**

2026-10-22 使用 tests/test_event_data.py 的确定性 fixture 调用真实 CLI。4 天、30 天训练、3 轮树、cal_days=0、stride=24、seed=42、100 次 bootstrap、块长 2。生成 192 条带时区的成对预测，4 个指标及其 CI；status 为 synthetic_engineering_only，effect_verified 为 false。三个 CSV 的首行均标注“示例”，读取时使用 `pd.read_csv(path, comment="#")`。

该次 CPU 单线程脚本计时 0.866 秒，进程峰值 RSS 168.9 MiB（见 run_meta.json）。仅说明这个小规模输入跑通，不能外推完整真实规模的性能、覆盖率或价格效果。

run_meta.json 保留当次临时输入路径及内容哈希，输入可用 fixture 再生成。从仓库根目录进入 backend 后执行：

```bash
cd backend
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
root = Path(tempfile.mkdtemp(prefix='t4-example-'))
args = runpy.run_path('tests/test_event_data.py')['smoke_inputs'](root)
command = [sys.executable, 'scripts/run_event_ablation.py']
for key, value in vars(args).items():
    command.append('--' + key.replace('_', '-'))
    if not isinstance(value, bool):
        command.append(str(value))
subprocess.run(command, check=True)
print('Example outputs:', args.output)
PY
```

参数、依赖版本、源代码哈希、输入和预测哈希均见 [run_meta.json](run_meta.json)。统计数值只用于核查代码路径，正式验收缺口见 [T4 报告](../t4_event_calendar.md)。
