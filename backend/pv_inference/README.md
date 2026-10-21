# 冻结光伏推理发布包

10份已验证joblib原样冻结。仅含特征构建、加载、infer CLI与验证代码；无训练、GridSearchCV或模型搜索。

## 拟上传目录

解压后保持backend/pv_inference目录。仅新增该目录，不覆盖backend现有app、requirements或服务配置。GitHub main/backend尚未通过只读核对（自动审批服务403），上传前先审查路径并等待用户确认。

```text
backend/pv_inference/
  __init__.py / __main__.py / features.py / infer.py / selftest.py
  manifest.json / requirements-infer.lock.txt / SELFTEST_RESULT.json
  models/  # 10份joblib，字节未变
  verification/  # 40真实参考样例及1份30帧图像样例
  examples/state_grid_1_request.json
  docs/RESEARCH.md / RESOURCES.md
```

## 安装与运行

使用Python3.12。在仓库根目录执行：

```bash
python3.12 -m venv .venv-infer
source .venv-infer/bin/activate
pip install -r backend/pv_inference/requirements-infer.lock.txt
PYTHONPATH=backend python -m pv_inference --model state_grid_1 \
  --input backend/pv_inference/examples/state_grid_1_request.json --output prediction.json
PYTHONPATH=backend python -m pv_inference.selftest --output selftest.json
```

macOS需可用libomp。此本机测试利用已有torch/lib目录作为DYLD_LIBRARY_PATH，仅为本机OpenMP路径，torch不是推理依赖。线上应使用正常安装的OpenMP运行库，不依赖开发者绝对路径。其他平台需锁版本安装、selftest通过后使用。

```python
from pv_inference import load_model
model = load_model('state_grid_1', threads=1)
result = model.predict(observations=history, issued_at=issue_time)
```

state_grid需要64条历史，skippd需要30条分钟历史。ts和issued_at必须带时区、整分钟，时间连续升序，最后ts恰等于issued_at，禁止NaN或未来数据；available_at可选，缺省由调用方保证可用性。原数据负数仪表偏移在预处理时已置零；推理端负功率拒绝，不自动填补缺测。字段及单位见manifest。
skippd_images另传images(30,64,64,3)uint8和image_timestamps；CLI JSON可用descriptors(30,26)及image_timestamps。图像时序必须严格匹配历史行，照片拍摄早于对应功率分钟结束的实际来源语义由上游保证。
输出为15/60/240min点预测和P10/P50/P90，单位kW，P50不是点预测同义词。Advanced predict_matrices需float32、精确有序schema、冻结Cap归一化基线；不建议调用方自行猜特征。

## 加载安全与冻结规则

manifest是加载契约，包含权重哈希/尺寸、特征名/顺序/哈希、包版本、资源需求、输入语义、数据和验证指标。加载前严格校验版本、权重SHA256、schema，失败即拒绝。joblib本质是pickle，只有来自可信渠道的权重和manifest才能加载；校验不能证明恶意pickle安全，替换权重和manifest两者的攻击需靠可信发布摘要/签名防御。
权重不训练、不重序列化、不重选模型。运行时仅设置线程数，磁盘权重哈希保持不变。替换权重或包版本需要新发布号并重验；不得静默回退其他模型。

| 模型 | 冻结SHA256 | Cap/kW |
| --- | --- | --- |
| skippd | 3df1476abc0a8878a955e6b59dc2c561407987f6821f60aed069dd147f884a56 | 30.1 |
| skippd_images | 0c9315cc434249a6fde73eaaa2b4e0969f74dd1ee392a5d2413312cfe675fddb | 30.1 |
| state_grid_1 | 22c65db50b85aeca087e67b2435c11f37ce5919260ed3aab91f051dbbae80ce9 | 50000 |
| state_grid_2 | 7de34dcb23ae3f8419df7351bfc0d62f14e368939e691b119ecd19b8323744a3 | 130000 |
| state_grid_3 | 5e4e4729afb638818146979f2c47219268da1cdf61f34d7efdecab7b37061a08 | 30000 |
| state_grid_4 | 371b40d4a8874adf5693c620580c7ec481b477ab1ce526e2bc7d36d30f9687f5 | 130000 |
| state_grid_5 | 0cf36ec57bf1dfc9774e9ffa2d7728f79c4a182c2504e17950460807e04492cf | 110000 |
| state_grid_6 | 354094a253d923f9ac5661d25b867564861e5efddc6d8a46f583deb330007365 | 35000 |
| state_grid_7 | 516b77fca770b61219111429ff252e31fd93a2786d5811c25be95bb866785f29 | 30000 |
| state_grid_8 | 462235a1ce1ea94defb9a18a28a784144a8726019fd6f8590397a3d262f3b1fb | 30000 |

[科研说明](docs/RESEARCH.md)说明单次时序验证、测试指标、应用范围和全部局限；[资源实测](docs/RESOURCES.md)说明性能的边界。上传前请确认main/backend路径和是否同意上传包内的小量验证历史/图像样例。
[五折CV与7:3审计](docs/CV_REPORT.md)已附入。本轮研究权重因潜在数据泄漏风险已舍弃；当前发布包只包含冻结推理权重。
