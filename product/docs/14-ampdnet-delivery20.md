# AMPDNet20轮交付模型接入

## 模型与选型

接入AMPDNet_delivery_20261017中的15个已有权重：Solar1/2/6/7/8，15min/1h/4h，full，全年，60/20/20，seed42。选择规则固定，不按测试成绩排名；每组20轮完整训练中验证MAE最低的检查点为部署权重。`training.json`保留20轮曲线与最佳轮次。

部署目录：`product/backend/data/model_releases/ampdnet_delivery20/`。`index.json`列所有模型，单模型JSON记录缩放器、架构、SHA256和来源，`verification.json`记录全部测试窗口推理核验。`.pth`为原权重，`.torchscript.pt`为部署图。运行依赖基础后端加PyTorch、NumPy；不需要从research导入模型。作者、论文、上游commit与许可未核实状态保留在manifest，接入不产生新的上游再分发许可。

## API

`POST /api/v1/forecast`，示意：

```json
{
  "site_id": "YOUR_SITE_ID",
  "targets": ["pv_power"],
  "horizon_hours": 4,
  "resolution_minutes": 15,
  "issued_at": "2020-10-01T10:00:00+08:00",
  "options": {
    "pv_model": "ampdnet",
    "intraday": true,
    "ampdnet_input": {
      "station": "Solar1",
      "task": "4h",
      "observations": [
        {"ts": "时间戳", "tsi": 400, "dni": 100, "ghi": 150, "temperature_c": 20}
      ]
    }
  }
}
```

以上observations仅示意字段，必须完整提供：15min=48点、1h=96点、4h=192点，`horizon_hours`分别0.25/1/4。辐照W/m²、温度℃，输入顺序为TSI/DNI/GHI/温度，无历史功率。所有时间有时区；按Asia/Shanghai每日06:00—18:45、15分钟槽位保留，18:45下一保留点为次日06:00；不得跳过白天槽位。末点必须等于issued_at，未来观测、乱序和非有限值会被拒绝。

系统站点名义容量必须匹配所选模型：Solar1=50000kW、Solar2=130000kW、Solar6=35000kW、Solar7/8=30000kW。显式选择源站只表示调用该模型，不能证明任意同容量站有跨站泛化能力。站点时区要求Asia/Shanghai。

输出1/4/16个真实未来15分钟功率点；若目标超过同日18:45，接口返回422，而非把次日06:00错误标成15分钟后。原始网络输出经训练前缀minmax反变换（MW）后乘1000为kW，按[0,名义Cap]裁剪；未裁剪输出保留在components.raw_prediction_kw。返回权重哈希、部署图哈希、20轮训练和验证最佳轮次，accuracy和概率区间为空。单次调用只提供短时预测，不拼接为24小时调度曲线，不自动启用现场控制。

## 界面

光伏预测页新增“AMPDNet ·20轮交付模型（实验）”，导入完整请求JSON，在当前选中站点运行并显示预测曲线、模型版本与范围说明。默认auto、统计、物理预测保持可用，未自动切换；历史观测需来自用户授权采集或公开数据，不伪造实时气象输入。

## 验证

```bash
python -m pip install -r product/backend/requirements.txt
python -m pip install 'torch>=2.2,<3' numpy
cd product/backend
python -m pytest -q tests/test_ampdnet_api.py
cd ../frontend
npm ci
npm run build
```

原权重到部署图的全测试集重评需要本机授权的交付压缩包、公开处理CSV与固定commit的作者代码；入口`scripts/release_ampdnet_delivery20.py --extract`，不重训。上传包已提供核验产物，不要求部署机下载原始数据。

## 科学结论边界

20轮训练真实存在；不等于所有模型收敛。原训练采用删夜序列、60/20/20，存在多步划分目标重叠、Solar2统一100MW过滤和Solar8双向插值问题。这些历史限制不能通过导出图或在线拒绝跨夜“修复”。结果用于历史协议实验模型接入，不能称为已完成新的60/10/30主实验或证明真实企业收益。AC可用容量未核实。
