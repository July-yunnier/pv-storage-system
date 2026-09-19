"""光储协同优化系统。

模块划分
--------
models      数据模型（Site / Measurement / Forecast / Schedule）
tariff      电价 Provider（tou / spot / tou_plus_spot 可插拔）
forecast    预测引擎（光伏功率 / 负荷 / 电价）
optimizer   优化内核（dispatch 日调度 + sizing 容量规划，共用 core）
api         FastAPI 接口层
"""

__version__ = "0.1.0"
