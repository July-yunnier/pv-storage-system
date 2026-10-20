"""Opt-in inference for the fixed 20-epoch AMPDNet delivery release.

Each artifact has its own training-prefix scaler. Training compressed nighttime;
we preserve its input sampling and reject output paths crossing a night boundary.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from app.domain.forecast.result import ForecastModel, ForecastPoint, ForecastProvenance, TargetForecast

RELEASE = Path(__file__).resolve().parents[3] / 'data/model_releases/ampdnet_delivery20'
TASKS = {'15min': (48, 1), '1h': (96, 4), '4h': (192, 16)}
STATIONS = {'Solar1', 'Solar2', 'Solar6', 'Solar7', 'Solar8'}
FEATURE_ATTRIBUTES = ['tsi', 'dni', 'ghi', 'temperature_c']


def _next_daylight(ts: datetime) -> datetime:
    if ts.hour == 18 and ts.minute == 45:
        return (ts + timedelta(days=1)).replace(hour=6, minute=0)
    return ts + timedelta(minutes=15)


def _daylight_slot(ts: datetime) -> bool:
    return (6 <= ts.hour < 19 and ts.minute % 15 == 0
            and ts.second == 0 and ts.microsecond == 0)


def predict_ampdnet(inputs, issued_at: datetime, cap_kw: float, timezone_name: str) -> TargetForecast:
    if inputs.station not in STATIONS or inputs.task not in TASKS:
        raise ValueError('unsupported AMPDNet station/task')
    if timezone_name != 'Asia/Shanghai':
        raise ValueError('交付模型要求Asia/Shanghai时区和明确的源站点映射')
    tz = ZoneInfo('Asia/Shanghai')
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError('AMPDNet issued_at必须包含时区')
    issued_at = issued_at.astimezone(tz)
    if not _daylight_slot(issued_at):
        raise ValueError('AMPDNet仅支持06:00—18:45的15分钟发布时间')
    length, steps = TASKS[inputs.task]
    observations = inputs.observations
    if len(observations) != length:
        raise ValueError(f'所选AMPDNet任务需要{length}个历史白天采样点')
    times = [row.ts.astimezone(tz) for row in observations]
    if any(not _daylight_slot(t) for t in times) or times[-1] != issued_at:
        raise ValueError('历史白天时间槽无效或末点不等于issued_at')
    if any(_next_daylight(left) != right for left, right in zip(times, times[1:])):
        raise ValueError('历史必须按原白天采样连续；禁止白天缺口、重复、未来观测或乱序')
    targets = [issued_at + timedelta(minutes=15*(i+1)) for i in range(steps)]
    if any(t.date() != issued_at.date() or not _daylight_slot(t) for t in targets):
        raise ValueError('拒绝跨夜预测：交付模型的采样步不等于跨夜实际15分钟，当前任务须在18:45前结束')
    tag = f'{inputs.task}_{inputs.station}_processed_3_full_tr0.6_val0.2_s42'
    metadata = json.loads((RELEASE/f'{tag}.json').read_text())
    if abs(cap_kw - metadata['cap_kw']) > 1e-6:
        raise ValueError('站点容量与所选源站模型不一致；不支持跨容量缩放或自动跨站泛化')
    artifact = RELEASE / metadata['artifact_file']
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != metadata['artifact_sha256']:
        raise RuntimeError('AMPDNet模型SHA256校验失败')
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError('AMPDNet需要requirements-ai.txt中的PyTorch和NumPy依赖') from exc
    raw = np.asarray([[getattr(row, feature) for feature in FEATURE_ATTRIBUTES] for row in observations],dtype=np.float32)
    low = np.asarray(metadata['input_min'],dtype=np.float32)
    scale = np.asarray(metadata['input_max'],dtype=np.float32)-low
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('AMPDNet输入或归一化参数无效')
    # Preserve exact float32 minmax arithmetic used by the delivery training.
    x = torch.from_numpy(((raw-low)/scale)[None,:,:])
    model = torch.jit.load(str(artifact),map_location='cpu'); model.eval()
    with torch.inference_mode():
        norm = model(x).numpy()[0]
    if norm.shape != (steps,) or not np.isfinite(norm).all():
        raise RuntimeError('AMPDNet输出无效')
    raw_kw = (norm*np.float32(metadata['output_max_mw']-metadata['output_min_mw'])+np.float32(metadata['output_min_mw']))*np.float32(1000)
    clipped = np.clip(raw_kw,0,metadata['cap_kw'])
    return TargetForecast(
        target='pv_power',unit='kW',
        model=ForecastModel(name=f'ampdnet_{inputs.station.lower()}_{inputs.task}',version='delivery20-full-s42'),
        points=[ForecastPoint(ts=ts,value=round(float(value),6)) for ts,value in zip(targets,clipped)],
        provenance=ForecastProvenance(provider='provided_daylight_observations',window=f'{length}个白天历史采样点；输出{steps}个实际15分钟点',checkpoint_sha256=metadata['checkpoint_sha256'],artifact_sha256=digest,training_status=f'20 epochs completed; validation-selected epoch {metadata["selected_epoch"]}; historical protocol'),
        accuracy=None,p10=None,p90=None,
        components={'raw_prediction_kw':[float(x) for x in raw_kw]},
        notes=f'AMPDNet作者架构；{inputs.station}/full/seed42，20轮训练、验证MAE选第{metadata["selected_epoch"]}轮。输入TSI/DNI/GHI/温度，原白天压缩60/20/20协议；仅同日白天输出，拒绝跨夜。历史多步边界重叠、Solar2过滤/Solar8插值等训练局限未因接入而修复。非统一主实验、非日前或概率预测；未验证跨站泛化。Cap为文件名名义容量，非已核实AC容量；系统输出按[0,Cap]裁剪，原始输出保留于components。',
    )
