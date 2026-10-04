import { Card, Col, Row, Space, Statistic, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import type { EChartsOption, LineSeriesOption } from 'echarts'
import EnergyChart from '../charts/EnergyChart'
import { TIER_META } from './tier'
import type { CurvePoint, PlanSegment, ScheduleResponse } from '../api/types'

const SEGMENT_META: Record<PlanSegment['action'], { label: string; color: string }> = {
  charge_pv: { label: '光伏充电', color: '#95d475' },
  charge_grid: { label: '电网充电', color: '#1677ff' },
  discharge: { label: '放电', color: '#389e0d' },
}

const TIER_BG: Record<string, string> = {
  sharp: 'rgba(207,19,34,0.06)',
  peak: 'rgba(250,140,22,0.07)',
  valley: 'rgba(22,119,255,0.07)',
}

function slotLabels(plan: ScheduleResponse): string[] {
  return plan.schedule.map((p) => dayjs(p.ts).format('HH:mm'))
}

/** ① 预测输入：光伏 + 负荷 + 电价（计划就是根据这三条曲线算的）。 */
function buildInputsOption(
  plan: ScheduleResponse,
  measured: CurvePoint[] | undefined,
  labels: string[],
): EChartsOption {
  const measuredPv: (number | null)[] = Array(labels.length).fill(null)
  const measuredLoad: (number | null)[] = Array(labels.length).fill(null)
  if (measured) {
    for (const p of measured) {
      const d = dayjs(p.ts)
      const slot = d.hour() * 4 + Math.floor(d.minute() / 15)
      if (slot < 0 || slot >= labels.length) continue
      if (p.pv_power_kw != null) measuredPv[slot] = Math.round(p.pv_power_kw)
      if (p.load_power_kw != null) measuredLoad[slot] = Math.round(p.load_power_kw)
    }
  }
  const series: NonNullable<EChartsOption['series']> = [
    {
      name: '光伏预测',
      type: 'line',
      data: plan.schedule.map((p) => Math.round(p.pv_power_kw)),
      smooth: true,
      symbol: 'none',
      itemStyle: { color: '#e8722a' },
      lineStyle: { width: 2, color: '#e8722a', type: measured ? 'dashed' : 'solid' },
    },
    {
      name: '负荷预测',
      type: 'line',
      data: plan.schedule.map((p) => Math.round(p.load_power_kw)),
      smooth: true,
      symbol: 'none',
      itemStyle: { color: '#2b302b' },
      lineStyle: { width: 2, color: '#2b302b', type: measured ? 'dashed' : 'solid' },
    },
    {
      name: '电价',
      type: 'line',
      yAxisIndex: 1,
      step: 'end',
      symbol: 'none',
      itemStyle: { color: '#cf1322' },
      lineStyle: { width: 1.5, color: '#cf1322' },
      data: plan.schedule.map((p) => p.price),
    },
  ]
  if (measured) {
    series.unshift(
      {
        name: '光伏实测',
        type: 'line',
        data: measuredPv,
        symbol: 'none',
        itemStyle: { color: '#e8722a' },
        lineStyle: { width: 2.5, color: '#e8722a' },
        connectNulls: false,
      } as LineSeriesOption,
      {
        name: '负荷实测',
        type: 'line',
        data: measuredLoad,
        symbol: 'none',
        itemStyle: { color: '#2b302b' },
        lineStyle: { width: 2.5, color: '#2b302b' },
        connectNulls: false,
      } as LineSeriesOption,
    )
  }
  return {
    grid: { left: 56, right: 64, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels, boundaryGap: false },
    yAxis: [
      { type: 'value', name: 'kW' },
      {
        type: 'value',
        name: '电价',
        min: 0,
        axisLabel: { formatter: '{value}' },
        splitLine: { show: false },
      },
    ],
    tooltip: { trigger: 'axis' },
    legend: {},
    series,
  }
}

/** ② 调度计划主图：峰平谷底色 + 充电分来源堆叠柱 + 放电柱 + SOC 虚线。 */
function buildPlanOption(plan: ScheduleResponse, labels: string[]): EChartsOption {
  const tiers = plan.schedule.map((p) => p.tier)
  const markData: [Record<string, unknown>, Record<string, unknown>][] = []
  let i = 0
  while (i < tiers.length) {
    const tier = tiers[i]
    if (!tier || !TIER_BG[tier]) {
      i += 1
      continue
    }
    let j = i
    while (j + 1 < tiers.length && tiers[j + 1] === tier) j += 1
    markData.push([
      {
        xAxis: labels[i],
        itemStyle: { color: TIER_BG[tier] },
        name: TIER_META[tier]?.label,
      },
      { xAxis: labels[j] },
    ])
    i = j + 1
  }
  const markArea = {
    silent: true,
    label: { show: true, position: 'insideTop' as const, fontSize: 10, color: '#9ba19b' },
    data: markData,
  } as NonNullable<LineSeriesOption['markArea']>

  const pvCharge = plan.schedule.map((p) =>
    p.action === 'charge' ? Math.round(p.charge_from_pv_kw ?? (p.pv_power_kw > p.load_power_kw ? p.power_kw : 0)) : 0,
  )
  const gridCharge = plan.schedule.map((p) =>
    p.action === 'charge' ? Math.round(p.charge_from_grid_kw ?? (p.pv_power_kw > p.load_power_kw ? 0 : p.power_kw)) : 0,
  )
  return {
    grid: { left: 56, right: 64, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels },
    yAxis: [
      { type: 'value', name: 'kW' },
      {
        type: 'value',
        name: 'SOC',
        min: 0,
        max: 100,
        axisLabel: { formatter: '{value}%' },
        splitLine: { show: false },
      },
    ],
    tooltip: { trigger: 'axis' },
    legend: {},
    series: [
      {
        name: '光伏充电',
        type: 'bar',
        stack: 'charge',
        data: pvCharge,
        itemStyle: { color: '#95d475' },
        barWidth: '70%',
      },
      {
        name: '电网充电',
        type: 'bar',
        stack: 'charge',
        data: gridCharge,
        itemStyle: { color: '#1677ff' },
      },
      {
        name: '放电',
        type: 'bar',
        stack: 'discharge',
        data: plan.schedule.map((p) =>
          p.action === 'discharge' ? -Math.round(p.power_kw) : 0,
        ),
        itemStyle: { color: '#389e0d' },
      },
      {
        name: 'SOC',
        type: 'line',
        yAxisIndex: 1,
        symbol: 'none',
        itemStyle: { color: '#722ed1' },
        data: plan.schedule.map((p) => Math.round(p.soc_end * 1000) / 10),
        lineStyle: { type: 'dashed', width: 1.5, color: '#722ed1' },
        markArea,
      },
    ],
  }
}

/** 调度计划视图：预测输入 → 96 点计划 → 为什么这么排（分时段人话）。 */
export default function DispatchPlanView({
  plan,
  measured,
}: {
  plan: ScheduleResponse
  measured?: CurvePoint[]
}) {
  const labels = slotLabels(plan)
  const segments = plan.explanation ?? []

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card
        title="① 预测输入（计划的依据）"
        size="small"
        extra={
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {measured
              ? '实线为今日实测，虚线为计划采用的预测'
              : '光伏 / 负荷预测 + 分时电价，96 点计划由这三条曲线推导'}
          </Typography.Text>
        }
      >
        <EnergyChart option={buildInputsOption(plan, measured, labels)} height={280} />
      </Card>

      <Card
        title="② 充放电计划（96 点）"
        size="small"
        extra={
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            底色=电价峰平谷：谷段囤电（蓝/绿柱）、峰段放电（深绿向下）、紫虚线=SOC
          </Typography.Text>
        }
      >
        <EnergyChart option={buildPlanOption(plan, labels)} height={300} />
      </Card>

      <Card title="③ 为什么这么排（计划说明）" size="small">
        {segments.length === 0 ? (
          <Typography.Text type="secondary">
            该计划无可说明的充放电时段（全天静置）。
          </Typography.Text>
        ) : (
          <Space direction="vertical" size={10} style={{ width: '100%' }}>
            {segments.map((s) => (
              <div key={`${s.start}-${s.action}`} style={{ display: 'flex', gap: 10 }}>
                <Tag
                  color={SEGMENT_META[s.action].color}
                  style={{ minWidth: 64, textAlign: 'center', marginInlineEnd: 0 }}
                >
                  {SEGMENT_META[s.action].label}
                </Tag>
                <Typography.Text style={{ minWidth: 96 }} strong>
                  {s.start}–{s.end}
                </Typography.Text>
                <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                  {s.text}
                </Typography.Text>
              </div>
            ))}
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              其余时段静置：无套利空间或受 SOC / 功率约束。
            </Typography.Text>
          </Space>
        )}
      </Card>

      <Card size="small">
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="预期收益"
              value={plan.expected_benefit_yuan}
              precision={2}
              suffix="元"
              valueStyle={{ color: '#3f8600', fontWeight: 700 }}
            />
          </Col>
          <Col span={6}>
            <Statistic title="基准成本" value={plan.baseline_cost_yuan} precision={2} suffix="元" />
          </Col>
          <Col span={6}>
            <Statistic title="优化后成本" value={plan.optimized_cost_yuan} precision={2} suffix="元" />
          </Col>
          <Col span={6}>
            <Statistic
              title="等效满循环"
              value={plan.equivalent_full_cycles}
              precision={2}
              suffix="次"
            />
          </Col>
        </Row>
        {plan.benefit_breakdown && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            收益构成：放电替代购电 {plan.benefit_breakdown.discharge_value_yuan.toFixed(2)} 元
            − 谷段购电充电 {plan.benefit_breakdown.grid_charge_cost_yuan.toFixed(2)} 元；
            光伏余电免费充入 {plan.benefit_breakdown.surplus_charge_kwh.toFixed(1)} kWh。
          </Typography.Text>
        )}
      </Card>
    </Space>
  )
}
