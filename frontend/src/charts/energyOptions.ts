import dayjs from 'dayjs'
import type { EChartsOption, SeriesOption } from 'echarts'
import type { CurvePoint, SchedulePoint } from '../api/types'

const TIER_FILL: Record<string, string> = {
  sharp: 'rgba(207, 19, 34, 0.12)',
  peak: 'rgba(250, 140, 22, 0.12)',
  flat: 'rgba(140, 140, 140, 0.10)',
  valley: 'rgba(56, 158, 13, 0.12)',
}

type MarkAreaSegment = [
  { xAxis: string; itemStyle: { color: string } },
  { xAxis: string },
]

export interface EnergyChartInput {
  points: CurvePoint[]
  schedule?: SchedulePoint[] | null
  multiDay?: boolean
  showNetLoad?: boolean
}

function tooltipUnit(seriesName: string): string {
  if (seriesName === '电价') return ' 元/kWh'
  if (seriesName === 'SOC') return '%'
  return ' kW'
}

export function buildEnergyChartOption({
  points,
  schedule,
  multiDay = false,
  showNetLoad = true,
}: EnergyChartInput): EChartsOption {
  const fmt = multiDay ? 'MM-DD HH:mm' : 'HH:mm'
  const labels = points.map((p) => dayjs(p.ts).format(fmt))
  const labelIndex = new Map(labels.map((l, i) => [l, i]))

  const areas: MarkAreaSegment[] = []
  let segStart = 0
  for (let i = 1; i <= points.length; i++) {
    if (i === points.length || points[i].tier !== points[segStart].tier) {
      const tier = points[segStart].tier
      areas.push([
        {
          xAxis: labels[segStart],
          itemStyle: { color: TIER_FILL[tier] ?? TIER_FILL.flat },
        },
        { xAxis: labels[i - 1] },
      ])
      segStart = i
    }
  }

  const hasSchedule = !!schedule && schedule.length > 0

  const yAxis: Record<string, unknown>[] = [
    { type: 'value', name: '功率 (kW)' },
    {
      type: 'value',
      name: '电价 (元/kWh)',
      position: 'right',
      nameLocation: 'end',
      nameGap: 8,
      nameTextStyle: { align: 'right' },
      splitLine: { show: false },
    },
  ]
  if (hasSchedule) {
    yAxis.push({
      type: 'value',
      name: 'SOC(%)',
      min: 0,
      max: 100,
      position: 'right',
      offset: 52,
      nameLocation: 'end',
      nameGap: 8,
      nameTextStyle: { align: 'left' },
      axisLabel: { formatter: '{value}%' },
      splitLine: { show: false },
    })
  }

  const series: SeriesOption[] = []

  if (hasSchedule && schedule) {
    const charge: (number | null)[] = new Array(points.length).fill(null)
    const discharge: (number | null)[] = new Array(points.length).fill(null)
    const soc: (number | null)[] = new Array(points.length).fill(null)
    for (const sp of schedule) {
      const idx = labelIndex.get(dayjs(sp.ts).format(fmt))
      if (idx === undefined) continue
      if (sp.action === 'charge') charge[idx] = sp.power_kw
      else if (sp.action === 'discharge') discharge[idx] = -sp.power_kw
      soc[idx] = Math.round(sp.soc_end * 1000) / 10
    }
    series.push(
      {
        name: '充电',
        type: 'bar',
        stack: 'dispatch',
        data: charge,
        itemStyle: { color: '#1677ff' },
      },
      {
        name: '放电',
        type: 'bar',
        stack: 'dispatch',
        data: discharge,
        itemStyle: { color: '#52c41a' },
      },
      {
        name: 'SOC',
        type: 'line',
        yAxisIndex: 2,
        symbol: 'none',
        connectNulls: false,
        data: soc,
        lineStyle: { type: 'dashed', width: 1.5, color: '#722ed1' },
        itemStyle: { color: '#722ed1' },
      },
    )
  }

  series.push(
    {
      name: '光伏',
      type: 'line',
      symbol: 'none',
      connectNulls: false,
      data: points.map((p) => p.pv_power_kw),
      lineStyle: { color: '#f5a623' },
      itemStyle: { color: '#f5a623' },
      areaStyle: { color: 'rgba(245, 166, 35, 0.25)' },
      markArea: { silent: true, data: areas },
    },
    {
      name: '负荷',
      type: 'line',
      symbol: 'none',
      connectNulls: false,
      data: points.map((p) => p.load_power_kw),
      lineStyle: { color: '#262626', width: 1.5 },
      itemStyle: { color: '#262626' },
    },
  )
  if (showNetLoad) {
    series.push({
      name: '净负荷',
      type: 'line',
      symbol: 'none',
      connectNulls: false,
      data: points.map((p) => p.net_load_kw),
      lineStyle: { color: '#8c8c8c', type: 'dashed', width: 1 },
      itemStyle: { color: '#8c8c8c' },
    })
  }
  series.push({
    name: '电价',
    type: 'line',
    step: 'end',
    yAxisIndex: 1,
    symbol: 'none',
    connectNulls: false,
    data: points.map((p) => p.price),
    lineStyle: { color: '#cf1322', width: 1 },
    itemStyle: { color: '#cf1322' },
  })

  return {
    animation: false,
    grid: {
      left: 56,
      right: hasSchedule ? 108 : 64,
      top: 36,
      bottom: 64,
    },
    legend: { top: 0 },
    tooltip: {
      trigger: 'axis',
      formatter: (params: unknown) => {
        const items = params as {
          axisValue: string
          seriesName: string
          marker: string
          value: number | null
        }[]
        if (!items.length) return ''
        const lines = items.map(
          (p) =>
            `${p.marker}${p.seriesName}：${p.value == null ? '-' : p.value}${tooltipUnit(p.seriesName)}`,
        )
        return [items[0].axisValue, ...lines].join('<br/>')
      },
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { hideOverlap: true },
    },
    yAxis: yAxis as EChartsOption['yAxis'],
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
    series,
  }
}

export interface MergedCurves {
  points: CurvePoint[]
  backfilled: boolean
}

// 实测为 null 的点用调度计划生成时的输入（schedule 自带 pv/load）回填；
// 实测有值的点保持实测。schedule 只覆盖其所在日期，其余点不动。
export function mergeScheduleInputs(
  points: CurvePoint[],
  schedule: SchedulePoint[] | null | undefined,
): MergedCurves {
  if (!schedule || schedule.length === 0) return { points, backfilled: false }
  const byTs = new Map(schedule.map((sp) => [dayjs(sp.ts).valueOf(), sp]))
  let backfilled = false
  const merged = points.map((p) => {
    if (p.pv_power_kw != null && p.load_power_kw != null) return p
    const sp = byTs.get(dayjs(p.ts).valueOf())
    if (!sp) return p
    const pv = p.pv_power_kw ?? sp.pv_power_kw
    const load = p.load_power_kw ?? sp.load_power_kw
    backfilled = true
    return {
      ...p,
      pv_power_kw: pv,
      load_power_kw: load,
      net_load_kw:
        p.net_load_kw ??
        (pv != null && load != null
          ? Math.round((load - pv) * 100) / 100
          : null),
    }
  })
  return { points: merged, backfilled }
}
