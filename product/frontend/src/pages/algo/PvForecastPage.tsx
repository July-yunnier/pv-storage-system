import { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Space, Spin, Tabs, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import type { EChartsOption, LineSeriesOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import { methodLabel } from '../../components/AccuracyPanel'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import AmpdnetPanel from '../../components/AmpdnetPanel'
import { getCurves, runForecast } from '../../api/client'
import type { CurvePoint, ForecastResponse } from '../../api/types'

/** 明日光伏预测曲线（含 p10/p90 区间带）。 */
function buildForecastOption(forecast: ForecastResponse): EChartsOption {
  const target = forecast.results.find((r) => r.target === 'pv_power')
  if (!target) return {}
  const labels = target.points.map((p) => dayjs(p.ts).format('HH:mm'))
  const series: EChartsOption['series'] = []
  if (target.p10 && target.p90) {
    series.push(
      {
        name: 'p90',
        type: 'line',
        data: target.p90,
        lineStyle: { opacity: 0 },
        stack: 'band',
        symbol: 'none',
        tooltip: { show: false },
      },
      {
        name: '置信区间',
        type: 'line',
        data: target.p10.map((v, i) => Math.round((target.p90![i] - v) * 100) / 100),
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(232,114,42,0.15)' },
        stack: 'band',
        symbol: 'none',
        tooltip: { show: false },
      },
    )
  }
  if (target.components?.physics_raw) {
    series.push({
      name: '物理链原始输出',
      type: 'line',
      data: target.components.physics_raw.map((v) => Math.round(v * 10) / 10),
      smooth: true,
      symbol: 'none',
      itemStyle: { color: '#9ba19b' },
      lineStyle: { width: 1.5, type: 'dashed', color: '#9ba19b' },
    })
  }
  series.push({
    name: '预测功率',
    type: 'line',
    data: target.points.map((p) => Math.round(p.value * 10) / 10),
    smooth: true,
    symbol: 'none',
    itemStyle: { color: '#e8722a' },
    lineStyle: { width: 2.5, color: '#e8722a' },
    areaStyle: { color: 'rgba(232,114,42,0.10)' },
  })
  return {
    grid: { left: 56, right: 20, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels, boundaryGap: false },
    yAxis: { type: 'value', name: 'kW' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['物理链原始输出', '预测功率'] },
    series,
  }
}

/** 今日视图：实测实线（00:00~当前）+ 日内滚动预测虚线（当前~24:00）。 */
function buildTodayOption(
  measured: CurvePoint[],
  intraday: ForecastResponse | null,
): EChartsOption {
  const labels = Array.from({ length: 96 }, (_, i) =>
    `${String(Math.floor(i / 4)).padStart(2, '0')}:${String((i % 4) * 15).padStart(2, '0')}`,
  )
  const measuredData: (number | null)[] = Array(96).fill(null)
  for (const p of measured) {
    const d = dayjs(p.ts)
    const slot = d.hour() * 4 + Math.floor(d.minute() / 15)
    if (slot >= 0 && slot < 96 && p.pv_power_kw != null) {
      measuredData[slot] = Math.round(p.pv_power_kw * 10) / 10
    }
  }
  const forecastData: (number | null)[] = Array(96).fill(null)
  const target = intraday?.results.find((r) => r.target === 'pv_power')
  let firstForecast = -1
  if (target) {
    for (const p of target.points) {
      const d = dayjs(p.ts)
      const slot = d.hour() * 4 + Math.floor(d.minute() / 15)
      if (slot >= 0 && slot < 96) forecastData[slot] = Math.round(p.value * 10) / 10
    }
    // 接续点：把实测最后一个有效值接到预测起点，虚线不断档
    firstForecast = forecastData.findIndex((v) => v != null)
    if (firstForecast > 0 && forecastData[firstForecast - 1] == null) {
      forecastData[firstForecast - 1] = measuredData[firstForecast - 1]
    }
  }
  const baselineData: (number | null)[] = Array(96).fill(null)
  const corrData: (number | null)[] = Array(96).fill(null)
  if (target?.components) {
    const { baseline, correction } = target.components
    target.points.forEach((p, i) => {
      const d = dayjs(p.ts)
      const slot = d.hour() * 4 + Math.floor(d.minute() / 15)
      if (slot < 0 || slot >= 96) return
      if (baseline?.[i] != null) baselineData[slot] = Math.round(baseline[i] * 10) / 10
      if (correction?.[i] != null) corrData[slot] = Math.round(correction[i] * 10) / 10
    })
  }
  const lastMeasured = measuredData.reduce<number>(
    (acc, v, i) => (v != null ? i : acc),
    -1,
  )
  // 分尺度分区：① 0~15min 持续外推 ② 15min~4h 短临滚动修正 ③ 4h+ 回气候基线
  const zoneDefs = [
    { name: '①', from: firstForecast, to: firstForecast + 1, color: 'rgba(179,127,235,0.14)' },
    { name: '②', from: firstForecast + 1, to: firstForecast + 17, color: 'rgba(179,127,235,0.06)' },
    { name: '③', from: firstForecast + 17, to: 95, color: 'rgba(155,161,155,0.06)' },
  ].filter((z) => firstForecast >= 0 && z.from <= 95 && z.from <= Math.min(z.to, 95))
  const scaleMarkArea: LineSeriesOption['markArea'] = zoneDefs.length
    ? {
        silent: true,
        label: { show: true, position: 'insideTopLeft' as const, fontSize: 11, color: '#8c8c8c' },
        data: zoneDefs.map((z) => [
          { name: z.name, xAxis: labels[z.from], itemStyle: { color: z.color } },
          { xAxis: labels[Math.min(z.to, 95)] },
        ]) as unknown as NonNullable<LineSeriesOption['markArea']>['data'],
      }
    : undefined
  return {
    grid: { left: 56, right: 20, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels, boundaryGap: false },
    yAxis: { type: 'value', name: 'kW' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['今日实测', '剩余时段预测', '气候基线', '偏差修正'] },
    series: [
      {
        name: '今日实测',
        type: 'line',
        data: measuredData,
        symbol: 'none',
        itemStyle: { color: '#e8722a' },
        lineStyle: { width: 2.5, color: '#e8722a' },
        areaStyle: { color: 'rgba(232,114,42,0.12)' },
        connectNulls: false,
      },
      {
        name: '气候基线',
        type: 'line',
        data: baselineData,
        symbol: 'none',
        itemStyle: { color: '#9ba19b' },
        lineStyle: { width: 1.5, type: 'dotted', color: '#9ba19b' },
        connectNulls: false,
      },
      {
        name: '偏差修正',
        type: 'bar',
        data: corrData,
        itemStyle: { color: 'rgba(179,127,235,0.45)' },
        barWidth: '60%',
      },
      {
        name: '剩余时段预测',
        type: 'line',
        data: forecastData,
        symbol: 'none',
        itemStyle: { color: '#b37feb' },
        lineStyle: { width: 2, type: 'dashed', color: '#b37feb' },
        connectNulls: false,
        markArea: scaleMarkArea,
      },
    ],
    graphic: lastMeasured >= 0
      ? [
          {
            type: 'text',
            left: `${((lastMeasured + 1) / 96) * 88 + 6}%`,
            top: 40,
            style: { text: '▲ 现在', fontSize: 11, fill: '#9ba19b' },
          },
        ]
      : [],
  }
}

export default function PvForecastPage() {
  const { sites, siteId, setSiteId } = useSites()
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [forecastErr, setForecastErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [todayMeasured, setTodayMeasured] = useState<CurvePoint[]>([])
  const [intraday, setIntraday] = useState<ForecastResponse | null>(null)

  useEffect(() => {
    if (!siteId) return
    setLoading(true)
    setForecastErr(null)
    runForecast({
      site_id: siteId,
      targets: ['pv_power'],
      horizon_hours: 24,
      resolution_minutes: 15,
      options: { pv_model: 'auto', explain: true },
    })
      .then(setForecast)
      .catch((e) => {
        setForecast(null)
        setForecastErr(e instanceof Error ? e.message : '预测失败')
      })
      .finally(() => setLoading(false))

    const today = dayjs().format('YYYY-MM-DD')
    getCurves(siteId, today, today)
      .then((c) => setTodayMeasured(c.points))
      .catch(() => setTodayMeasured([]))
    runForecast({
      site_id: siteId,
      targets: ['pv_power'],
      horizon_hours: 24,
      resolution_minutes: 15,
      options: { pv_model: 'auto', intraday: true },
    })
      .then(setIntraday)
      .catch(() => setIntraday(null))
  }, [siteId])

  const option = useMemo(
    () => (forecast ? buildForecastOption(forecast) : {}),
    [forecast],
  )
  const todayOption = useMemo(
    () => buildTodayOption(todayMeasured, intraday),
    [todayMeasured, intraday],
  )
  const pv = forecast?.results.find((r) => r.target === 'pv_power')
  const dayKwh = pv
    ? Math.round(pv.points.reduce((s, p) => s + p.value, 0) * 0.25)
    : null
  const peak = pv ? Math.round(Math.max(...pv.points.map((p) => p.value))) : null

  return (
    <div>
      <PageHeader
        title="光伏预测"
        description="气象预报 → pvlib 物理转换 → 实测校准的融合链路；精度以封存对账与回放实验为准"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Card size="small">
        <Tabs
          items={[
            {
              key: 'today',
              label: `今日 · 实测 + 滚动修正（${dayjs().format('MM-DD')}）`,
              children: (
                <>
                  <Typography.Text
                    type="secondary"
                    style={{ fontSize: 12, display: 'block', marginBottom: 4 }}
                  >
                    实线为已发生实测，虚线为日内滚动修正预测（随实测每 15 分钟修正）
                  </Typography.Text>
                  <EnergyChart option={todayOption} height={320} />
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    计算过程：预测（紫色虚线）= 气候基线（灰点线，近 12 天同刻均值）+
                    偏差修正（紫柱，当前实测偏差随时效衰减叠加，可正可负）。
                    分尺度：① 0~15 分钟 · 持续性外推（近实时）；② 15 分钟~4 小时 ·
                    短临滚动修正（统计基线；站点装天空相机或接入卫星云图算法后，由该算法接管此区间）；
                    ③ 4 小时以上 · 回归气候基线（与次日 NWP 日前预测衔接）。
                  </Typography.Text>
                </>
              ),
            },
            {
              key: 'tomorrow',
              label: '明日 · 日前预测',
              children: (
                <>
                  {pv && (
                    <Space size={16} style={{ marginBottom: 4 }}>
                      <Typography.Text type="secondary">
                        日发电量 <b>{dayKwh?.toLocaleString()}</b> kWh
                      </Typography.Text>
                      <Typography.Text type="secondary">
                        峰值 <b>{peak}</b> kW
                      </Typography.Text>
                      <Tag color="orange">{methodLabel(pv.model.name)}</Tag>
                      {pv.provenance?.factor != null && (
                        <Tag>校准系数 ×{pv.provenance.factor.toFixed(3)}</Tag>
                      )}
                    </Space>
                  )}
                  <Spin spinning={loading}>
                    {forecastErr ? (
                      <Alert type="warning" showIcon message={forecastErr} />
                    ) : (
                      <EnergyChart option={option} height={320} />
                    )}
                  </Spin>
                  {pv?.notes && (
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {pv.notes}
                    </Typography.Text>
                  )}
                </>
              ),
            },
          ]}
        />
      </Card>
      {siteId && <AmpdnetPanel key={siteId} siteId={siteId} />}
    </div>
  )
}
