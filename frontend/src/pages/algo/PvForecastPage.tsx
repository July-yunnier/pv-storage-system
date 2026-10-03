import { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Space, Spin, Table, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import type { EChartsOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import AccuracyPanel, { methodLabel } from '../../components/AccuracyPanel'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import { getCurves, getReplayRun, listReplayRuns, runForecast } from '../../api/client'
import type {
  CurvePoint,
  ForecastResponse,
  ReplayForecastMetric,
  ReplayRunDetail,
} from '../../api/types'

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
  series.push({
    name: '预测功率',
    type: 'line',
    data: target.points.map((p) => Math.round(p.value * 10) / 10),
    smooth: true,
    symbol: 'none',
    lineStyle: { width: 2.5, color: '#e8722a' },
    areaStyle: { color: 'rgba(232,114,42,0.10)' },
  })
  return {
    grid: { left: 56, right: 20, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels, boundaryGap: false },
    yAxis: { type: 'value', name: 'kW' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['预测功率'] },
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
  if (target) {
    for (const p of target.points) {
      const d = dayjs(p.ts)
      const slot = d.hour() * 4 + Math.floor(d.minute() / 15)
      if (slot >= 0 && slot < 96) forecastData[slot] = Math.round(p.value * 10) / 10
    }
    // 接续点：把实测最后一个有效值接到预测起点，虚线不断档
    const firstForecast = forecastData.findIndex((v) => v != null)
    if (firstForecast > 0 && forecastData[firstForecast - 1] == null) {
      forecastData[firstForecast - 1] = measuredData[firstForecast - 1]
    }
  }
  const lastMeasured = measuredData.reduce<number>(
    (acc, v, i) => (v != null ? i : acc),
    -1,
  )
  return {
    grid: { left: 56, right: 20, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels, boundaryGap: false },
    yAxis: { type: 'value', name: 'kW' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['今日实测', '剩余时段预测'] },
    series: [
      {
        name: '今日实测',
        type: 'line',
        data: measuredData,
        symbol: 'none',
        lineStyle: { width: 2.5, color: '#e8722a' },
        areaStyle: { color: 'rgba(232,114,42,0.12)' },
        connectNulls: false,
      },
      {
        name: '剩余时段预测',
        type: 'line',
        data: forecastData,
        symbol: 'none',
        lineStyle: { width: 2, type: 'dashed', color: '#b37feb' },
        connectNulls: false,
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
  const { sites, site, siteId, setSiteId } = useSites()
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [forecastErr, setForecastErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [replay, setReplay] = useState<ReplayRunDetail | null>(null)
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

    listReplayRuns(siteId)
      .then(async (runs) => {
        const done = runs.find((r) => r.status === 'finished')
        setReplay(done ? await getReplayRun(done.run_id) : null)
      })
      .catch(() => setReplay(null))
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

  const forecastRows = Object.entries(
    (replay?.summary?.forecasts ?? {}) as Record<string, ReplayForecastMetric>,
  ).map(([method, m]) => ({ method, ...m }))

  return (
    <div>
      <PageHeader
        title="光伏预测"
        description="气象预报 → pvlib 物理转换 → 实测校准的融合链路；精度以封存对账与回放实验为准"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card
          title={`今日光伏（${dayjs().format('YYYY-MM-DD')}，实测 + 剩余预测）`}
          size="small"
          extra={
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              实线为已发生实测，虚线为日内滚动修正预测（随实测每 15 分钟修正）
            </Typography.Text>
          }
        >
          <EnergyChart option={todayOption} height={260} />
        </Card>

        <Card
          title={`明日光伏预测${site ? `（${site.pv.capacity_kwp} kWp）` : ''}`}
          size="small"
          extra={
            pv && (
              <Space size={16}>
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
            )
          }
        >
          <Spin spinning={loading}>
            {forecastErr ? (
              <Alert type="warning" showIcon message={forecastErr} />
            ) : (
              <EnergyChart option={option} height={300} />
            )}
          </Spin>
          {pv?.notes && (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {pv.notes}
            </Typography.Text>
          )}
        </Card>

        <Card
          title="预测方法对比（回放实验，30 天滚动）"
          size="small"
          extra={
            replay && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {replay.config.test_window?.[0]} ~ {replay.config.test_window?.[1]} ·
                气象代理：再分析数据
              </Typography.Text>
            )
          }
        >
          {forecastRows.length === 0 ? (
            <Typography.Text type="secondary">
              该站点暂无回放实验记录，可在「调度计划」页发起。
            </Typography.Text>
          ) : (
            <Table
              size="small"
              rowKey="method"
              dataSource={forecastRows}
              pagination={false}
              columns={[
                {
                  title: '方法',
                  dataIndex: 'method',
                  render: (m: string) => <Tag>{methodLabel(m)}</Tag>,
                },
                {
                  title: 'skill 全天',
                  dataIndex: 'skill_all_day',
                  render: (v: number | null) => v?.toFixed(4) ?? '—',
                  sorter: (a, b) => (b.skill_all_day ?? 0) - (a.skill_all_day ?? 0),
                  defaultSortOrder: 'ascend',
                },
                {
                  title: 'skill 白天',
                  dataIndex: 'skill_daylight',
                  render: (v: number | null) => v?.toFixed(4) ?? '—',
                },
                { title: 'RMSE (kW)', dataIndex: 'rmse', render: (v: number) => v?.toFixed(1) },
                {
                  title: '偏差 (kW)',
                  dataIndex: 'bias',
                  render: (v: number) => (
                    <Typography.Text type={v < -20 ? 'warning' : undefined}>
                      {v?.toFixed(1)}
                    </Typography.Text>
                  ),
                },
              ]}
            />
          )}
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            口径：skill 以「昨日延续」为基准（1=完美，0=与基准持平）；白天口径仅含日照时段。
            物理法偏差为气象代理坐标与光伏数据源站点不一致所致的系统性偏低，属已知近似。
          </Typography.Text>
        </Card>

        {siteId && <AccuracyPanel siteId={siteId} target="pv_power" />}
      </Space>
    </div>
  )
}
