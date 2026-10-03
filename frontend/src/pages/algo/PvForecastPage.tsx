import { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Space, Spin, Table, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import type { EChartsOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import AccuracyPanel, { methodLabel } from '../../components/AccuracyPanel'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import { getReplayRun, listReplayRuns, runForecast } from '../../api/client'
import type { ForecastResponse, ReplayForecastMetric, ReplayRunDetail } from '../../api/types'

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

export default function PvForecastPage() {
  const { sites, site, siteId, setSiteId } = useSites()
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [forecastErr, setForecastErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [replay, setReplay] = useState<ReplayRunDetail | null>(null)

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
