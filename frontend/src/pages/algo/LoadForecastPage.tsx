import { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Space, Spin, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import type { EChartsOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import AccuracyPanel, { methodLabel } from '../../components/AccuracyPanel'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import { runForecast } from '../../api/client'
import type { ForecastResponse } from '../../api/types'

function buildLoadOption(forecast: ForecastResponse): EChartsOption {
  const target = forecast.results.find((r) => r.target === 'load_power')
  if (!target) return {}
  const labels = target.points.map((p) => dayjs(p.ts).format('HH:mm'))
  return {
    grid: { left: 56, right: 20, top: 30, bottom: 28 },
    xAxis: { type: 'category', data: labels, boundaryGap: false },
    yAxis: { type: 'value', name: 'kW' },
    tooltip: { trigger: 'axis' },
    series: [
      {
        name: '预测负荷',
        type: 'line',
        data: target.points.map((p) => Math.round(p.value * 10) / 10),
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2.5, color: '#2f54eb' },
        areaStyle: { color: 'rgba(47,84,235,0.10)' },
      },
    ],
  }
}

export default function LoadForecastPage() {
  const { sites, site, siteId, setSiteId } = useSites()
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!siteId) return
    setLoading(true)
    setError(null)
    runForecast({
      site_id: siteId,
      targets: ['load_power'],
      horizon_hours: 24,
      resolution_minutes: 15,
    })
      .then(setForecast)
      .catch((e) => {
        setForecast(null)
        setError(e instanceof Error ? e.message : '预测失败')
      })
      .finally(() => setLoading(false))
  }, [siteId])

  const option = useMemo(
    () => (forecast ? buildLoadOption(forecast) : {}),
    [forecast],
  )
  const load = forecast?.results.find((r) => r.target === 'load_power')
  const dayKwh = load
    ? Math.round(load.points.reduce((s, p) => s + p.value, 0) * 0.25)
    : null
  const peak = load ? Math.round(Math.max(...load.points.map((p) => p.value))) : null

  return (
    <div>
      <PageHeader
        title="负荷预测"
        description="同刻均值 × 工作日/周末日历模式；节假日与异常天气是已知短板，属后续算法工作包"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card
          title={`明日负荷预测${site ? `（峰值约束 ${site.load.peak_kw ?? '—'} kW）` : ''}`}
          size="small"
          extra={
            load && (
              <Space size={16}>
                <Typography.Text type="secondary">
                  日用电量 <b>{dayKwh?.toLocaleString()}</b> kWh
                </Typography.Text>
                <Typography.Text type="secondary">
                  峰值 <b>{peak}</b> kW
                </Typography.Text>
                <Tag color="geekblue">{methodLabel(load.model.name)}</Tag>
              </Space>
            )
          }
        >
          <Spin spinning={loading}>
            {error ? (
              <Alert type="warning" showIcon message={error} />
            ) : (
              <EnergyChart option={option} height={300} />
            )}
          </Spin>
          {load?.notes && (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {load.notes}
            </Typography.Text>
          )}
        </Card>

        {siteId && <AccuracyPanel siteId={siteId} target="load_power" />}
      </Space>
    </div>
  )
}
