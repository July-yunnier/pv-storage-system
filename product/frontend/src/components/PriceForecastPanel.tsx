import { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Card, Empty, Input, Select, Space, Typography } from 'antd'
import type { EChartsOption } from 'echarts'
import { runForecast } from '../api/client'
import type { TargetForecast } from '../api/types'
import EnergyChart from '../charts/EnergyChart'

export default function PriceForecastPanel({ siteId }: { siteId: string }) {
  const [model, setModel] = useState<'auto' | 'mqrnn'>('auto')
  const [hours, setHours] = useState(24)
  const [seriesId, setSeriesId] = useState('')
  const [result, setResult] = useState<TargetForecast | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  // 站点切换后卸载旧请求对应的面板，避免把另一个站点结果显示在当前站点。
  useEffect(() => { setResult(null); setError('') }, [model, hours, seriesId])

  async function generate() {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await runForecast({ site_id: siteId, targets: ['price'],
        horizon_hours: hours, resolution_minutes: 15,
        options: { price_model: model, price_series_id: seriesId.trim() || undefined, include_uncertainty: true } })
      setResult(response.results[0])
    } catch (e) {
      setError(e instanceof Error ? e.message : '电价预测失败')
    } finally {
      setLoading(false)
    }
  }

  const option = useMemo<EChartsOption>(() => result ? {
    grid: { left: 64, right: 20, top: 40, bottom: 28 },
    legend: { data: ['电价 p50', 'p10', 'p90'] },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', boundaryGap: false,
      data: result.points.map(p => p.ts.slice(5, 16).replace('T', ' ')) },
    yAxis: { type: 'value', name: '元/kWh', scale: true },
    series: [
      { name: '电价 p50', type: 'line', symbol: 'none', data: result.points.map(p => p.value),
        lineStyle: { color: '#cf1322', width: 2 } },
      { name: 'p10', type: 'line', symbol: 'none', data: result.p10 ?? [], lineStyle: { type: 'dashed', width: 1 } },
      { name: 'p90', type: 'line', symbol: 'none', data: result.p90 ?? [], lineStyle: { type: 'dashed', width: 1 } },
    ],
  } : {}, [result])

  return <Card title="日前电价与概率区间" size="small">
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Space wrap>
        <Select aria-label="电价模型" value={model} onChange={setModel} disabled={loading} style={{ width: 240 }}
          options={[{ value: 'auto', label: '遵循站点电价方案' }, { value: 'mqrnn', label: 'MQ-RNN 现货概率预测' }]} />
        <Select aria-label="预测时长" value={hours} onChange={setHours} disabled={loading} style={{ width: 120 }}
          options={[24, 48, 72].map(value => ({ value, label: `${value} 小时` }))} />
        <Input aria-label="价格序列ID" value={seriesId} onChange={e => setSeriesId(e.target.value)} disabled={loading}
          placeholder="市场/节点 ID（默认本站）" style={{ width: 240 }} />
        <Button type="primary" loading={loading} disabled={!siteId} onClick={() => void generate()}>生成日前曲线</Button>
      </Space>
      {error && <Alert type="warning" showIcon message={error} />}
      {result ? <>
        <EnergyChart option={option} height={300} />
        <Typography.Text type="secondary">{result.notes}</Typography.Text>
      </> : !error && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="选择电价方案并生成曲线；MQ-RNN 需先完成本站模型训练和价格历史导入" />}
    </Space>
  </Card>
}
