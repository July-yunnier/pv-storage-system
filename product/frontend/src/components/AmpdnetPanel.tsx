import { useState } from 'react'
import { Alert, Button, Card, Space, Typography } from 'antd'
import type { EChartsOption } from 'echarts'
import EnergyChart from '../charts/EnergyChart'
import { runForecast } from '../api/client'
import type { ForecastRequest, ForecastResponse } from '../api/types'

/** Historical delivery model is an explicit, timestamp-checked research option. */
export default function AmpdnetPanel({ siteId }: { siteId?: string }) {
  const [request, setRequest] = useState<ForecastRequest | null>(null)
  const [result, setResult] = useState<ForecastResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const forecast = result && result.site_id === siteId ? result.results[0] : null
  const option: EChartsOption = {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', name: '目标时间', data: forecast?.points.map(p => p.ts) ?? [] },
    yAxis: { type: 'value', name: 'kW' },
    series: [{ name: 'AMPDNet预测', type: 'line', data: forecast?.points.map(p => p.value) ?? [] }],
  }
  async function predict() {
    if (!request || !siteId) return
    setError(null); setResult(null); setLoading(true)
    try {
      setResult(await runForecast({ ...request, site_id: siteId }))
    } catch (e) {
      setError(e instanceof Error ? e.message : '预测失败')
    } finally { setLoading(false) }
  }
  return (
    <Card title="AMPDNet · 20轮交付模型（实验）" size="small" style={{ marginTop: 16 }}>
      <Typography.Paragraph type="secondary">
        五个源站点、15分钟/1小时/4小时模型；使用原20轮训练中的验证最佳权重。
        导入包含发布时间和历史TSI、DNI、GHI、温度的请求JSON。当前站点容量须匹配所选源站，
        原协议按白天采样，接口拒绝跨夜输出。尚未验证跨站泛化，默认预测保持原有模型。
      </Typography.Paragraph>
      <Space wrap>
        <input aria-label="导入AMPDNet预测请求JSON" type="file" accept=".json,application/json" onChange={async e => {
          setRequest(null); setResult(null); setError(null)
          const file = e.target.files?.[0]
          if (!file) return
          try {
            const value = JSON.parse(await file.text()) as ForecastRequest
            if (value.options?.pv_model !== 'ampdnet' || !value.options.ampdnet_input || !value.issued_at) {
              throw new Error('请选择包含options.ampdnet_input和issued_at的AMPDNet请求JSON')
            }
            setRequest(value)
          } catch (e) { setError(e instanceof Error ? e.message : 'JSON无效') }
        }} />
        <Button onClick={predict} loading={loading} disabled={!request || !siteId}>运行AMPDNet预测</Button>
        {request?.options?.ampdnet_input && <Typography.Text>
          {request.options.ampdnet_input.station} · {request.options.ampdnet_input.task}
        </Typography.Text>}
      </Space>
      {error && <Alert style={{ marginTop: 12 }} type="warning" showIcon message={error} />}
      {forecast && <>
        <EnergyChart option={option} height={260} />
        <Typography.Paragraph type="secondary">{forecast.notes}</Typography.Paragraph>
      </>}
    </Card>
  )
}
