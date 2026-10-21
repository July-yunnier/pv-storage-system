import { useEffect, useMemo, useState } from 'react'
import { Card, Col, Row, Space, Statistic, Table, Tag, Typography } from 'antd'
import type { EChartsOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import PriceForecastPanel from '../../components/PriceForecastPanel'
import { getTariff } from '../../api/client'
import type { Tariff, TariffTier, TierName } from '../../api/types'

const TIER_LABEL: Record<TierName, string> = {
  sharp: '尖峰',
  peak: '高峰',
  flat: '平段',
  valley: '低谷',
}
const TIER_COLOR: Record<TierName, string> = {
  sharp: '#cf1322',
  peak: '#e8722a',
  flat: '#8c8c8c',
  valley: '#2f54eb',
}

/** 把 hours 段（"08:00-10:00" 或小时数）展开为 96 点的时段索引。 */
function tierCoversSlot(hours: (string | number)[], slot: number): boolean {
  const h = Math.floor(slot / 4)
  for (const raw of hours) {
    if (typeof raw === 'number') {
      if (raw === h) return true
      continue
    }
    const m = raw.match(/(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})/)
    if (!m) continue
    const start = Number(m[1]) * 60 + Number(m[2])
    let end = Number(m[3]) * 60 + Number(m[4])
    if (end === 0) end = 24 * 60 // "19:00-00:00" 跨天到午夜
    const t = h * 60 + 7.5 // 时段中点
    if (end > start ? t >= start && t < end : t >= start || t < end) return true
  }
  return false
}

export default function PricePage() {
  const { sites, site, siteId, setSiteId } = useSites()
  const [tariff, setTariff] = useState<Tariff | null>(null)

  useEffect(() => {
    if (!site) return
    getTariff(site.tariff_id)
      .then(setTariff)
      .catch(() => setTariff(null))
  }, [site])

  const slots = useMemo(() => {
    if (!tariff) return []
    return Array.from({ length: 96 }, (_, i) => {
      const tier = tariff.tiers.find((t) => tierCoversSlot(t.hours, i))
      return { slot: i, tier }
    })
  }, [tariff])

  const option = useMemo<EChartsOption>(() => {
    if (!tariff) return {}
    const labels = Array.from({ length: 96 }, (_, i) =>
      `${String(Math.floor(i / 4)).padStart(2, '0')}:${String((i % 4) * 15).padStart(2, '0')}`,
    )
    return {
      grid: { left: 64, right: 20, top: 30, bottom: 28 },
      xAxis: { type: 'category', data: labels, boundaryGap: false },
      yAxis: { type: 'value', name: '元/kWh' },
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const p = (params as { dataIndex: number; value: number }[])[0]
          const s = slots[p.dataIndex]
          return `${labels[p.dataIndex]}<br/>${s?.tier ? TIER_LABEL[s.tier.name] : ''}：${p.value} 元/kWh`
        },
      },
      series: [
        {
          name: '电价',
          type: 'line',
          step: 'end',
          symbol: 'none',
          data: slots.map((s) => s.tier?.price ?? null),
          lineStyle: { width: 2, color: '#cf1322' },
          areaStyle: { color: 'rgba(207,19,34,0.08)' },
        },
      ],
    }
  }, [tariff, slots])

  const prices = slots.map((s) => s.tier?.price).filter((v): v is number => v != null)
  const maxPrice = prices.length ? Math.max(...prices) : 0
  const minPrice = prices.length ? Math.min(...prices) : 0

  return (
    <div>
      <PageHeader
        title="电价中心"
        description="分时政策价格与 MQ-RNN 现货概率预测；电能量价格叠加一次输配及基金附加"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <PriceForecastPanel key={siteId} siteId={siteId ?? ''} />
        <Row gutter={16}>
          <Col span={6}>
            <Card size="small">
              <Statistic title="电价方案" value={tariff?.name ?? '—'} valueStyle={{ fontSize: 16 }} />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="峰谷价差（元/kWh）"
                value={(maxPrice - minPrice).toFixed(2)}
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic title="最高价（元/kWh）" value={maxPrice.toFixed(2)} />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic title="最低价（元/kWh）" value={minPrice.toFixed(2)} />
            </Card>
          </Col>
        </Row>

        <Card title="24 小时电价曲线" size="small">
          <EnergyChart option={option} height={260} />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            输配电价与基金附加 {(tariff?.transmission_and_fund ?? 0).toFixed(2)} 元/kWh
            为单独附加项；本政策参考曲线与下方时段表仅展示电能量部分。
          </Typography.Text>
        </Card>

        <Card title="时段划分" size="small">
          <Table
            size="small"
            rowKey="name"
            dataSource={tariff?.tiers ?? []}
            pagination={false}
            columns={[
              {
                title: '时段类型',
                dataIndex: 'name',
                render: (n: TierName) => (
                  <Tag color={TIER_COLOR[n]}>{TIER_LABEL[n]}</Tag>
                ),
              },
              { title: '价格（元/kWh）', dataIndex: 'price', render: (v: number) => v.toFixed(2) },
              {
                title: '覆盖时段',
                dataIndex: 'hours',
                render: (hours: TariffTier['hours']) => hours.map(String).join('、'),
              },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}
