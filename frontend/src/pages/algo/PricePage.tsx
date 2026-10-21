import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Card,
  Col,
  Descriptions,
  Empty,
  Row,
  Segmented,
  Select,
  Space,
  Spin,
  Statistic,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  Button,
} from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import type { EChartsOption, LineSeriesOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import AccuracyPanel from '../../components/AccuracyPanel'
import { getPriceForecast, getPriceHistory, getTariff } from '../../api/client'
import type {
  PriceForecastView,
  PriceHistoryView,
  PriceModelKind,
  Tariff,
  TariffTier,
  TierName,
} from '../../api/types'

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

const MODEL_LABEL: Record<PriceModelKind, string> = {
  auto: '自动（有工件用模型，否则统计画像）',
  lgbm: 'LightGBM 分位数',
  statistical: '统计画像（政策+价差分位）',
  tou: '政策表（确定性）',
}

const MODEL_TAG: Record<string, { text: string; color: string }> = {
  price_lgbm_quantile: { text: 'LightGBM 分位数', color: 'volcano' },
  price_diurnal_quantile: { text: '统计画像', color: 'blue' },
  tou_policy: { text: '政策表', color: 'default' },
}

function fmtTime(iso: string): string {
  const d = new Date(iso)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function num(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

export default function PricePage() {
  const { sites, site, siteId, setSiteId } = useSites()
  const [tariff, setTariff] = useState<Tariff | null>(null)
  const [days, setDays] = useState(1)
  const [model, setModel] = useState<PriceModelKind>('auto')
  const [uncertainty, setUncertainty] = useState(true)
  const [forecast, setForecast] = useState<PriceForecastView | null>(null)
  const [history, setHistory] = useState<PriceHistoryView | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    if (!site) return
    getTariff(site.tariff_id)
      .then(setTariff)
      .catch(() => setTariff(null))
  }, [site])

  useEffect(() => {
    if (!siteId) return
    setLoading(true)
    setError(null)
    Promise.all([
      getPriceForecast(siteId, { days, model, uncertainty }),
      getPriceHistory(siteId, 30).catch(() => null),
    ])
      .then(([fc, his]) => {
        setForecast(fc)
        setHistory(his)
      })
      .catch((e: Error) => {
        setForecast(null)
        setError(e.message)
      })
      .finally(() => setLoading(false))
  }, [siteId, days, model, uncertainty, reloadKey])

  const forecastOption = useMemo<EChartsOption>(() => {
    if (!forecast || forecast.points.length === 0) return {}
    const labels = forecast.points.map((p) => fmtTime(p.ts))
    const p10 = forecast.points.map((p) => p.p10 ?? p.p50)
    const band = forecast.points.map((p) => (p.p90 ?? p.p50) - (p.p10 ?? p.p50))
    const p50 = forecast.points.map((p) => p.p50)
    const tou = forecast.points.map((p) => p.tou_price)
    const arbitrage = forecast.summary.arbitrage
    const chargeIdx = arbitrage
      ? forecast.points.findIndex((p) => p.ts === arbitrage.charge_start)
      : -1
    const dischargeIdx = arbitrage
      ? forecast.points.findIndex((p) => p.ts === arbitrage.discharge_start)
      : -1
    const slotCount = arbitrage ? arbitrage.slot_hours * (60 / forecast.resolution_minutes) : 0
    const windowAreas =
      chargeIdx >= 0 && dischargeIdx >= 0
        ? [
            [
              {
                xAxis: labels[chargeIdx],
                itemStyle: { color: 'rgba(47,84,235,0.16)' },
              },
              { xAxis: labels[Math.min(chargeIdx + slotCount - 1, labels.length - 1)] },
            ],
            [
              {
                xAxis: labels[dischargeIdx],
                itemStyle: { color: 'rgba(207,19,34,0.14)' },
              },
              { xAxis: labels[Math.min(dischargeIdx + slotCount - 1, labels.length - 1)] },
            ],
          ]
        : undefined
    const markArea = windowAreas
      ? ({
          silent: true,
          itemStyle: { opacity: 0.35 },
          data: windowAreas,
        } as NonNullable<LineSeriesOption['markArea']>)
      : undefined
    // 竖线标注起点：markArea 的文字在窄区间里会被竖排折行，改用 markLine 顶部标签
    const markLine =
      chargeIdx >= 0 && dischargeIdx >= 0
        ? ({
            silent: true,
            symbol: 'none',
            lineStyle: { type: 'dashed', width: 1, color: '#9ba19b' },
            data: [
              {
                xAxis: labels[chargeIdx],
                lineStyle: { color: '#2f54eb' },
                label: { formatter: '充电起', position: 'end' as const, fontSize: 11, color: '#2f54eb', rotate: 0 },
              },
              {
                xAxis: labels[dischargeIdx],
                lineStyle: { color: '#cf1322' },
                label: { formatter: '放电起', position: 'end' as const, fontSize: 11, color: '#cf1322', rotate: 0 },
              },
            ],
          } as NonNullable<LineSeriesOption['markLine']>)
        : undefined

    return {
      grid: { left: 62, right: 24, top: 40, bottom: 46 },
      legend: { top: 4, itemWidth: 14, textStyle: { fontSize: 12 }, data: ['预测 P50', '目录电价'] },
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const first = (params as { dataIndex: number }[])[0]
          const p = forecast.points[first.dataIndex]
          const tier = p.tier ? ` · ${TIER_LABEL[p.tier]}` : ''
          const band =
            p.p10 !== null && p.p90 !== null
              ? `<br/>区间 P10–P90：${num(p.p10)} ~ ${num(p.p90)}`
              : ''
          return `${fmtTime(p.ts)}${tier}<br/>预测 P50：<b>${num(p.p50)}</b> 元/kWh${band}${
            p.tou_price !== null ? `<br/>目录电价：${num(p.tou_price)} 元/kWh` : ''
          }`
        },
      },
      xAxis: {
        type: 'category',
        data: labels,
        boundaryGap: false,
        axisLabel: { fontSize: 11, hideOverlap: true },
      },
      yAxis: { type: 'value', name: '元/kWh', scale: true, axisLabel: { fontSize: 11 } },
      series: [
        {
          name: 'P10',
          type: 'line',
          stack: 'band',
          data: p10,
          symbol: 'none',
          lineStyle: { width: 0, opacity: 0 },
          itemStyle: { opacity: 0 },
          areaStyle: { color: 'transparent' },
          silent: true,
          tooltip: { show: false },
        },
        {
          name: '区间 P10–P90',
          type: 'line',
          stack: 'band',
          data: band,
          symbol: 'none',
          lineStyle: { width: 0, opacity: 0 },
          itemStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(232,114,42,0.15)' },
          silent: true,
          tooltip: { show: false },
        },
        {
          name: '预测 P50',
          type: 'line',
          data: p50,
          symbol: 'none',
          smooth: true,
          lineStyle: { width: 2.2, color: '#cf1322' },
          markArea,
          markLine,
        },
        {
          name: '目录电价',
          type: 'line',
          step: 'end',
          data: tou,
          symbol: 'none',
          connectNulls: false,
          lineStyle: { width: 1.4, type: 'dashed', color: '#8c8c8c' },
        },
      ],
    }
  }, [forecast])

  const historyOption = useMemo<EChartsOption>(() => {
    if (!history || history.points.length === 0) return {}
    const labels = history.points.map((p) => fmtTime(p.ts))
    const values = history.points.map((p) => p.price)
    return {
      grid: { left: 62, right: 24, top: 24, bottom: 44 },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (v) => `${Number(v).toFixed(3)} 元/kWh`,
      },
      xAxis: {
        type: 'category',
        data: labels,
        boundaryGap: false,
        axisLabel: { fontSize: 11, hideOverlap: true },
      },
      yAxis: { type: 'value', name: '元/kWh', scale: true, axisLabel: { fontSize: 11 } },
      series: [
        {
          name: '历史电价',
          type: 'line',
          data: values,
          symbol: 'none',
          lineStyle: { width: 1.4, color: '#1f6feb' },
          areaStyle: { color: 'rgba(31,111,235,0.10)' },
        },
      ],
    }
  }, [history])

  const refresh = useCallback(() => setReloadKey((k) => k + 1), [])

  const tag = forecast ? MODEL_TAG[forecast.model] : undefined
  const demoSource = forecast ? /gefcom|demo|synthetic/i.test(forecast.data_note) : false
  const statistics = forecast?.summary

  return (
    <div>
      <PageHeader
        title="电价中心"
        description="从政策表到市场价：未来 1–7 天电价分位数预测，并直接折算储能套利窗口"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />

      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card size="small">
          <Space size={16} wrap>
            <Space size={8}>
              <Typography.Text type="secondary">预测区间</Typography.Text>
              <Segmented
                value={days}
                onChange={(v) => setDays(Number(v))}
                options={[
                  { label: '明日 24h', value: 1 },
                  { label: '3 天', value: 3 },
                  { label: '7 天', value: 7 },
                ]}
              />
            </Space>
            <Space size={8}>
              <Typography.Text type="secondary">模型</Typography.Text>
              <Select
                value={model}
                onChange={(v: PriceModelKind) => setModel(v)}
                style={{ width: 280 }}
                options={(Object.keys(MODEL_LABEL) as PriceModelKind[]).map((k) => ({
                  value: k,
                  label: MODEL_LABEL[k],
                }))}
              />
            </Space>
            <Space size={8}>
              <Typography.Text type="secondary">不确定性区间</Typography.Text>
              <Switch checked={uncertainty} onChange={setUncertainty} size="small" />
            </Space>
            <Button size="small" icon={<ReloadOutlined />} onClick={refresh}>
              重新预测
            </Button>
            {forecast && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                发报 {fmtTime(forecast.issued_at)} · 粒度 {forecast.resolution_minutes} 分钟
              </Typography.Text>
            )}
          </Space>
        </Card>

        {demoSource && (
          <Alert
            type="warning"
            showIcon
            message="当前站点使用的是演示用公开市场序列（GEFCom2014-P），不是本地电价"
            description="这条曲线的价格水平与波动形态来自 PJM 区域 1 的历史 LMP，仅用于验证「历史→特征→预测→套利窗口」的完整链路。接入本省现货/代理购电价格后，同一套模型即可给出本地预测。"
          />
        )}
        {error && <Alert type="error" showIcon message="电价预测失败" description={error} />}

        <Spin spinning={loading}>
          <Row gutter={16}>
            <Col span={5}>
              <Card size="small">
                <Statistic
                  title={`预测均价（未来 ${days} 天）`}
                  value={num(statistics?.mean)}
                  suffix="元/kWh"
                  valueStyle={{ fontSize: 20, color: '#cf1322' }}
                />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small">
                <Statistic
                  title="区间最高 / 最低"
                  value={num(statistics?.max)}
                  suffix={`/ ${num(statistics?.min)}`}
                  valueStyle={{ fontSize: 20 }}
                />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small">
                <Statistic
                  title="预测峰谷价差"
                  value={num(statistics?.peak_valley_spread)}
                  suffix="元/kWh"
                  valueStyle={{ fontSize: 20 }}
                />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small">
                <Statistic
                  title="套利净价差（2h 充放）"
                  value={num(statistics?.arbitrage?.net_spread)}
                  suffix="元/kWh"
                  valueStyle={{
                    fontSize: 20,
                    color: (statistics?.arbitrage?.net_spread ?? 0) > 0 ? '#cf1322' : '#3f8600',
                  }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small">
                <Statistic
                  title="预测方法"
                  value={tag?.text ?? (forecast?.model ?? '—')}
                  valueStyle={{ fontSize: 15 }}
                />
                {forecast?.points[0]?.tier && (
                  <Tag color={TIER_COLOR[forecast.points[0].tier]} style={{ marginTop: 6 }}>
                    当前时段 {TIER_LABEL[forecast.points[0].tier]}
                  </Tag>
                )}
              </Card>
            </Col>
          </Row>

          <Card
            title="未来电价曲线（P50 实线 + P10–P90 区间）"
            size="small"
            style={{ marginTop: 16 }}
            extra={
              <Tooltip title={forecast?.method_note}>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  口径说明
                </Typography.Text>
              </Tooltip>
            }
          >
            {forecast && forecast.points.length > 0 ? (
              <>
                <EnergyChart option={forecastOption} height={330} />
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {forecast.method_note}
                </Typography.Text>
              </>
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={loading ? '正在预测…' : '暂无可预测的电价数据'}
              />
            )}
          </Card>

          {statistics?.arbitrage && (
            <Card title="储能套利窗口（由预测曲线直接折算）" size="small" style={{ marginTop: 16 }}>
              <Descriptions size="small" column={4}>
                <Descriptions.Item label="充电起点">
                  {fmtTime(statistics.arbitrage.charge_start)}
                </Descriptions.Item>
                <Descriptions.Item label="放电起点">
                  {fmtTime(statistics.arbitrage.discharge_start)}
                </Descriptions.Item>
                <Descriptions.Item label="充电均价">
                  {num(statistics.arbitrage.charge_price)} 元/kWh
                </Descriptions.Item>
                <Descriptions.Item label="放电均价">
                  {num(statistics.arbitrage.discharge_price)} 元/kWh
                </Descriptions.Item>
                <Descriptions.Item label="毛价差">
                  {num(statistics.arbitrage.gross_spread)} 元/kWh
                </Descriptions.Item>
                <Descriptions.Item label="净价差（含效率）">
                  <Typography.Text strong style={{ color: '#cf1322' }}>
                    {num(statistics.arbitrage.net_spread)} 元/kWh
                  </Typography.Text>
                </Descriptions.Item>
                <Descriptions.Item label="单次时长">
                  {statistics.arbitrage.slot_hours} 小时
                </Descriptions.Item>
                <Descriptions.Item label="往返效率">
                  {(statistics.arbitrage.round_trip_eff * 100).toFixed(0)}%
                </Descriptions.Item>
              </Descriptions>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {statistics.arbitrage.note}
              </Typography.Text>
            </Card>
          )}

          <Card
            title="历史现货电价（近 30 天）"
            size="small"
            style={{ marginTop: 16 }}
            extra={
              history?.stats ? (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  均值 {num(history.stats.mean)} · 最高 {num(history.stats.max)} · 最低{' '}
                  {num(history.stats.min)} · 负电价 {history.stats.negative_hours} 小时
                </Typography.Text>
              ) : null
            }
          >
            {history && history.points.length > 0 ? (
              <EnergyChart option={historyOption} height={220} />
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="站点尚无现货电价历史：导入 CSV 后即可启用数据驱动预测（见 backend/scripts/import_spot_price.py）"
              />
            )}
          </Card>

          {siteId && (
            <Card size="small" style={{ marginTop: 16 }}>
              <AccuracyPanel siteId={siteId} target="price" days={60} />
            </Card>
          )}
        </Spin>

        <Card title="分时电价时段（政策表）" size="small">
          <Table
            size="small"
            rowKey="name"
            pagination={false}
            dataSource={tariff?.tiers ?? []}
            columns={[
              {
                title: '时段类型',
                dataIndex: 'name',
                render: (n: TierName) => <Tag color={TIER_COLOR[n]}>{TIER_LABEL[n]}</Tag>,
              },
              { title: '价格（元/kWh）', dataIndex: 'price', render: (v: number) => v.toFixed(2) },
              {
                title: '覆盖时段',
                dataIndex: 'hours',
                render: (hours: TariffTier['hours']) => hours.map(String).join('、'),
              },
            ]}
          />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            输配电价与基金附加 {(tariff?.transmission_and_fund ?? 0).toFixed(2)} 元/kWh 含在上方价格口径中。
            政策内时段价格是确定量；未来曲线上的波动来自市场（现货/代理购电）部分。
          </Typography.Text>
        </Card>
      </Space>
    </div>
  )
}
