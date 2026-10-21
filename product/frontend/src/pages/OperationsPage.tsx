import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Col,
  Collapse,
  Empty,
  InputNumber,
  Popover,
  Row,
  Segmented,
  Select,
  Slider,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd'
import { LineChartOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { Link } from 'react-router-dom'
import type { EChartsOption } from 'echarts'
import PageHeader from '../components/PageHeader'
import SourceTag from '../components/SourceTag'
import StatusRing from '../components/StatusRing'
import TodayPanel from '../components/TodayPanel'
import EnergyChart from '../charts/EnergyChart'
import {
  createSchedule,
  getDispatchRun,
  getOverview,
  listDispatchRuns,
  listSites,
  runForecast,
} from '../api/client'
import type {
  DataSource,
  DispatchRunDetail,
  DispatchRunSummary,
  ForecastResponse,
  ScheduleAction,
  SchedulePoint,
  Site,
  SiteOverview,
} from '../api/types'

const ACTION_COLOR: Record<ScheduleAction, string> = {
  charge: '#2f7cf6',
  discharge: '#34a853',
  idle: '#e4e7e2',
}

const ACTION_LABEL: Record<ScheduleAction, string> = {
  charge: '充电',
  discharge: '放电',
  idle: '静置',
}

const MODEL_LABEL: Record<string, string> = {
  pv_diurnal_stats: '统计法',
  pvlib_pvwatts: '物理法',
  pvlib_blend: '融合',
  pv_intraday_deviation: '日内修正',
  load_diurnal_mean: '同刻均值',
  load_mes_multitask_attention: 'MES多任务注意力',
  load_intraday_kalman: '日内卡尔曼修正',
  tou_policy: '分时电价',
  price_mqrnn: 'MQ-RNN电价预测',
  price_deepar: 'DeepAR旧基线',
}

interface Block {
  action: ScheduleAction
  startLabel: string
  endLabel: string
  powerKw: number
  ratio: number
}

/** 把 96 点计划压成连续动作块，用于 24h 时间带。 */
function toBlocks(schedule: SchedulePoint[]): Block[] {
  const blocks: Block[] = []
  for (const p of schedule) {
    const last = blocks[blocks.length - 1]
    const label = dayjs(p.ts).format('HH:mm')
    if (last && last.action === p.action) {
      last.endLabel = label
      last.powerKw = Math.max(last.powerKw, p.power_kw)
      last.ratio += 1 / schedule.length
    } else {
      blocks.push({
        action: p.action,
        startLabel: label,
        endLabel: label,
        powerKw: p.power_kw,
        ratio: 1 / schedule.length,
      })
    }
  }
  return blocks
}

function MiniForecastChart({ option }: { option: EChartsOption }) {
  return <EnergyChart option={option} height={150} />
}

function buildPvOption(forecast: ForecastResponse): EChartsOption {
  const target = forecast.results.find((r) => r.target === 'pv_power')
  if (!target) return {}
  const labels = target.points.map((p) => dayjs(p.ts).format('HH:mm'))
  const series: EChartsOption['series'] = []
  if (target.p10 && target.p90) {
    series.push({
      name: 'p10-p90',
      type: 'line',
      data: target.p90,
      lineStyle: { opacity: 0 },
      stack: 'band',
      symbol: 'none',
      tooltip: { show: false },
    })
    series.push({
      name: '区间',
      type: 'line',
      data: target.p10.map((v, i) => Math.round((target.p90![i] - v) * 100) / 100),
      lineStyle: { opacity: 0 },
      areaStyle: { color: 'rgba(245, 166, 35, 0.18)' },
      stack: 'band',
      symbol: 'none',
      tooltip: { show: false },
    })
  }
  series.push({
    name: '光伏功率',
    type: 'line',
    symbol: 'none',
    data: target.points.map((p) => p.value),
    lineStyle: { color: '#f5a623', width: 2 },
    itemStyle: { color: '#f5a623' },
    areaStyle: { color: 'rgba(245, 166, 35, 0.25)' },
  })
  return {
    animation: false,
    grid: { left: 44, right: 12, top: 8, bottom: 20 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLabel: { interval: 15 } },
    yAxis: { type: 'value', name: 'kW' },
    series,
  }
}

function buildLoadOption(forecast: ForecastResponse): EChartsOption {
  const load = forecast.results.find((r) => r.target === 'load_power')
  if (!load) return {}
  const price = forecast.results.find((r) => r.target === 'price')
  const labels = load.points.map((p) => dayjs(p.ts).format('HH:mm'))
  const series: EChartsOption['series'] = [
    {
      name: '负荷功率',
      type: 'line',
      symbol: 'none',
      data: load.points.map((p) => p.value),
      lineStyle: { color: '#262626', width: 2 },
      itemStyle: { color: '#262626' },
      areaStyle: { color: 'rgba(38, 38, 38, 0.08)' },
    },
  ]
  if (price) {
    series.push({
      name: '电价',
      type: 'line',
      step: 'end',
      yAxisIndex: 1,
      symbol: 'none',
      data: price.points.map((p) => p.value),
      lineStyle: { color: '#cf1322', width: 1 },
      itemStyle: { color: '#cf1322' },
    })
  }
  return {
    animation: false,
    grid: { left: 44, right: 40, top: 8, bottom: 20 },
    legend: { show: false },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLabel: { interval: 15 } },
    yAxis: [
      { type: 'value', name: 'kW' },
      { type: 'value', name: '元/kWh', position: 'right', splitLine: { show: false } },
    ],
    series,
  }
}

function buildPlanOption(schedule: SchedulePoint[]): EChartsOption {
  const labels = schedule.map((p) => dayjs(p.ts).format('HH:mm'))
  return {
    animation: false,
    grid: { left: 48, right: 44, top: 28, bottom: 20 },
    legend: { top: 0 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLabel: { interval: 7 } },
    yAxis: [
      { type: 'value', name: 'kW' },
      {
        type: 'value',
        name: 'SOC',
        min: 0,
        max: 100,
        position: 'right',
        axisLabel: { formatter: '{value}%' },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: '充电',
        type: 'bar',
        stack: 'power',
        data: schedule.map((p) => (p.action === 'charge' ? p.power_kw : 0)),
        itemStyle: { color: ACTION_COLOR.charge },
      },
      {
        name: '放电',
        type: 'bar',
        stack: 'power',
        data: schedule.map((p) => (p.action === 'discharge' ? -p.power_kw : 0)),
        itemStyle: { color: ACTION_COLOR.discharge },
      },
      {
        name: 'SOC',
        type: 'line',
        yAxisIndex: 1,
        symbol: 'none',
        data: schedule.map((p) => Math.round(p.soc_end * 1000) / 10),
        lineStyle: { type: 'dashed', width: 1.5, color: '#7c5cff' },
        itemStyle: { color: '#7c5cff' },
      },
    ],
  }
}

/** 电费去哪了：基准购电 vs 执行计划后的购电，差值即省下的钱。 */
function buildCostCompareOption(schedule: SchedulePoint[]): EChartsOption {
  const labels = schedule.map((p) => dayjs(p.ts).format('HH:mm'))
  return {
    animation: false,
    grid: { left: 48, right: 44, top: 28, bottom: 20 },
    legend: { top: 0 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLabel: { interval: 7 } },
    yAxis: [{ type: 'value', name: 'kW' }],
    series: [
      {
        name: '基准购电（无储能）',
        type: 'line',
        symbol: 'none',
        data: schedule.map((p) => Math.round(Math.max(p.load_power_kw - p.pv_power_kw, 0) * 10) / 10),
        lineStyle: { color: '#bfbfbf', width: 2, type: 'dashed' },
        itemStyle: { color: '#bfbfbf' },
        areaStyle: { color: 'rgba(191, 191, 191, 0.15)' },
      },
      {
        name: '优化后购电（执行计划）',
        type: 'line',
        symbol: 'none',
        data: schedule.map((p) => Math.round(p.grid_import_kw * 10) / 10),
        lineStyle: { color: '#2f7cf6', width: 2 },
        itemStyle: { color: '#2f7cf6' },
        areaStyle: { color: 'rgba(22, 119, 255, 0.12)' },
      },
    ],
  }
}

/** 光伏计算路线：按模型高亮实际走过的节点；每个节点可点开看详情。 */
function PvPipeline({ forecast, site }: { forecast: ForecastResponse; site?: Site }) {
  const target = forecast.results.find((r) => r.target === 'pv_power')
  if (!target) return null
  const name = target.model.name
  const prov = target.provenance
  const explain = prov?.explain
  const arrow = (
    <Typography.Text type="secondary" style={{ margin: '0 2px' }}>
      →
    </Typography.Text>
  )

  const weatherChart: EChartsOption | null = useMemo(() => {
    const weather = explain?.weather
    if (!weather || weather.length === 0) return null
    return {
      animation: false,
      grid: { left: 44, right: 12, top: 24, bottom: 20 },
      legend: { top: 0, textStyle: { fontSize: 11 } },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: weather.map((p) => dayjs(p.ts).format('HH:00')),
        axisLabel: { interval: 3, fontSize: 10 },
      },
      yAxis: { type: 'value', name: 'W/m²', axisLabel: { fontSize: 10 } },
      series: [
        {
          name: 'GHI 总辐射',
          type: 'line',
          symbol: 'none',
          data: weather.map((p) => p.ghi_wm2),
          lineStyle: { color: '#f5a623' },
          itemStyle: { color: '#f5a623' },
        },
        {
          name: 'DNI 直射',
          type: 'line',
          symbol: 'none',
          data: weather.map((p) => p.dni_wm2),
          lineStyle: { color: '#cf1322' },
          itemStyle: { color: '#cf1322' },
        },
        {
          name: 'DHI 散射',
          type: 'line',
          symbol: 'none',
          data: weather.map((p) => p.dhi_wm2),
          lineStyle: { color: '#2f7cf6' },
          itemStyle: { color: '#2f7cf6' },
        },
      ],
    }
  }, [explain])

  const weatherDetail = (
    <div style={{ width: 340 }}>
      <Space direction="vertical" size={4}>
        <Typography.Text strong>气象预报原始输入</Typography.Text>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          数据源 {prov?.provider ?? '—'} · 拉取于{' '}
          {prov?.fetched_at ? dayjs(prov.fetched_at).format('MM-DD HH:mm') : '—'} · {prov?.window ?? ''}
        </Typography.Text>
        {weatherChart ? (
          <EnergyChart option={weatherChart} height={150} />
        ) : (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            原始辐照数据未随本次请求返回（explain 未开启）
          </Typography.Text>
        )}
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          GHI 是水平面接收到的总太阳辐射，是光伏功率的最直接驱动量
        </Typography.Text>
      </Space>
    </div>
  )

  const pvlibDetail = (
    <div style={{ width: 300 }}>
      <Space direction="vertical" size={4}>
        <Typography.Text strong>pvlib 物理转换（五步）</Typography.Text>
        {[
          '1. 太阳位置：按经纬度算每时刻太阳高度角/方位角',
          '2. 转置：Hay-Davies 把水平面辐射换算到组件斜面',
          '3. 温度：Faiman 模型算电池板温度',
          '4. 直流功率：PVWatts（温度系数 −0.4%/°C）',
          '5. 系统损耗与装机容量裁剪',
        ].map((line) => (
          <Typography.Text key={line} style={{ fontSize: 12 }}>
            {line}
          </Typography.Text>
        ))}
        {site && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            站点参数：{site.pv.capacity_kwp} kWp · 倾角 {site.pv.tilt_deg}° · 方位角{' '}
            {site.pv.azimuth_deg}°
          </Typography.Text>
        )}
      </Space>
    </div>
  )

  const backtest = explain?.backtest
  const blendDetail = (
    <div style={{ width: 300 }}>
      <Space direction="vertical" size={4}>
        <Typography.Text strong>实测校准：物理 × 系数</Typography.Text>
        <Typography.Text style={{ fontSize: 12 }}>
          系数 = 回测期实测发电量 ÷ 物理模型回测发电量
        </Typography.Text>
        {backtest ? (
          <>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              回测 {backtest.dates.length} 天（{backtest.dates[0]} ~ {backtest.dates[backtest.dates.length - 1]}）
            </Typography.Text>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              实测 {backtest.sum_measured_kwh.toLocaleString()} kWh ÷ 物理{' '}
              {backtest.sum_physics_kwh.toLocaleString()} kWh ={' '}
              <Typography.Text strong>{backtest.factor}</Typography.Text>
            </Typography.Text>
          </>
        ) : (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            回测明细未随本次请求返回
          </Typography.Text>
        )}
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          物理负责"明天和晴天的差别"，校准负责"这个站和教科书的差别"（积灰、朝向偏差、遮挡）
        </Typography.Text>
      </Space>
    </div>
  )

  const resultDetail = (
    <div style={{ width: 280 }}>
      <Space direction="vertical" size={4}>
        <Typography.Text strong>最终预测</Typography.Text>
        <Typography.Text style={{ fontSize: 12 }}>
          模型 {target.model.name} v{target.model.version}，15 分钟 × 96 点
        </Typography.Text>
        {target.p10 && target.p90 && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            阴影区间 = 按历史回测残差给出的 p10~p90 不确定带（全天混合口径，夜间偏宽是已知近似）
          </Typography.Text>
        )}
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          {target.notes}
        </Typography.Text>
      </Space>
    </div>
  )

  const node = (
    label: string,
    active: boolean,
    detail: ReactNode,
    key: string,
  ) => (
    <Popover key={key} content={detail} placement="bottom">
      <Tag color={active ? 'geekblue' : 'default'} style={{ cursor: 'pointer' }}>
        {label}
      </Tag>
    </Popover>
  )

  const isStats = name === 'pv_diurnal_stats'
  const isBlend = name === 'pvlib_blend'
  const isPhysics = name === 'pvlib_pvwatts'

  return (
    <div style={{ marginTop: 8 }}>
      <Space size={2} wrap style={{ fontSize: 12 }}>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          计算路线（点击节点看详情）：
        </Typography.Text>
        {isStats ? (
          <>
            {node('历史同刻均值', true, resultDetail, 'stats')}
            {arrow}
            {node('明日曲线', true, resultDetail, 'out')}
          </>
        ) : (
          <>
            {node(
              prov?.provider === 'open_meteo' ? 'Open-Meteo 气象预报' : '气象预报',
              true,
              weatherDetail,
              'weather',
            )}
            {arrow}
            {node('pvlib 物理转换', true, pvlibDetail, 'pvlib')}
            {arrow}
            {isBlend
              ? node(
                  `实测校准${prov?.factor != null ? ` ×${prov.factor}` : ''}`,
                  true,
                  blendDetail,
                  'blend',
                )
              : null}
            {isBlend ? arrow : null}
            {isPhysics && arrow}
            {node('明日曲线', true, resultDetail, 'out')}
          </>
        )}
        {prov && (
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            （{[prov.window, prov.fetched_at ? `拉取于 ${dayjs(prov.fetched_at).format('MM-DD HH:mm')}` : null]
              .filter(Boolean)
              .join(' · ')}）
          </Typography.Text>
        )}
      </Space>
    </div>
  )
}

function dailyEnergyKwh(points: { value: number }[]): number {
  return Math.round(points.reduce((s, p) => s + p.value, 0) * 0.25)
}

function dailyPeakKw(points: { value: number }[]): number {
  return Math.round(Math.max(0, ...points.map((p) => p.value)))
}

export default function OperationsPage() {
  const { message } = AntdApp.useApp()
  const tomorrow = useMemo(() => dayjs().add(1, 'day'), [])
  const tomorrowStr = tomorrow.format('YYYY-MM-DD')

  const [sites, setSites] = useState<Site[]>([])
  const [siteId, setSiteId] = useState<string>()
  const [overview, setOverview] = useState<SiteOverview | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [forecastError, setForecastError] = useState<string | null>(null)
  const [runs, setRuns] = useState<DispatchRunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [tomorrowRun, setTomorrowRun] = useState<DispatchRunDetail | null>(null)
  const [planLoading, setPlanLoading] = useState(false)
  const [soc, setSoc] = useState(50)
  const [source, setSource] = useState<'auto' | DataSource>('auto')
  const [generating, setGenerating] = useState(false)

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((data) => {
        if (cancelled) return
        setSites(data)
        setSiteId((prev) =>
          prev && data.some((s) => s.site_id === prev) ? prev : data[0]?.site_id,
        )
      })
      .catch((e) => {
        if (!cancelled)
          message.error(e instanceof Error ? e.message : '站点列表加载失败')
      })
    return () => {
      cancelled = true
    }
  }, [message])

  useEffect(() => {
    if (!siteId) return
    let cancelled = false
    getOverview(siteId)
      .then((data) => {
        if (!cancelled) setOverview(data)
      })
      .catch((e) => {
        if (!cancelled)
          message.error(e instanceof Error ? e.message : '站点信息加载失败')
      })
    return () => {
      cancelled = true
    }
  }, [siteId, message])

  useEffect(() => {
    if (!siteId) return
    let cancelled = false
    setForecast(null)
    setForecastError(null)
    runForecast({
      site_id: siteId,
      targets: ['pv_power', 'load_power', 'price'],
      horizon_hours: 24,
      resolution_minutes: 15,
      options: { explain: true },
    })
      .then((data) => {
        if (!cancelled) setForecast(data)
      })
      .catch((e) => {
        if (!cancelled)
          setForecastError(e instanceof Error ? e.message : '预测加载失败')
      })
    return () => {
      cancelled = true
    }
  }, [siteId])

  const loadRuns = useCallback(async (sid: string) => {
    setRunsLoading(true)
    try {
      setRuns(await listDispatchRuns(sid))
    } catch {
      setRuns([])
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (siteId) void loadRuns(siteId)
    else setRuns([])
  }, [siteId, loadRuns])

  const tomorrowSummary = useMemo(
    () =>
      runs
        .filter((r) => r.date === tomorrowStr)
        .sort((a, b) => dayjs(b.created_at).valueOf() - dayjs(a.created_at).valueOf())[0] ??
      null,
    [runs, tomorrowStr],
  )

  useEffect(() => {
    if (!siteId || !tomorrowSummary) {
      setTomorrowRun(null)
      return
    }
    let cancelled = false
    setPlanLoading(true)
    getDispatchRun(siteId, tomorrowSummary.run_id)
      .then((data) => {
        if (!cancelled) setTomorrowRun(data)
      })
      .catch(() => {
        if (!cancelled) setTomorrowRun(null)
      })
      .finally(() => {
        if (!cancelled) setPlanLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [siteId, tomorrowSummary])

  const generate = async () => {
    if (!siteId) return
    setGenerating(true)
    try {
      const data = await createSchedule({
        site_id: siteId,
        date: tomorrowStr,
        initial_soc: soc / 100,
        strategy: 'rule_v1',
        resolution_minutes: 15,
        source,
      })
      message.success(
        `明日计划已生成，预期收益 ${data.expected_benefit_yuan.toFixed(2)} 元`,
      )
      if (data.run_id) {
        setPlanLoading(true)
        getDispatchRun(siteId, data.run_id)
          .then(setTomorrowRun)
          .catch(() => setTomorrowRun(null))
          .finally(() => setPlanLoading(false))
      }
      void loadRuns(siteId)
    } catch (e) {
      message.error(e instanceof Error ? e.message : '计划生成失败')
    } finally {
      setGenerating(false)
    }
  }

  const pvTarget = forecast?.results.find((r) => r.target === 'pv_power') ?? null
  const loadTarget = forecast?.results.find((r) => r.target === 'load_power') ?? null
  const blocks = tomorrowRun ? toBlocks(tomorrowRun.schedule) : []
  const ds = overview?.data_status ?? null

  const subtitle = overview
    ? [
        overview.site.name,
        `光伏 ${overview.site.pv_capacity_kwp} kWp`,
        `储能 ${overview.site.storage_pcs_kw} kW / ${overview.site.storage_capacity_kwh} kWh`,
        `明日 ${tomorrowStr}`,
      ].join(' · ')
    : `明日 ${tomorrowStr}`

  return (
    <>
      <PageHeader
        title="运营"
        description={subtitle}
        extra={
          <Space>
            <Select
              value={siteId}
              onChange={setSiteId}
              style={{ minWidth: 200 }}
              options={sites.map((s) => ({ value: s.site_id, label: s.name }))}
              placeholder="选择站点"
            />
            <Link to="/curves">
              <Button icon={<LineChartOutlined />}>验数</Button>
            </Link>
          </Space>
        }
      />

      <Tabs
        defaultActiveKey="today"
        items={[
          {
            key: 'today',
            label: '今日',
            children: <TodayPanel siteId={siteId} />,
          },
          {
            key: 'tomorrow',
            label: '明日',
            children: (
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                <Row gutter={16}>
          <Col xs={24} lg={12}>
            <Card
              title={`① 明日光伏预测（${tomorrowStr}）`}
              extra={
                pvTarget ? (
                  <Tag color="orange">
                    {MODEL_LABEL[pvTarget.model.name] ?? pvTarget.model.name}
                  </Tag>
                ) : null
              }
            >
              {forecastError ? (
                <Alert type="warning" showIcon message={forecastError} />
              ) : !pvTarget ? (
                <div style={{ textAlign: 'center', padding: 32 }}>
                  <Spin />
                </div>
              ) : (
                <>
                  <MiniForecastChart option={buildPvOption(forecast!)} />
                  <Space split="|" size={16} style={{ marginTop: 8 }}>
                    <Typography.Text type="secondary">
                      日发电量{' '}
                      <Typography.Text strong>
                        {dailyEnergyKwh(pvTarget.points).toLocaleString()}
                      </Typography.Text>{' '}
                      kWh
                    </Typography.Text>
                    <Typography.Text type="secondary">
                      峰值 <Typography.Text strong>{dailyPeakKw(pvTarget.points)}</Typography.Text>{' '}
                      kW
                    </Typography.Text>
                  </Space>
                  <PvPipeline
                    forecast={forecast!}
                    site={sites.find((s) => s.site_id === siteId)}
                  />
                </>
              )}
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card title={`② 明日负荷预测（${tomorrowStr}）`}>
              {!loadTarget ? (
                forecastError ? (
                  <Alert type="warning" showIcon message={forecastError} />
                ) : (
                  <div style={{ textAlign: 'center', padding: 32 }}>
                    <Spin />
                  </div>
                )
              ) : (
                <>
                  <MiniForecastChart option={buildLoadOption(forecast!)} />
                  <Space split="|" size={16} style={{ marginTop: 8 }}>
                    <Typography.Text type="secondary">
                      日用电量{' '}
                      <Typography.Text strong>
                        {dailyEnergyKwh(loadTarget.points).toLocaleString()}
                      </Typography.Text>{' '}
                      kWh
                    </Typography.Text>
                    <Typography.Text type="secondary">
                      峰值 <Typography.Text strong>{dailyPeakKw(loadTarget.points)}</Typography.Text>{' '}
                      kW
                    </Typography.Text>
                  </Space>
                </>
              )}
            </Card>
          </Col>
        </Row>

        <Card
          title={`③ 明日计划（${tomorrowStr}）`}
          extra={
            <Space size={12} wrap>
              <Typography.Text type="secondary">初始 SOC</Typography.Text>
              <Slider
                min={0}
                max={100}
                value={soc}
                onChange={setSoc}
                style={{ width: 120 }}
                tooltip={{ formatter: (v) => `${v}%` }}
              />
              <InputNumber
                min={0}
                max={100}
                value={soc}
                onChange={(v) => setSoc(v ?? 0)}
                style={{ width: 64 }}
              />
              <Segmented
                value={source}
                onChange={(v) => setSource(v as 'auto' | DataSource)}
                options={[
                  { label: '自动', value: 'auto' },
                  { label: '实测', value: 'measurements' },
                  { label: '预测', value: 'forecast' },
                ]}
              />
              <Button
                type="primary"
                onClick={generate}
                loading={generating}
                disabled={!siteId}
              >
                {tomorrowRun ? '重新生成' : '生成明日计划'}
              </Button>
            </Space>
          }
        >
          {planLoading ? (
            <div style={{ textAlign: 'center', padding: 48 }}>
              <Spin />
            </div>
          ) : !tomorrowRun ? (
            <Empty description="系统将于 23:00 自动生成明日计划（日前调度）；现在生成亦可" />
          ) : (
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              {tomorrowRun.data_source === 'forecast' && (
                <Alert
                  type="info"
                  showIcon
                  message="本计划基于预测数据生成，实际收益以实测为准"
                />
              )}
              <Row gutter={24}>
                <Col xs={12} md={6}>
                  <Statistic
                    title="预期收益（元）"
                    value={tomorrowRun.expected_benefit_yuan}
                    precision={2}
                    valueStyle={{ color: '#3f8600', fontWeight: 600, fontSize: 28 }}
                  />
                </Col>
                <Col xs={12} md={6}>
                  <Statistic
                    title="优化后成本（元）"
                    value={tomorrowRun.optimized_cost_yuan}
                    precision={2}
                  />
                </Col>
                <Col xs={12} md={6}>
                  <Statistic
                    title="等效全循环次数"
                    value={tomorrowRun.equivalent_full_cycles}
                    precision={2}
                  />
                </Col>
                <Col xs={12} md={6}>
                  <Space direction="vertical" size={4}>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      策略 {tomorrowRun.strategy}
                    </Typography.Text>
                    <Space size={4}>
                      <SourceTag source={tomorrowRun.data_source ?? 'measurements'} />
                      {tomorrowSummary?.auto_generated && (
                        <Tag color="orange" style={{ fontSize: 11 }}>
                          自动生成于 {dayjs(tomorrowSummary.created_at).format('HH:mm')}
                        </Tag>
                      )}
                    </Space>
                    <Link to={`/dispatch/${tomorrowRun.run_id}?site=${siteId}`}>
                      查看详情
                    </Link>
                  </Space>
                </Col>
              </Row>

              <div>
                <div
                  style={{
                    display: 'flex',
                    height: 36,
                    borderRadius: 6,
                    overflow: 'hidden',
                  }}
                >
                  {blocks.map((b, i) => (
                    <div
                      key={i}
                      title={`${b.startLabel} ~ ${b.endLabel} ${ACTION_LABEL[b.action]} ${Math.round(b.powerKw)} kW`}
                      style={{
                        flexGrow: b.ratio,
                        flexBasis: 0,
                        background: ACTION_COLOR[b.action],
                        borderRight: '1px solid #fff',
                        minWidth: b.action === 'idle' ? 0 : 2,
                      }}
                    />
                  ))}
                </div>
                <div style={{ position: 'relative', height: 18, marginTop: 2 }}>
                  {['00:00', '06:00', '12:00', '18:00', '24:00'].map((t, i) => (
                    <Typography.Text
                      key={t}
                      type="secondary"
                      style={{
                        fontSize: 11,
                        position: 'absolute',
                        whiteSpace: 'nowrap',
                        left: `${i * 25}%`,
                        transform: i === 4 ? 'translateX(-100%)' : 'none',
                      }}
                    >
                      {t}
                    </Typography.Text>
                  ))}
                </div>
                <Space size={16} style={{ marginTop: 4 }}>
                  {(Object.keys(ACTION_LABEL) as ScheduleAction[]).map((a) => (
                    <Space key={a} size={4}>
                      <span
                        style={{
                          display: 'inline-block',
                          width: 10,
                          height: 10,
                          borderRadius: 2,
                          background: ACTION_COLOR[a],
                        }}
                      />
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        {ACTION_LABEL[a]}
                      </Typography.Text>
                    </Space>
                  ))}
                </Space>
              </div>

              <EnergyChart option={buildPlanOption(tomorrowRun.schedule)} height={240} />

              <div>
                <Typography.Text strong>电费去哪了</Typography.Text>
                <Typography.Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>
                  灰线=没有储能要买多少电，蓝线=执行计划后买多少电，差值就是省下的钱
                </Typography.Text>
                <EnergyChart
                  option={buildCostCompareOption(tomorrowRun.schedule)}
                  height={200}
                />
                {tomorrowRun.benefit_breakdown ? (
                  <Space split="|" size={12} wrap style={{ fontSize: 12 }}>
                    <Typography.Text type="secondary">
                      放电抵购电{' '}
                      <Typography.Text strong style={{ color: '#3f8600' }}>
                        +{tomorrowRun.benefit_breakdown.discharge_value_yuan.toFixed(2)}
                      </Typography.Text>{' '}
                      元（{Math.round(tomorrowRun.benefit_breakdown.discharge_kwh)} kWh 高峰时段）
                    </Typography.Text>
                    <Typography.Text type="secondary">
                      谷段充电成本{' '}
                      <Typography.Text strong style={{ color: '#cf1322' }}>
                        −{tomorrowRun.benefit_breakdown.grid_charge_cost_yuan.toFixed(2)}
                      </Typography.Text>{' '}
                      元（{Math.round(tomorrowRun.benefit_breakdown.grid_charge_kwh)} kWh）
                    </Typography.Text>
                    {tomorrowRun.benefit_breakdown.surplus_charge_kwh > 0 && (
                      <Typography.Text type="secondary">
                        其中光伏余电{' '}
                        <Typography.Text strong>
                          {Math.round(tomorrowRun.benefit_breakdown.surplus_charge_kwh)}
                        </Typography.Text>{' '}
                        kWh（免费）
                      </Typography.Text>
                    )}
                  </Space>
                ) : (
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    该计划生成于收益分解功能之前，重新生成后可查看构成
                  </Typography.Text>
                )}
              </div>
            </Space>
          )}
        </Card>

        <Collapse
          items={[
            {
              key: 'history',
              label: '④ 历史计划',
              children: (
                <Table
                  rowKey="run_id"
                  size="small"
                  loading={runsLoading}
                  pagination={{ pageSize: 8, hideOnSinglePage: true }}
                  dataSource={runs}
                  locale={{
                    emptyText: <Empty description="暂无调度记录" />,
                  }}
                  columns={[
                    {
                      title: '生成时间',
                      dataIndex: 'created_at',
                      render: (v: string) => dayjs(v).format('YYYY-MM-DD HH:mm:ss'),
                    },
                    { title: '计划日期', dataIndex: 'date' },
                    {
                      title: '来源',
                      dataIndex: 'data_source',
                      width: 80,
                      render: (v: DataSource | undefined) => (
                        <SourceTag source={v ?? 'measurements'} />
                      ),
                    },
                    {
                      title: '预期收益（元）',
                      dataIndex: 'expected_benefit_yuan',
                      align: 'right',
                      render: (v: number) => (
                        <span style={{ color: '#3f8600' }}>{v.toFixed(2)}</span>
                      ),
                    },
                    {
                      title: '循环次数',
                      dataIndex: 'equivalent_full_cycles',
                      align: 'right',
                      render: (v: number) => v.toFixed(2),
                    },
                    {
                      title: '操作',
                      key: 'action',
                      width: 80,
                      render: (_: unknown, run: DispatchRunSummary) => (
                        <Link to={`/dispatch/${run.run_id}?site=${run.site_id}`}>
                          查看
                        </Link>
                      ),
                    },
                  ]}
                />
              ),
            },
            {
              key: 'health',
              label: '⑤ 数据健康',
              children: ds ? (
                <Space size={48} style={{ width: '100%', justifyContent: 'center', padding: 8 }}>
                  <StatusRing
                    label="光伏功率"
                    completeness={ds.pv_completeness}
                    status={ds.pv_status}
                  />
                  <StatusRing
                    label="负荷功率"
                    completeness={ds.load_completeness}
                    status={ds.load_status}
                  />
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    锚定最近实测日 {ds.date}
                  </Typography.Text>
                </Space>
              ) : (
                <Empty description="暂无数据状态" />
              ),
            },
          ]}
              />
              </Space>
            ),
          },
        ]}
      />
    </>
  )
}
