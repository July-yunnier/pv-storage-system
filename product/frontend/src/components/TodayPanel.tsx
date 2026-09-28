import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Badge,
  Button,
  Card,
  Col,
  Empty,
  Row,
  Space,
  Statistic,
  Tag,
  Typography,
} from 'antd'
import { ThunderboltOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { Link } from 'react-router-dom'
import type { EChartsOption } from 'echarts'
import EnergyChart from '../charts/EnergyChart'
import {
  createSchedule,
  getCurves,
  getDispatchRun,
  getLive,
  getRolling,
  listDispatchRuns,
  simulateLive,
} from '../api/client'
import type {
  CurvePoint,
  DispatchRunDetail,
  LiveStatus,
  RollingState,
  SchedulePoint,
} from '../api/types'
import { TIER_META } from './tier'

const CHARGE = '#2f7cf6'
const DISCHARGE = '#34a853'
const SOC = '#7c5cff'
const PV = '#f5a623'

function tierTag(tier: string | null) {
  if (!tier) return null
  const meta = TIER_META[tier as keyof typeof TIER_META] ?? {
    label: tier,
    color: '#8c8c8c',
  }
  return <Tag color={meta.color}>{meta.label}</Tag>
}

/** 今日实况图：现在线分古今，左侧实测（定死），右侧原计划 vs 最新修正（动态）。 */
function buildTodayOption(
  points: CurvePoint[],
  basePlan: SchedulePoint[] | null,
  revision: SchedulePoint[] | null,
  nowLabel: string,
): EChartsOption {
  const labels = points.map((p) => dayjs(p.ts).format('HH:mm'))
  const first = labels[0]
  const last = labels[labels.length - 1]
  const labelIndex = new Map(labels.map((l, i) => [l, i]))

  const zeros: (number | null)[] = new Array(labels.length).fill(null)
  const baseCharge = [...zeros]
  const baseDischarge = [...zeros]
  const baseSoc = [...zeros]
  for (const sp of basePlan ?? []) {
    const idx = labelIndex.get(dayjs(sp.ts).format('HH:mm'))
    if (idx === undefined) continue
    if (sp.action === 'charge') baseCharge[idx] = sp.power_kw
    else if (sp.action === 'discharge') baseDischarge[idx] = -sp.power_kw
    baseSoc[idx] = Math.round(sp.soc_end * 1000) / 10
  }
  const revCharge = [...zeros]
  const revDischarge = [...zeros]
  const revSoc = [...zeros]
  for (const sp of revision ?? []) {
    const idx = labelIndex.get(dayjs(sp.ts).format('HH:mm'))
    if (idx === undefined) continue
    if (sp.action === 'charge') revCharge[idx] = sp.power_kw
    else if (sp.action === 'discharge') revDischarge[idx] = -sp.power_kw
    revSoc[idx] = Math.round(sp.soc_end * 1000) / 10
  }

  const hasPlan = !!basePlan && basePlan.length > 0
  const hasRev = !!revision && revision.length > 0
  const yAxis: Record<string, unknown>[] = [
    { type: 'value', name: 'kW' },
    {
      type: 'value',
      name: '电价',
      position: 'right',
      splitLine: { show: false },
      axisLabel: { fontSize: 10 },
    },
  ]
  if (hasPlan) {
    yAxis.push({
      type: 'value',
      name: 'SOC',
      min: 0,
      max: 100,
      position: 'right',
      offset: 48,
      axisLabel: { formatter: '{value}%', fontSize: 10 },
      splitLine: { show: false },
    })
  }

  const series: EChartsOption['series'] = [
    {
      name: '实测光伏',
      type: 'line',
      symbol: 'none',
      connectNulls: false,
      data: points.map((p) => p.pv_power_kw),
      lineStyle: { color: PV, width: 2 },
      itemStyle: { color: PV },
      areaStyle: { color: 'rgba(245, 166, 35, 0.18)' },
      markLine: {
        silent: true,
        symbol: 'none',
        label: {
          formatter: `现在 ${nowLabel}`,
          fontSize: 11,
          color: '#cf1322',
        },
        lineStyle: { color: '#cf1322', type: 'dashed', width: 1.5 },
        data: [{ xAxis: nowLabel }],
      },
      markArea: {
        silent: true,
        data: [
          [
            {
              xAxis: first,
              itemStyle: { color: 'rgba(43, 48, 43, 0.05)' },
              label: { show: false },
            },
            { xAxis: nowLabel },
          ],
          [
            {
              xAxis: nowLabel,
              itemStyle: { color: 'rgba(232, 114, 42, 0.05)' },
              label: { show: false },
            },
            { xAxis: last },
          ],
        ],
      },
    },
    {
      name: '实测负荷',
      type: 'line',
      symbol: 'none',
      connectNulls: false,
      data: points.map((p) => p.load_power_kw),
      lineStyle: { color: '#2b302b', width: 2 },
      itemStyle: { color: '#2b302b' },
    },
    {
      name: '电价',
      type: 'line',
      step: 'end',
      yAxisIndex: 1,
      symbol: 'none',
      data: points.map((p) => p.price),
      lineStyle: { color: '#cf1322', width: 1 },
      itemStyle: { color: '#cf1322' },
    },
  ]
  if (hasPlan) {
    series.push(
      {
        name: '原计划·充电',
        type: 'bar',
        stack: 'base',
        data: baseCharge,
        itemStyle: { color: 'rgba(47, 124, 246, 0.35)' },
      },
      {
        name: '原计划·放电',
        type: 'bar',
        stack: 'base',
        data: baseDischarge,
        itemStyle: { color: 'rgba(52, 168, 83, 0.35)' },
      },
      {
        name: '原计划·SOC',
        type: 'line',
        yAxisIndex: 2,
        symbol: 'none',
        data: baseSoc,
        lineStyle: { type: 'dashed', width: 1, color: 'rgba(124, 92, 255, 0.45)' },
        itemStyle: { color: 'rgba(124, 92, 255, 0.45)' },
      },
    )
  }
  if (hasRev) {
    series.push(
      {
        name: '修正计划·充电',
        type: 'bar',
        stack: 'rev',
        data: revCharge,
        itemStyle: { color: CHARGE },
      },
      {
        name: '修正计划·放电',
        type: 'bar',
        stack: 'rev',
        data: revDischarge,
        itemStyle: { color: DISCHARGE },
      },
      {
        name: '修正计划·SOC',
        type: 'line',
        yAxisIndex: 2,
        symbol: 'none',
        data: revSoc,
        lineStyle: { type: 'dashed', width: 2, color: SOC },
        itemStyle: { color: SOC },
      },
    )
  }

  return {
    animation: false,
    grid: { left: 52, right: hasPlan ? 100 : 56, top: 40, bottom: 24 },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLabel: { interval: 7 } },
    yAxis: yAxis as EChartsOption['yAxis'],
    series,
  }
}

export default function TodayPanel({ siteId }: { siteId: string | undefined }) {
  const { message } = AntdApp.useApp()
  const [live, setLive] = useState<LiveStatus | null>(null)
  const [rolling, setRolling] = useState<RollingState | null>(null)
  const [simulating, setSimulating] = useState(false)
  const [makingPlan, setMakingPlan] = useState(false)
  const [points, setPoints] = useState<CurvePoint[] | null>(null)
  const [plan, setPlan] = useState<DispatchRunDetail | null>(null)
  const [curvesError, setCurvesError] = useState<string | null>(null)
  const autoSimulatedRef = useRef(false)

  const todayStr = dayjs().format('YYYY-MM-DD')
  const nowLabel = dayjs().format('HH:mm')

  const loadAll = useCallback(
    async (sid: string) => {
      try {
        const [liveData, rollingData] = await Promise.all([getLive(sid), getRolling(sid)])
        setLive(liveData)
        setRolling(rollingData)
      } catch {
        setLive(null)
        setRolling(null)
      }
    },
    [],
  )

  useEffect(() => {
    if (!siteId) return
    void loadAll(siteId)
    const timer = setInterval(() => void loadAll(siteId), 30000)
    return () => clearInterval(timer)
  }, [siteId, loadAll])

  const runSimulate = useCallback(
    async (sid: string) => {
      setSimulating(true)
      try {
        const result = await simulateLive(sid)
        message.success(
          `已接入模拟实时数据（光伏 ${result.inserted.pv_power ?? 0} 点，截至 ${dayjs().format('HH:mm')}）`,
        )
        await loadAll(sid)
      } catch (e) {
        message.error(e instanceof Error ? e.message : '模拟数据接入失败')
      } finally {
        setSimulating(false)
      }
    },
    [loadAll, message],
  )

  const loadCurves = useCallback(async (sid: string) => {
    setPoints(null)
    setPlan(null)
    setCurvesError(null)
    try {
      const runs = await listDispatchRuns(sid)
      const todayRun = runs.find((r) => r.date === todayStr)
      const [curves, detail] = await Promise.all([
        getCurves(sid, todayStr, todayStr),
        todayRun ? getDispatchRun(sid, todayRun.run_id) : Promise.resolve(null),
      ])
      setPoints(curves.points)
      setPlan(detail)
    } catch (e) {
      setCurvesError(e instanceof Error ? e.message : '今日曲线加载失败')
    }
  }, [todayStr])

  useEffect(() => {
    if (siteId) void loadCurves(siteId)
  }, [siteId, loadCurves])

  // 实测曲线每 2 分钟补拉一次（新槽位到达后图上的"现在线"右侧逐步变成左侧）
  useEffect(() => {
    if (!siteId) return
    const timer = setInterval(() => void loadCurves(siteId), 120000)
    return () => clearInterval(timer)
  }, [siteId, loadCurves])

  // 今日无数据时自动接入一次模拟源（演示友好）
  useEffect(() => {
    if (!siteId || !live || autoSimulatedRef.current) return
    if (live.pv_points_today === 0) {
      autoSimulatedRef.current = true
      void runSimulate(siteId)
    }
  }, [siteId, live, runSimulate])

  const makeTodayPlan = async () => {
    if (!siteId) return
    setMakingPlan(true)
    try {
      const data = await createSchedule({
        site_id: siteId,
        date: todayStr,
        initial_soc: 0.5,
        strategy: 'rule_v1',
        resolution_minutes: 15,
        source: 'auto',
      })
      message.success(`今日计划已生成，预期收益 ${data.expected_benefit_yuan.toFixed(2)} 元`)
      // 重新模拟让 SOC 跟随新计划，再拉取全部状态
      await simulateLive(siteId)
      await Promise.all([loadAll(siteId), loadCurves(siteId)])
    } catch (e) {
      message.error(e instanceof Error ? e.message : '今日计划生成失败')
    } finally {
      setMakingPlan(false)
    }
  }

  const chartOption = useMemo(
    () =>
      points
        ? buildTodayOption(points, plan?.schedule ?? null, rolling?.latest_schedule ?? null, nowLabel)
        : null,
    [points, plan, rolling, nowLabel],
  )

  const price = live?.price_now
  const nextChange = price?.next_change_at ? dayjs(price.next_change_at) : null
  const minutesToChange = nextChange
    ? Math.max(0, nextChange.diff(dayjs(), 'minute'))
    : null
  const latestRevision = rolling?.revisions[rolling.revisions.length - 1] ?? null

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={16}>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="当前光伏功率"
              value={live?.latest_pv ? Math.round(live.latest_pv.value) : '—'}
              suffix={live?.latest_pv ? 'kW' : ''}
              valueStyle={{ color: PV, fontWeight: 600 }}
            />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {live?.latest_pv
                ? `更新于 ${dayjs(live.latest_pv.ts).format('HH:mm')}`
                : '今日暂无实测'}
            </Typography.Text>
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="当前负荷功率"
              value={live?.latest_load ? Math.round(live.latest_load.value) : '—'}
              suffix={live?.latest_load ? 'kW' : ''}
            />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {live?.latest_load
                ? `更新于 ${dayjs(live.latest_load.ts).format('HH:mm')}`
                : '今日暂无实测'}
            </Typography.Text>
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="当前 SOC"
              value={live?.latest_soc ? Math.round(live.latest_soc.value * 100) : '—'}
              suffix={live?.latest_soc ? '%' : ''}
              valueStyle={{ color: SOC, fontWeight: 600 }}
            />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {live?.latest_soc
                ? `更新于 ${dayjs(live.latest_soc.ts).format('HH:mm')}`
                : '今日暂无实测'}
            </Typography.Text>
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="当前电价"
              value={price ? price.price.toFixed(2) : '—'}
              suffix={price ? '元/kWh' : ''}
              valueStyle={{ color: '#cf1322', fontWeight: 600 }}
            />
            <Space size={4}>
              {tierTag(price?.tier ?? null)}
              {minutesToChange != null && nextChange && (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {minutesToChange >= 60
                    ? `${Math.floor(minutesToChange / 60)} 小时后`
                    : `${minutesToChange} 分钟后`}
                  转{price?.next_tier === 'valley' ? '谷段' : price?.next_tier === 'sharp' ? '尖峰' : '下一时段'}
                </Typography.Text>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      <div style={{ marginTop: -8 }}>
        {(() => {
          const stamps = [live?.latest_pv?.ts, live?.latest_load?.ts]
            .filter((t): t is string => !!t)
            .map((t) => dayjs(t))
            .sort((a, b) => b.valueOf() - a.valueOf())
          return stamps.length > 0 ? (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              数据截至 {stamps[0].format('HH:mm')} · 每 15 分钟自动进数，30 秒刷新展示
            </Typography.Text>
          ) : null
        })()}
      </div>

      {live && live.pv_points_today === 0 && (
        <Alert
          type="info"
          showIcon
          icon={<ThunderboltOutlined />}
          message="今日暂无实测数据"
          description={
            <Space>
              <span>可接入模拟实时数据源（按物理链生成此刻之前的今日数据，标记为 simulator）</span>
              <Button
                size="small"
                type="primary"
                loading={simulating}
                onClick={() => siteId && void runSimulate(siteId)}
              >
                接入模拟数据
              </Button>
            </Space>
          }
        />
      )}

      <Card
        title={
          <Space size={8}>
            <span>今日实况（{todayStr}）</span>
            {rolling && rolling.latest_revision > 0 && (
              <Badge
                count={`已修正 ${rolling.latest_revision} 次`}
                color="#e8722a"
                style={{ fontSize: 11 }}
              />
            )}
          </Space>
        }
        extra={
          <Space size={12} wrap>
            {rolling?.deviation_now != null && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                光伏偏差 {Math.round(rolling.deviation_now * 100)}%
              </Typography.Text>
            )}
            {latestRevision && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                最新修正收益 {latestRevision.expected_benefit_yuan.toFixed(2)} 元（
                {latestRevision.trigger_reason}）
              </Typography.Text>
            )}
            {!plan ? (
              <Button type="primary" loading={makingPlan} onClick={makeTodayPlan}>
                生成今日计划
              </Button>
            ) : (
              <Link to={`/dispatch/${plan.run_id}?site=${siteId}`}>查看原计划</Link>
            )}
          </Space>
        }
      >
        {curvesError ? (
          <Alert type="error" showIcon message="今日曲线加载失败" description={curvesError} />
        ) : !chartOption ? (
          <div style={{ textAlign: 'center', padding: 48 }}>
            <Empty description="等待数据…" />
          </div>
        ) : (
          <>
            <EnergyChart option={chartOption} height={400} />
            <Typography.Text
              type="secondary"
              style={{ fontSize: 12, display: 'block', marginTop: 4 }}
            >
              红线左侧为已发生实测（定死），右侧为未来：浅灰柱=今早原计划，彩色柱=最新修正计划。
              自动修正运行中：光伏实测偏差超 15% 即自动重算（30 分钟冷却防抖），无需人工干预；每 30 秒自动刷新
            </Typography.Text>
          </>
        )}
      </Card>
    </Space>
  )
}
