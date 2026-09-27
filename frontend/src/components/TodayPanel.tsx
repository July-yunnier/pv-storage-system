import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  App as AntdApp,
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
import EnergyChart from '../charts/EnergyChart'
import { buildEnergyChartOption, mergeScheduleInputs } from '../charts/energyOptions'
import {
  getCurves,
  getDispatchRun,
  getLive,
  listDispatchRuns,
  simulateLive,
} from '../api/client'
import type {
  CurvePoint,
  DispatchRunDetail,
  LiveStatus,
} from '../api/types'
import { TIER_META } from './tier'

function tierTag(tier: string | null) {
  if (!tier) return null
  const meta = TIER_META[tier as keyof typeof TIER_META] ?? {
    label: tier,
    color: '#8c8c8c',
  }
  return <Tag color={meta.color}>{meta.label}</Tag>
}

export default function TodayPanel({ siteId }: { siteId: string | undefined }) {
  const { message } = AntdApp.useApp()
  const [live, setLive] = useState<LiveStatus | null>(null)
  const [simulating, setSimulating] = useState(false)
  const [points, setPoints] = useState<CurvePoint[] | null>(null)
  const [plan, setPlan] = useState<DispatchRunDetail | null>(null)
  const [curvesError, setCurvesError] = useState<string | null>(null)
  const autoSimulatedRef = useRef(false)

  const todayStr = dayjs().format('YYYY-MM-DD')

  const loadLive = useCallback(async (sid: string) => {
    try {
      setLive(await getLive(sid))
    } catch {
      setLive(null)
    }
  }, [])

  useEffect(() => {
    if (!siteId) return
    void loadLive(siteId)
    const timer = setInterval(() => void loadLive(siteId), 30000)
    return () => clearInterval(timer)
  }, [siteId, loadLive])

  const runSimulate = useCallback(
    async (sid: string) => {
      setSimulating(true)
      try {
        const result = await simulateLive(sid)
        message.success(
          `已接入模拟实时数据（光伏 ${result.inserted.pv_power ?? 0} 点，截至 ${dayjs().format('HH:mm')}）`,
        )
        await loadLive(sid)
      } catch (e) {
        message.error(e instanceof Error ? e.message : '模拟数据接入失败')
      } finally {
        setSimulating(false)
      }
    },
    [loadLive, message],
  )

  // 今日无数据时自动接入一次模拟源（演示友好）
  useEffect(() => {
    if (!siteId || !live || autoSimulatedRef.current) return
    if (live.pv_points_today === 0) {
      autoSimulatedRef.current = true
      void runSimulate(siteId)
    }
  }, [siteId, live, runSimulate])

  useEffect(() => {
    if (!siteId) return
    let cancelled = false
    setPoints(null)
    setPlan(null)
    setCurvesError(null)
    const load = async () => {
      try {
        const runs = await listDispatchRuns(siteId)
        const todayRun = runs.find((r) => r.date === todayStr)
        const [curves, detail] = await Promise.all([
          getCurves(siteId, todayStr, todayStr),
          todayRun ? getDispatchRun(siteId, todayRun.run_id) : Promise.resolve(null),
        ])
        if (cancelled) return
        setPoints(curves.points)
        setPlan(detail)
      } catch (e) {
        if (!cancelled)
          setCurvesError(e instanceof Error ? e.message : '今日曲线加载失败')
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [siteId, todayStr])

  const { points: displayPoints, backfilled } = useMemo(
    () => (points ? mergeScheduleInputs(points, plan?.schedule) : { points, backfilled: false }),
    [points, plan],
  )

  const chartOption = useMemo(
    () =>
      displayPoints
        ? buildEnergyChartOption({ points: displayPoints, schedule: plan?.schedule ?? null })
        : null,
    [displayPoints, plan],
  )

  const price = live?.price_now
  const nextChange = price?.next_change_at
    ? dayjs(price.next_change_at)
    : null
  const minutesToChange = nextChange ? Math.max(0, nextChange.diff(dayjs(), 'minute')) : null

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Row gutter={16}>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="当前光伏功率"
              value={live?.latest_pv ? Math.round(live.latest_pv.value) : '—'}
              suffix={live?.latest_pv ? 'kW' : ''}
              valueStyle={{ color: '#f5a623', fontWeight: 600 }}
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
              valueStyle={{ color: '#722ed1', fontWeight: 600 }}
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
        title={`今日实况（${todayStr}）`}
        extra={
          plan ? (
            <Space size={4}>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                今日计划预期收益 {plan.expected_benefit_yuan.toFixed(2)} 元
              </Typography.Text>
              <Link to={`/dispatch/${plan.run_id}?site=${siteId}`}>查看计划</Link>
            </Space>
          ) : (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              今日暂无调度计划（计划自明日起效）
            </Typography.Text>
          )
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
            <EnergyChart option={chartOption} height={360} />
            {backfilled && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                缺失段为计划生成时的预测输入
              </Typography.Text>
            )}
            <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
              实线为今日实测（光伏/负荷），柱状与虚线为今日计划（如有）；每 30 秒自动刷新
            </Typography.Text>
          </>
        )}
      </Card>
    </Space>
  )
}
