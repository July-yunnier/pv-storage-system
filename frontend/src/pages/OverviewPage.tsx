import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Card,
  Col,
  Empty,
  List,
  Progress,
  Radio,
  Row,
  Space,
  Spin,
  Statistic,
  Tag,
  Typography,
} from 'antd'
import { Link } from 'react-router-dom'
import dayjs from 'dayjs'
import PageHeader from '../components/PageHeader'
import SourceTag from '../components/SourceTag'
import EnergyChart from '../charts/EnergyChart'
import { buildEnergyChartOption, mergeScheduleInputs } from '../charts/energyOptions'
import {
  getCurves,
  getDispatchRun,
  getOverview,
  listSites,
} from '../api/client'
import type {
  CurvePoint,
  QualityStatus,
  SchedulePoint,
  SiteOverview,
} from '../api/types'

const STATUS_COLOR: Record<string, string> = {
  complete: '#52c41a',
  partial: '#faad14',
  empty: '#f5222d',
}

const STATUS_LABEL: Record<string, string> = {
  complete: '完整',
  partial: '部分缺失',
  empty: '无数据',
}

function toPercent(v: number): number {
  return Math.round((v <= 1 ? v * 100 : v) * 10) / 10
}

function StatusRing({
  label,
  completeness,
  status,
}: {
  label: string
  completeness: number
  status: QualityStatus
}) {
  return (
    <Space direction="vertical" align="center" size={4}>
      <Progress
        type="circle"
        percent={toPercent(completeness)}
        size={88}
        strokeColor={STATUS_COLOR[status] ?? '#faad14'}
      />
      <Space size={4}>
        <Typography.Text type="secondary">{label}</Typography.Text>
        <Tag color={STATUS_COLOR[status]}>{STATUS_LABEL[status] ?? status}</Tag>
      </Space>
    </Space>
  )
}

type CurveMode = 'plan' | 'measured'

export default function OverviewPage() {
  const { message } = AntdApp.useApp()
  const [siteId, setSiteId] = useState<string>()
  const [overview, setOverview] = useState<SiteOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [curveMode, setCurveMode] = useState<CurveMode>('plan')
  const [points, setPoints] = useState<CurvePoint[] | null>(null)
  const [schedule, setSchedule] = useState<SchedulePoint[] | null>(null)
  const [curvesError, setCurvesError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((data) => {
        if (!cancelled) setSiteId(data[0]?.site_id)
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : '站点列表加载失败')
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [message])

  useEffect(() => {
    if (!siteId) return
    let cancelled = false
    setLoading(true)
    setError(null)
    getOverview(siteId)
      .then((data) => {
        if (cancelled) return
        setOverview(data)
        setCurveMode(data.latest_run ? 'plan' : 'measured')
      })
      .catch((e) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : '站点总览加载失败')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [siteId])

  const planDate = overview?.latest_run?.date ?? null
  const measuredDate = overview?.latest_measurement_date ?? null
  const curveDate =
    curveMode === 'plan' ? (planDate ?? measuredDate) : (measuredDate ?? planDate)

  useEffect(() => {
    if (!siteId || !overview || !curveDate) return
    let cancelled = false
    setCurvesError(null)
    setPoints(null)
    setSchedule(null)
    const wantSchedule = curveMode === 'plan' && overview.latest_run
    Promise.all([
      getCurves(siteId, curveDate, curveDate),
      wantSchedule
        ? getDispatchRun(siteId, overview.latest_run!.run_id)
        : Promise.resolve(null),
    ])
      .then(([curves, runDetail]) => {
        if (cancelled) return
        setPoints(curves.points)
        setSchedule(runDetail?.schedule ?? null)
      })
      .catch((e) => {
        if (!cancelled)
          setCurvesError(e instanceof Error ? e.message : '曲线数据加载失败')
      })
    return () => {
      cancelled = true
    }
  }, [siteId, overview, curveDate, curveMode])

  const { points: displayPoints, backfilled } = useMemo(
    () => (points ? mergeScheduleInputs(points, schedule) : { points, backfilled: false }),
    [points, schedule],
  )

  const chartOption = useMemo(
    () =>
      displayPoints
        ? buildEnergyChartOption({ points: displayPoints, schedule, showNetLoad: false })
        : null,
    [displayPoints, schedule],
  )

  const run = overview?.latest_run ?? null
  const ds = overview?.data_status ?? null

  const subtitle = overview
    ? [
        overview.site.name,
        `光伏 ${overview.site.pv_capacity_kwp} kWp`,
        `储能 ${overview.site.storage_pcs_kw} kW / ${overview.site.storage_capacity_kwh} kWh`,
        `数据更新于 ${dayjs(overview.generated_at).format('MM-DD HH:mm')}`,
      ].join(' · ')
    : undefined

  return (
    <>
      <PageHeader title="站点总览" description={subtitle} />
      {loading ? (
        <Card>
          <div style={{ textAlign: 'center', padding: 48 }}>
            <Spin />
          </div>
        </Card>
      ) : error ? (
        <Alert type="error" showIcon message="站点总览加载失败" description={error} />
      ) : overview ? (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Row gutter={16}>
            <Col xs={12} lg={6}>
              <Card>
                {overview.current_soc ? (
                  <>
                    <Statistic
                      title="当前 SOC"
                      value={Math.round(overview.current_soc.value * 100)}
                      suffix="%"
                    />
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      来源时间 {dayjs(overview.current_soc.ts).format('MM-DD HH:mm')}
                    </Typography.Text>
                  </>
                ) : (
                  <>
                    <Statistic title="当前 SOC" value="—" />
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      暂无 SOC 实测数据
                    </Typography.Text>
                  </>
                )}
              </Card>
            </Col>
            <Col xs={12} lg={6}>
              <Card>
                {run ? (
                  <>
                    <Statistic
                      title="最新计划预期收益（元）"
                      value={run.expected_benefit_yuan}
                      precision={2}
                      valueStyle={{ color: '#3f8600', fontWeight: 600 }}
                    />
                    <Space size={4}>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        计划日期 {run.date}
                      </Typography.Text>
                      <SourceTag source={run.data_source} />
                    </Space>
                  </>
                ) : (
                  <Empty description="暂无调度计划" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                )}
              </Card>
            </Col>
            <Col xs={12} lg={6}>
              <Card>
                {run ? (
                  <>
                    <Statistic
                      title="优化后成本（元）"
                      value={run.optimized_cost_yuan}
                      precision={2}
                    />
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      计划日期 {run.date}
                    </Typography.Text>
                  </>
                ) : (
                  <Empty description="暂无调度计划" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                )}
              </Card>
            </Col>
            <Col xs={12} lg={6}>
              <Card>
                {run ? (
                  <>
                    <Statistic
                      title="等效全循环次数"
                      value={run.equivalent_full_cycles}
                      precision={2}
                    />
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      计划日期 {run.date}
                    </Typography.Text>
                  </>
                ) : (
                  <Empty description="暂无调度计划" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                )}
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} lg={12}>
              <Card
                title={`计划摘要（${run?.date ?? '—'}）`}
                extra={
                  run && siteId ? (
                    <Link to={`/dispatch/${run.run_id}?site=${siteId}`}>查看详情</Link>
                  ) : null
                }
              >
                {!run ? (
                  <Empty description="暂无调度计划" />
                ) : run.windows.length === 0 ? (
                  <Empty description="当日无充放电动作" />
                ) : (
                  <>
                    <List
                      size="small"
                      dataSource={run.windows}
                      renderItem={(w) => (
                        <List.Item>
                          <Space size={8}>
                            <Tag color={w.action === 'charge' ? '#1677ff' : '#52c41a'}>
                              {w.action === 'charge' ? '充电' : '放电'}
                            </Tag>
                            <span>
                              {w.start} ~ {w.end} @ {Math.round(w.max_power_kw)} kW
                            </span>
                          </Space>
                        </List.Item>
                      )}
                    />
                    <Space size={4}>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        预期收益 {run.expected_benefit_yuan.toFixed(2)} 元 · 策略{' '}
                        {run.strategy} · 数据来源
                      </Typography.Text>
                      <SourceTag source={run.data_source} />
                    </Space>
                  </>
                )}
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title={`数据状态（${ds?.date ?? '—'}）`}>
                {!ds ? (
                  <Empty description="暂无数据状态" />
                ) : (
                  <>
                    <Space
                      size={48}
                      style={{ width: '100%', justifyContent: 'center', padding: 16 }}
                    >
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
                    </Space>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      锚定最近实测日期
                    </Typography.Text>
                  </>
                )}
              </Card>
            </Col>
          </Row>

          <Card
            title={`功率曲线（${curveDate ?? '—'}）`}
            extra={
              planDate && measuredDate ? (
                <Radio.Group
                  value={curveMode}
                  onChange={(e) => setCurveMode(e.target.value as CurveMode)}
                  optionType="button"
                  buttonStyle="solid"
                  size="small"
                  options={[
                    { label: `计划日 ${planDate}`, value: 'plan' },
                    { label: `最近实测日 ${measuredDate}`, value: 'measured' },
                  ]}
                />
              ) : null
            }
          >
            {!curveDate ? (
              <Empty description="暂无可展示日期" />
            ) : curvesError ? (
              <Alert
                type="error"
                showIcon
                message="曲线数据加载失败"
                description={curvesError}
              />
            ) : !chartOption ? (
              <div style={{ textAlign: 'center', padding: 48 }}>
                <Spin />
              </div>
            ) : (
              <>
                <EnergyChart option={chartOption} height={300} />
                {backfilled && (
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    光伏/负荷缺失段为调度计划生成时的预测输入
                  </Typography.Text>
                )}
              </>
            )}
          </Card>
        </Space>
      ) : null}
    </>
  )
}
