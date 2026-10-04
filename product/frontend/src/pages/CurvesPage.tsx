import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  DatePicker,
  Empty,
  Form,
  Radio,
  Select,
  Space,
  Spin,
  Typography,
} from 'antd'
import dayjs, { Dayjs } from 'dayjs'
import { Link } from 'react-router-dom'
import { ArrowLeftOutlined } from '@ant-design/icons'
import PageHeader from '../components/PageHeader'
import EnergyChart from '../charts/EnergyChart'
import { buildEnergyChartOption, mergeScheduleInputs } from '../charts/energyOptions'
import {
  getCurves,
  getDispatchRun,
  getOverview,
  listDispatchRuns,
  listSites,
} from '../api/client'
import type {
  CurvePoint,
  DispatchRunSummary,
  SchedulePoint,
  Site,
} from '../api/types'

type ViewMode = '24h' | '72h'

export default function CurvesPage() {
  const { message } = AntdApp.useApp()
  const [sites, setSites] = useState<Site[]>([])
  const [sitesLoading, setSitesLoading] = useState(true)
  const [selectedSiteId, setSelectedSiteId] = useState<string>()
  const [view, setView] = useState<ViewMode>('24h')
  const [date, setDate] = useState<Dayjs>(dayjs())
  const [runs, setRuns] = useState<DispatchRunSummary[]>([])
  const [overlayRunId, setOverlayRunId] = useState<string>('none')
  const [points, setPoints] = useState<CurvePoint[]>([])
  const [schedule, setSchedule] = useState<SchedulePoint[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [queried, setQueried] = useState(false)
  const [rangeNote, setRangeNote] = useState<string | null>(null)
  const autoQueriedRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((data) => {
        if (cancelled) return
        setSites(data)
        setSelectedSiteId(data[0]?.site_id)
      })
      .catch((e) => {
        if (!cancelled)
          message.error(e instanceof Error ? e.message : '站点列表加载失败')
      })
      .finally(() => {
        if (!cancelled) setSitesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [message])

  const site = sites.find((s) => s.site_id === selectedSiteId)

  const query = useCallback(
    async (overrides?: { dateStr?: string; runId?: string }) => {
      if (!site) return
      setLoading(true)
      setError(null)
      const start = overrides?.dateStr ?? date.format('YYYY-MM-DD')
      const runId = overrides?.runId ?? overlayRunId
      const effectiveRunId = runId === 'none' ? '' : runId
      const end = dayjs(start)
        .add(view === '72h' ? 2 : 0, 'day')
        .format('YYYY-MM-DD')
      try {
        const [curves, runDetail] = await Promise.all([
          getCurves(site.site_id, start, end),
          effectiveRunId
            ? getDispatchRun(site.site_id, effectiveRunId)
            : Promise.resolve(null),
        ])
        setPoints(curves.points)
        setSchedule(runDetail?.schedule ?? null)
        setRangeNote(
          `数据来自 API（GET /sites/${site.site_id}/curves），时间范围 ${curves.start} ~ ${curves.end}，分辨率 ${curves.resolution_minutes} 分钟`,
        )
        setQueried(true)
      } catch (e) {
        setPoints([])
        setSchedule(null)
        setRangeNote(null)
        setError(e instanceof Error ? e.message : '曲线数据加载失败')
        setQueried(true)
      } finally {
        setLoading(false)
      }
    },
    [site, date, view, overlayRunId],
  )

  useEffect(() => {
    autoQueriedRef.current = false
    setRuns([])
    setOverlayRunId('none')
    if (!selectedSiteId) return
    let cancelled = false
    Promise.all([
      listDispatchRuns(selectedSiteId),
      getOverview(selectedSiteId),
    ])
      .then(([runsData, overview]) => {
        if (cancelled) return
        setRuns(runsData)
        const dateStr =
          overview.latest_measurement_date ?? dayjs().format('YYYY-MM-DD')
        // 默认日期有对应 run 才叠加，否则不叠加
        const runId = runsData.find((r) => r.date === dateStr)?.run_id ?? 'none'
        setDate(dayjs(dateStr))
        setOverlayRunId(runId)
        if (!autoQueriedRef.current) {
          autoQueriedRef.current = true
          void query({ dateStr, runId })
        }
      })
      .catch(() => {
        // 调度记录/总览不可用时仅影响默认值，不阻断曲线查询
        if (cancelled) return
        if (!autoQueriedRef.current) {
          autoQueriedRef.current = true
          void query()
        }
      })
    return () => {
      cancelled = true
    }
    // query 依赖会随用户操作变化，此 effect 只需在站点切换时执行
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSiteId])

  const noData =
    queried &&
    !error &&
    (points.length === 0 ||
      (!schedule &&
        points.every((p) => p.pv_power_kw == null && p.load_power_kw == null)))

  const { points: displayPoints, backfilled } = useMemo(
    () => mergeScheduleInputs(points, schedule),
    [points, schedule],
  )

  const option = useMemo(
    () =>
      buildEnergyChartOption({
        points: displayPoints,
        schedule,
        multiDay: view === '72h',
      }),
    [displayPoints, schedule, view],
  )

  return (
    <>
      <PageHeader
        title="曲线分析（验数）"
        description="电价、光伏、负荷、储能与 SOC 同一时间轴分析"
        extra={
          <Link to="/">
            <Button icon={<ArrowLeftOutlined />}>返回运营</Button>
          </Link>
        }
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card>
          <Form layout="inline" style={{ rowGap: 8 }}>
            <Form.Item label="站点" style={{ marginBottom: 0 }}>
              <Select
                value={selectedSiteId}
                onChange={setSelectedSiteId}
                loading={sitesLoading}
                disabled={sitesLoading || sites.length === 0}
                placeholder="请选择站点"
                style={{ minWidth: 200 }}
                options={sites.map((s) => ({ value: s.site_id, label: s.name }))}
                notFoundContent={
                  <Empty description="暂无站点" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                }
              />
            </Form.Item>
            <Form.Item label="视图" style={{ marginBottom: 0 }}>
              <Radio.Group
                value={view}
                onChange={(e) => setView(e.target.value as ViewMode)}
                optionType="button"
                buttonStyle="solid"
                options={[
                  { label: '24小时', value: '24h' },
                  { label: '72小时', value: '72h' },
                ]}
              />
            </Form.Item>
            <Form.Item label="起始日期" style={{ marginBottom: 0 }}>
              <DatePicker
                value={date}
                onChange={(d) => d && setDate(d)}
                allowClear={false}
              />
            </Form.Item>
            <Form.Item label="叠加调度计划" style={{ marginBottom: 0 }}>
              <Select
                value={overlayRunId}
                onChange={setOverlayRunId}
                style={{ minWidth: 240 }}
                options={[
                  { value: 'none', label: '不叠加' },
                  ...runs.map((r) => ({
                    value: r.run_id,
                    label: `计划 ${r.date} · 生成 ${dayjs(r.created_at).format('MM-DD HH:mm')}`,
                  })),
                ]}
              />
            </Form.Item>
            <Form.Item style={{ marginBottom: 0 }}>
              <Button type="primary" onClick={() => void query()} loading={loading} disabled={!site}>
                查询
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <Card>
          {loading ? (
            <div style={{ textAlign: 'center', padding: 80 }}>
              <Spin />
            </div>
          ) : error ? (
            <Alert type="error" showIcon message="曲线数据加载失败" description={error} />
          ) : noData ? (
            <Empty
              description="该日期暂无实测数据——试试切换日期，或叠加调度计划查看运行方案"
              style={{ padding: 48 }}
            />
          ) : !queried ? (
            <Empty
              description="请选择条件后点击查询"
              style={{ padding: 48 }}
            />
          ) : (
            <>
              <EnergyChart option={option} height={480} />
              {rangeNote && (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {rangeNote}
                  {backfilled && '；光伏/负荷缺失段为调度计划生成时的预测输入'}
                </Typography.Text>
              )}
            </>
          )}
        </Card>
      </Space>
    </>
  )
}
