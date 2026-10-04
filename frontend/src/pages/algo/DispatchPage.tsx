import { useCallback, useEffect, useState } from 'react'
import { Card, Empty, Space, Spin, Table, Tabs, Tag } from 'antd'
import dayjs from 'dayjs'
import { Link } from 'react-router-dom'
import PageHeader from '../../components/PageHeader'
import DispatchPlanView from '../../components/DispatchPlanView'
import { SiteSelect, useSites } from '../../components/SitePicker'
import { getCurves, getDispatchRun, listDispatchRuns } from '../../api/client'
import type {
  CurvePoint,
  DispatchRunDetail,
  DispatchRunSummary,
} from '../../api/types'

/** 调度计划（产品主交付界面）：
 *  三条预测曲线 → 96 点充放电计划（充电分光伏/电网来源）→ 分时段人话说明。
 *  今日 = 实测叠加 + 今日计划；明日 = 日前计划。 */
export default function DispatchPage() {
  const { sites, siteId, setSiteId } = useSites()
  const [runs, setRuns] = useState<DispatchRunSummary[]>([])
  const [todayPlan, setTodayPlan] = useState<DispatchRunDetail | null>(null)
  const [tomorrowPlan, setTomorrowPlan] = useState<DispatchRunDetail | null>(null)
  const [todayMeasured, setTodayMeasured] = useState<CurvePoint[]>([])
  const [loading, setLoading] = useState(false)

  const load = useCallback(() => {
    if (!siteId) return
    setLoading(true)
    const today = dayjs().format('YYYY-MM-DD')
    const tomorrow = dayjs().add(1, 'day').format('YYYY-MM-DD')
    listDispatchRuns(siteId)
      .then(async (rs) => {
        setRuns(rs)
        const todayRun = rs.find((r) => r.date === today)
        const tomorrowRun =
          rs.find((r) => r.date === tomorrow && r.data_source !== 'measurements') ??
          rs.find((r) => r.date === tomorrow) ??
          rs.find((r) => r.data_source === 'forecast')
        const [t, tm] = await Promise.all([
          todayRun ? getDispatchRun(siteId, todayRun.run_id) : Promise.resolve(null),
          tomorrowRun ? getDispatchRun(siteId, tomorrowRun.run_id) : Promise.resolve(null),
        ])
        setTodayPlan(t)
        setTomorrowPlan(tm)
      })
      .catch(() => {
        setRuns([])
        setTodayPlan(null)
        setTomorrowPlan(null)
      })
      .finally(() => setLoading(false))
    getCurves(siteId, today, today)
      .then((c) => setTodayMeasured(c.points))
      .catch(() => setTodayMeasured([]))
  }, [siteId])

  useEffect(load, [load])

  return (
    <div>
      <PageHeader
        title="调度计划"
        description="光伏预测 + 负荷预测 + 分时电价 → 96 点充放电计划；每日 23:00 自动生成次日计划，日内随实测自动修正"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Spin spinning={loading}>
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Card size="small">
            <Tabs
              defaultActiveKey="tomorrow"
              items={[
                {
                  key: 'today',
                  label: `今日计划（${dayjs().format('MM-DD')}）`,
                  children: todayPlan ? (
                    <DispatchPlanView plan={todayPlan} measured={todayMeasured} />
                  ) : (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description="今日计划未生成（系统每日 23:00 自动生成次日计划，可在总览页手动生成）"
                    />
                  ),
                },
                {
                  key: 'tomorrow',
                  label: `明日计划（${dayjs().add(1, 'day').format('MM-DD')}）`,
                  children: tomorrowPlan ? (
                    <DispatchPlanView plan={tomorrowPlan} />
                  ) : (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description="明日计划未生成（系统每日 23:00 自动生成，可在总览页手动生成）"
                    />
                  ),
                },
              ]}
            />
          </Card>

          <Card title="线上调度记录" size="small">
            <Table
              size="small"
              rowKey="run_id"
              dataSource={runs}
              pagination={{ pageSize: 8, size: 'small' }}
              columns={[
                { title: '日期', dataIndex: 'date', width: 110 },
                {
                  title: '数据源',
                  dataIndex: 'data_source',
                  render: (s: string) => (
                    <Tag>{s === 'forecast' ? '预测' : '实测'}</Tag>
                  ),
                },
                {
                  title: '预期收益（元）',
                  dataIndex: 'expected_benefit_yuan',
                  render: (v: number) => v?.toFixed(2) ?? '—',
                },
                {
                  title: '等效满循环',
                  dataIndex: 'equivalent_full_cycles',
                  render: (v: number) => v?.toFixed(2) ?? '—',
                },
                {
                  title: '创建时间',
                  dataIndex: 'created_at',
                  render: (v: string) => dayjs(v).format('YYYY-MM-DD HH:mm'),
                },
                {
                  title: '操作',
                  render: (_, r) => (
                    <Link to={`/dispatch/${r.run_id}?site=${siteId}`}>查看详情</Link>
                  ),
                },
              ]}
            />
          </Card>
        </Space>
      </Spin>
    </div>
  )
}
