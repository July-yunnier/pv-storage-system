import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  DatePicker,
  Empty,
  Form,
  InputNumber,
  Segmented,
  Select,
  Slider,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import dayjs, { Dayjs } from 'dayjs'
import { useNavigate } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import SourceTag from '../components/SourceTag'
import { createSchedule, listDispatchRuns, listSites } from '../api/client'
import type { DataSource, DispatchRunSummary, Site } from '../api/types'

type SourceFilter = 'all' | DataSource

const FILTER_EMPTY_TEXT: Record<SourceFilter, string> = {
  all: '暂无调度记录',
  measurements: '暂无实测类调度记录',
  forecast: '暂无预测类调度记录',
}

export default function DispatchPage() {
  const { message } = AntdApp.useApp()
  const navigate = useNavigate()
  const [sites, setSites] = useState<Site[]>([])
  const [sitesLoading, setSitesLoading] = useState(true)
  const [selectedSiteId, setSelectedSiteId] = useState<string>()
  const [date, setDate] = useState<Dayjs>(dayjs())
  const [soc, setSoc] = useState(50)
  const [generating, setGenerating] = useState(false)
  const [generateError, setGenerateError] = useState<string | null>(null)
  const [runs, setRuns] = useState<DispatchRunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [runsError, setRunsError] = useState<string | null>(null)
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((data) => {
        if (cancelled) return
        setSites(data)
        setSelectedSiteId((prev) =>
          prev && data.some((s) => s.site_id === prev) ? prev : data[0]?.site_id,
        )
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

  const loadRuns = useCallback(async (siteId: string) => {
    setRunsLoading(true)
    setRunsError(null)
    try {
      setRuns(await listDispatchRuns(siteId))
    } catch (e) {
      setRuns([])
      setRunsError(e instanceof Error ? e.message : '调度记录加载失败')
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (selectedSiteId) void loadRuns(selectedSiteId)
    else setRuns([])
  }, [selectedSiteId, loadRuns])

  const generate = async () => {
    if (!site) return
    setGenerating(true)
    setGenerateError(null)
    try {
      const data = await createSchedule({
        site_id: site.site_id,
        date: date.format('YYYY-MM-DD'),
        initial_soc: soc / 100,
        strategy: 'rule_v1',
        resolution_minutes: 15,
      })
      if (data.run_id) {
        message.success(
          `调度计划已生成，预期收益 ${data.expected_benefit_yuan.toFixed(2)} 元`,
        )
        navigate(`/dispatch/${data.run_id}?site=${site.site_id}`)
      } else {
        message.warning(
          '计划已生成，但后端未返回运行编号，暂无法跳转详情页',
        )
      }
      void loadRuns(site.site_id)
    } catch (e) {
      setGenerateError(e instanceof Error ? e.message : '调度计划生成失败')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <>
      <PageHeader
        title="调度计划"
        description="生成指定日期的储能充放电计划，查看约束与预期收益"
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card title="生成计划">
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Form layout="inline" style={{ rowGap: 8 }}>
              <Form.Item label="站点" style={{ marginBottom: 0 }}>
                <Select
                  value={selectedSiteId}
                  onChange={setSelectedSiteId}
                  loading={sitesLoading}
                  disabled={sitesLoading || sites.length === 0}
                  placeholder="请选择站点"
                  style={{ minWidth: 220 }}
                  options={sites.map((s) => ({ value: s.site_id, label: s.name }))}
                  notFoundContent={
                    <Empty description="暂无站点" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                  }
                />
              </Form.Item>
              <Form.Item label="日期" style={{ marginBottom: 0 }}>
                <DatePicker
                  value={date}
                  onChange={(d) => d && setDate(d)}
                  allowClear={false}
                />
              </Form.Item>
              <Form.Item label="初始 SOC" style={{ marginBottom: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, width: 300 }}>
                  <Slider
                    min={0}
                    max={100}
                    value={soc}
                    onChange={(v) => setSoc(v)}
                    style={{ flex: 1 }}
                    tooltip={{ formatter: (v) => `${v}%` }}
                  />
                  <InputNumber
                    min={0}
                    max={100}
                    value={soc}
                    onChange={(v) => setSoc(v ?? 0)}
                    style={{ width: 80 }}
                  />
                  <Typography.Text type="secondary">%</Typography.Text>
                </div>
              </Form.Item>
              <Form.Item style={{ marginBottom: 0 }}>
                <Button
                  type="primary"
                  onClick={generate}
                  loading={generating}
                  disabled={!site}
                >
                  生成计划
                </Button>
              </Form.Item>
            </Form>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              自动选择数据来源：当日有完整实测数据用实测，否则用统计预测
            </Typography.Text>
            {generateError && (
              <Alert
                type="error"
                showIcon
                message="调度计划生成失败"
                description={generateError}
              />
            )}
          </Space>
        </Card>

        <Card
          title="历史记录"
          extra={
            <Segmented
              value={sourceFilter}
              onChange={(v) => setSourceFilter(v as SourceFilter)}
              options={[
                { label: '全部', value: 'all' },
                { label: '实测', value: 'measurements' },
                { label: '预测', value: 'forecast' },
              ]}
            />
          }
        >
          {runsError ? (
            <Alert
              type="error"
              showIcon
              message="调度记录加载失败"
              description={runsError}
              action={
                <Button
                  size="small"
                  onClick={() => selectedSiteId && loadRuns(selectedSiteId)}
                >
                  重试
                </Button>
              }
            />
          ) : (
            <Table
              rowKey="run_id"
              size="small"
              loading={runsLoading}
              pagination={{ pageSize: 10, hideOnSinglePage: true }}
              dataSource={runs.filter(
                (r) =>
                  sourceFilter === 'all' ||
                  (r.data_source ?? 'measurements') === sourceFilter,
              )}
              locale={{
                emptyText: <Empty description={FILTER_EMPTY_TEXT[sourceFilter]} />,
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
                  title: '策略',
                  dataIndex: 'strategy',
                  render: (v: string) => <Tag color="blue">{v}</Tag>,
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
                    <Button
                      type="link"
                      size="small"
                      onClick={() =>
                        navigate(`/dispatch/${run.run_id}?site=${run.site_id}`)
                      }
                    >
                      查看
                    </Button>
                  ),
                },
              ]}
            />
          )}
        </Card>
      </Space>
    </>
  )
}
