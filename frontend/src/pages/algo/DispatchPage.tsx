import { useCallback, useEffect, useState } from 'react'
import {
  App as AntdApp,
  Button,
  Card,
  Popconfirm,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import { ExperimentOutlined } from '@ant-design/icons'
import { Link } from 'react-router-dom'
import PageHeader from '../../components/PageHeader'
import { methodLabel } from '../../components/AccuracyPanel'
import { SiteSelect, useSites } from '../../components/SitePicker'
import {
  getReplayRun,
  listDispatchRuns,
  listReplayRuns,
  runReplay,
} from '../../api/client'
import type {
  DispatchRunSummary,
  ReplayRunDetail,
  ReplayStrategyMetric,
} from '../../api/types'

const STRATEGY_LABEL: Record<string, string> = {
  d0_no_storage: 'D0 无储能基线',
  d1_fixed_curve: 'D1 固定曲线（传统定时）',
  d4_rule_predictive: 'D2 预测驱动规则（现网）',
  d5_rule_oracle: 'D5 事后最优（上界参考）',
}

interface StrategyRow extends ReplayStrategyMetric {
  key: string
  pvMethod: string
  strategy: string
}

export default function DispatchPage() {
  const { message } = AntdApp.useApp()
  const { sites, siteId, setSiteId } = useSites()
  const [replay, setReplay] = useState<ReplayRunDetail | null>(null)
  const [runs, setRuns] = useState<DispatchRunSummary[]>([])
  const [running, setRunning] = useState(false)

  const load = useCallback(() => {
    if (!siteId) return
    listReplayRuns(siteId)
      .then(async (rs) => {
        const done = rs.find((r) => r.status === 'finished')
        setReplay(done ? await getReplayRun(done.run_id) : null)
      })
      .catch(() => setReplay(null))
    listDispatchRuns(siteId)
      .then(setRuns)
      .catch(() => setRuns([]))
  }, [siteId])

  useEffect(load, [load])

  const startReplay = async () => {
    if (!siteId) return
    setRunning(true)
    try {
      const run = await runReplay(siteId, 30)
      message.success(`回放实验已完成：${run.run_id}`)
      load()
    } catch (e) {
      message.error(e instanceof Error ? e.message : '回放失败')
    } finally {
      setRunning(false)
    }
  }

  const rows: StrategyRow[] = Object.entries(
    (replay?.summary?.strategies ?? {}) as Record<string, ReplayStrategyMetric>,
  ).map(([key, m]) => {
    const [pvMethod, strategy] = key.split('+')
    return { key, pvMethod, strategy, ...m }
  })
  const baseline = rows.find((r) => r.strategy === 'd0_no_storage')

  return (
    <div>
      <PageHeader
        title="调度计划"
        description="预测曲线 + 电价 + 储能约束 → 96 点充放电计划；策略好坏用 30 天回放实验说话"
        extra={
          <Space>
            <SiteSelect sites={sites} value={siteId} onChange={setSiteId} />
            <Popconfirm
              title="运行 30 天回放实验？"
              description="对该站点的历史数据重跑全部预测法 × 调度策略，约需 1-2 分钟"
              onConfirm={startReplay}
            >
              <Button icon={<ExperimentOutlined />} loading={running} type="primary">
                运行回放实验
              </Button>
            </Popconfirm>
          </Space>
        }
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card
          title="策略对比（回放实验）"
          size="small"
          extra={
            replay && (
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {replay.config.test_window?.[0]} ~ {replay.config.test_window?.[1]} ·{' '}
                {replay.config.test_days} 天 · SOC 跨日连续
              </Typography.Text>
            )
          }
        >
          <Table
            size="small"
            rowKey="key"
            dataSource={rows}
            loading={running}
            pagination={false}
            columns={[
              {
                title: '策略',
                dataIndex: 'strategy',
                render: (s: string) => (
                  <Tag color={s === 'd5_rule_oracle' ? 'purple' : s === 'd1_fixed_curve' ? 'gold' : 'default'}>
                    {STRATEGY_LABEL[s] ?? s}
                  </Tag>
                ),
              },
              {
                title: '预测输入',
                dataIndex: 'pvMethod',
                render: (m: string) => (m === '__actual__' ? '实测' : methodLabel(m)),
              },
              {
                title: '总电费（元）',
                dataIndex: 'total_cost_yuan',
                render: (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }),
              },
              {
                title: '总节省（元）',
                dataIndex: 'total_saving_yuan',
                render: (v: number, r) =>
                  r.strategy === 'd0_no_storage' ? (
                    '—'
                  ) : (
                    <Typography.Text strong style={{ color: '#34a853' }}>
                      {v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </Typography.Text>
                  ),
                sorter: (a, b) => b.total_saving_yuan - a.total_saving_yuan,
                defaultSortOrder: 'ascend',
              },
              {
                title: '光伏自用率',
                dataIndex: 'pv_self_use_ratio',
                render: (v: number) => `${(v * 100).toFixed(1)}%`,
              },
              {
                title: 'SOC 违规',
                dataIndex: 'soc_violations',
                render: (v: number) =>
                  v === 0 ? <Tag color="green">0</Tag> : <Tag color="red">{v}</Tag>,
              },
            ]}
          />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            基线（D0{baseline ? ` 总电费 ${baseline.total_cost_yuan.toLocaleString(undefined, { maximumFractionDigits: 0 })} 元` : ''}）
            为无储能场景。固定曲线跑赢预测驱动规则是已知事实——它正是 LP 优化器工作包（D3）的立项依据。
          </Typography.Text>
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
              { title: '创建时间', dataIndex: 'created_at' },
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
    </div>
  )
}
