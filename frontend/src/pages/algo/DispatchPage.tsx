import { useCallback, useEffect, useState } from 'react'
import { Card, Col, Row, Space, Statistic, Table, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import { Link } from 'react-router-dom'
import PageHeader from '../../components/PageHeader'
import { SiteSelect, useSites } from '../../components/SitePicker'
import { getReplayRun, listDispatchRuns, listReplayRuns } from '../../api/client'
import type {
  DispatchRunSummary,
  ReplayRunDetail,
  ReplayStrategyMetric,
} from '../../api/types'

const yuan = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })

/** 调度计划（产品向）：线上调度记录 + 最新回放得出的策略收益概览。
 *  完整 策略×预测 矩阵与回放实验入口在「测评中心 · 调度策略」。 */
export default function DispatchPage() {
  const { sites, siteId, setSiteId } = useSites()
  const [runs, setRuns] = useState<DispatchRunSummary[]>([])
  const [replay, setReplay] = useState<ReplayRunDetail | null>(null)

  const load = useCallback(() => {
    if (!siteId) return
    listDispatchRuns(siteId)
      .then(setRuns)
      .catch(() => setRuns([]))
    listReplayRuns(siteId)
      .then(async (rs) => {
        const done = rs.find((r) => r.status === 'finished')
        setReplay(done ? await getReplayRun(done.run_id) : null)
      })
      .catch(() => setReplay(null))
  }, [siteId])

  useEffect(load, [load])

  const strategies = (replay?.summary?.strategies ?? {}) as Record<
    string,
    ReplayStrategyMetric
  >
  const find = (strategy: string, pvMethod?: string) =>
    Object.entries(strategies).find(([key]) => {
      const [m, s] = key.split('+')
      return s === strategy && (pvMethod === undefined || m === pvMethod)
    })?.[1]
  const current = find('d4_rule_predictive', 'blend') ?? find('d4_rule_predictive')
  const fixed = find('d1_fixed_curve')
  const oracle = find('d5_rule_oracle')
  const headroom =
    oracle && current ? oracle.total_saving_yuan - current.total_saving_yuan : null

  return (
    <div>
      <PageHeader
        title="调度计划"
        description="预测曲线 + 电价 + 储能约束 → 96 点充放电计划；每日 23:00 自动生成次日计划，日内随实测自动修正"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        {current && (
          <Card
            title="策略收益概览（最近回放口径）"
            size="small"
            extra={
              replay && (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {replay.config.test_window?.[0]} ~ {replay.config.test_window?.[1]} ·{' '}
                  {replay.config.test_days} 天 · 完整矩阵见「测评中心 · 调度策略」
                </Typography.Text>
              )
            }
          >
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="现网策略 · 30 天节省"
                  value={yuan(current.total_saving_yuan)}
                  suffix="元"
                  valueStyle={{ color: '#e8722a', fontWeight: 700 }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="传统定时策略"
                  value={fixed ? yuan(fixed.total_saving_yuan) : '—'}
                  suffix={fixed ? '元' : ''}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="理论上界（事后最优）"
                  value={oracle ? yuan(oracle.total_saving_yuan) : '—'}
                  suffix={oracle ? '元' : ''}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="优化空间（上界 − 现网）"
                  value={headroom != null ? yuan(headroom) : '—'}
                  suffix={headroom != null ? '元/月' : ''}
                  valueStyle={{ color: '#b37feb', fontWeight: 700 }}
                />
              </Col>
            </Row>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              传统定时跑赢现网预测驱动规则是已知事实——优化空间正是 LP 优化器工作包（D3）的立项依据。
            </Typography.Text>
          </Card>
        )}

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
    </div>
  )
}
