import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Popconfirm,
  Space,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd'
import { ExperimentOutlined } from '@ant-design/icons'
import type { EChartsOption } from 'echarts'
import PageHeader from '../components/PageHeader'
import AccuracyPanel, { methodLabel } from '../components/AccuracyPanel'
import { SiteSelect, useSites } from '../components/SitePicker'
import EnergyChart from '../charts/EnergyChart'
import { getReplayRun, listReplayRuns, runReplay } from '../api/client'
import type {
  ReplayForecastMetric,
  ReplayRunDetail,
  ReplayStrategyMetric,
} from '../api/types'

const STRATEGY_LABEL: Record<string, string> = {
  d0_no_storage: 'D0 无储能基线',
  d1_fixed_curve: 'D1 固定曲线（传统定时）',
  d4_rule_predictive: 'D2 预测驱动规则（现网）',
  d5_rule_oracle: 'D5 事后最优（上界参考）',
}

const STRATEGY_COLOR: Record<string, string> = {
  d0_no_storage: '#9ba19b',
  d1_fixed_curve: '#faad14',
  d4_rule_predictive: '#e8722a',
  d5_rule_oracle: '#b37feb',
}

interface StrategyRow extends ReplayStrategyMetric {
  key: string
  pvMethod: string
  strategy: string
}

function strategyRowLabel(r: StrategyRow): string {
  const input = r.pvMethod === '__actual__' ? '实测' : methodLabel(r.pvMethod)
  return `${STRATEGY_LABEL[r.strategy] ?? r.strategy} · ${input}`
}

/** 各策略 30 天节省横向条形图：一眼看出梯队（D0 → D2×预测输入 → D1 → D5 上界）。 */
function buildStrategyBarOption(rows: StrategyRow[]): EChartsOption {
  const sorted = [...rows].sort((a, b) => a.total_saving_yuan - b.total_saving_yuan)
  return {
    grid: { left: 236, right: 80, top: 16, bottom: 28 },
    xAxis: { type: 'value', name: '30 天节省（元）' },
    yAxis: {
      type: 'category',
      data: sorted.map(strategyRowLabel),
      axisLabel: { fontSize: 11 },
    },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    series: [
      {
        type: 'bar',
        barWidth: 16,
        data: sorted.map((r) => ({
          value: Math.round(r.total_saving_yuan),
          itemStyle: { color: STRATEGY_COLOR[r.strategy] ?? '#9ba19b' },
        })),
        label: {
          show: true,
          position: 'right',
          fontSize: 11,
          formatter: (p) => Number(p.value ?? 0).toLocaleString(),
        },
      },
    ],
  }
}

/** 测评中心：算法精度对账与回放实验矩阵。内部工程界面，不面向客户。 */
export default function EvalCenterPage() {
  const { message } = AntdApp.useApp()
  const { sites, siteId, setSiteId } = useSites()
  const [replay, setReplay] = useState<ReplayRunDetail | null>(null)
  const [running, setRunning] = useState(false)

  const load = useCallback(() => {
    if (!siteId) return
    listReplayRuns(siteId)
      .then(async (runs) => {
        const done = runs.find((r) => r.status === 'finished')
        setReplay(done ? await getReplayRun(done.run_id) : null)
      })
      .catch(() => setReplay(null))
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

  const forecastRows = Object.entries(
    (replay?.summary?.forecasts ?? {}) as Record<string, ReplayForecastMetric>,
  ).map(([method, m]) => ({ method, ...m }))

  const strategyRows: StrategyRow[] = Object.entries(
    (replay?.summary?.strategies ?? {}) as Record<string, ReplayStrategyMetric>,
  ).map(([key, m]) => {
    const [pvMethod, strategy] = key.split('+')
    return { key, pvMethod, strategy, ...m }
  })
  const baseline = strategyRows.find((r) => r.strategy === 'd0_no_storage')

  const replayWindow = replay && (
    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
      {replay.config.test_window?.[0]} ~ {replay.config.test_window?.[1]} ·{' '}
      {replay.config.test_days} 天 · 气象代理：再分析数据
    </Typography.Text>
  )

  return (
    <div>
      <PageHeader
        title="测评中心"
        description="算法精度对账与回放实验矩阵 · 内部工程界面，用于算法验收与对比，不作为客户交付内容"
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
        <Alert
          type="info"
          showIcon
          message="口径说明：skill 以「昨日延续」为基准（1=完美，0=与基准持平）；白天口径仅含日照时段。新算法接入后在同一测试窗口进入对比矩阵。"
        />
        <Card size="small">
          <Tabs
            items={[
              {
                key: 'pv',
                label: '光伏预测',
                children: (
                  <Space direction="vertical" size={16} style={{ width: '100%' }}>
                    <Card
                      title="预测方法对比（回放实验，30 天滚动）"
                      size="small"
                      extra={replayWindow}
                    >
                      {forecastRows.length === 0 ? (
                        <Typography.Text type="secondary">
                          该站点暂无回放实验记录，点击右上角「运行回放实验」生成。
                        </Typography.Text>
                      ) : (
                        <Table
                          size="small"
                          rowKey="method"
                          dataSource={forecastRows}
                          pagination={false}
                          columns={[
                            {
                              title: '方法',
                              dataIndex: 'method',
                              render: (m: string) => <Tag>{methodLabel(m)}</Tag>,
                            },
                            {
                              title: 'skill 全天',
                              dataIndex: 'skill_all_day',
                              render: (v: number | null) => v?.toFixed(4) ?? '—',
                              sorter: (a, b) =>
                                (b.skill_all_day ?? 0) - (a.skill_all_day ?? 0),
                              defaultSortOrder: 'ascend',
                            },
                            {
                              title: 'skill 白天',
                              dataIndex: 'skill_daylight',
                              render: (v: number | null) => v?.toFixed(4) ?? '—',
                            },
                            {
                              title: 'RMSE (kW)',
                              dataIndex: 'rmse',
                              render: (v: number) => v?.toFixed(1),
                            },
                            {
                              title: '偏差 (kW)',
                              dataIndex: 'bias',
                              render: (v: number) => (
                                <Typography.Text
                                  type={v < -20 ? 'warning' : undefined}
                                >
                                  {v?.toFixed(1)}
                                </Typography.Text>
                              ),
                            },
                          ]}
                        />
                      )}
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        物理法偏差为气象代理坐标与光伏数据源站点不一致所致的系统性偏低，属已知近似。
                      </Typography.Text>
                    </Card>
                    {siteId && <AccuracyPanel siteId={siteId} target="pv_power" />}
                  </Space>
                ),
              },
              {
                key: 'load',
                label: '负荷预测',
                children: (
                  <Space direction="vertical" size={16} style={{ width: '100%' }}>
                    {siteId && <AccuracyPanel siteId={siteId} target="load_power" />}
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      学习类负荷模型（MES 排产 + 多任务注意力）已接入主链路：服务端自动构造
                      30 天负荷历史 + 气象温湿度 + 班次日历特征，负荷预测页可切换模型对比。
                      日内滚动启用残差卡尔曼修正。负荷回放矩阵待回放链路接入该模型后开放，
                      对比口径与光伏一致。注意：模型在仿真两班制数据上训练，精度不得作为
                      真实工厂表现对外宣称。
                    </Typography.Text>
                  </Space>
                ),
              },
              {
                key: 'dispatch',
                label: '调度策略',
                children: (
                  <Space direction="vertical" size={16} style={{ width: '100%' }}>
                    <Card title="策略梯队（30 天节省）" size="small" extra={replayWindow}>
                      {strategyRows.length === 0 ? (
                        <Typography.Text type="secondary">
                          该站点暂无回放实验记录，点击右上角「运行回放实验」生成。
                        </Typography.Text>
                      ) : (
                        <EnergyChart
                          option={buildStrategyBarOption(strategyRows)}
                          height={300}
                        />
                      )}
                    </Card>
                    <Card title="策略 × 预测输入 完整矩阵" size="small">
                      <Table
                        size="small"
                        rowKey="key"
                        dataSource={strategyRows}
                        loading={running}
                        pagination={false}
                        columns={[
                          {
                            title: '策略',
                            dataIndex: 'strategy',
                            render: (s: string) => (
                              <Tag
                                color={
                                  s === 'd5_rule_oracle'
                                    ? 'purple'
                                    : s === 'd1_fixed_curve'
                                      ? 'gold'
                                      : 'default'
                                }
                              >
                                {STRATEGY_LABEL[s] ?? s}
                              </Tag>
                            ),
                          },
                          {
                            title: '预测输入',
                            dataIndex: 'pvMethod',
                            render: (m: string) =>
                              m === '__actual__' ? '实测' : methodLabel(m),
                          },
                          {
                            title: '总电费（元）',
                            dataIndex: 'total_cost_yuan',
                            render: (v: number) =>
                              v.toLocaleString(undefined, { maximumFractionDigits: 0 }),
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
                        基线（D0
                        {baseline
                          ? ` 总电费 ${baseline.total_cost_yuan.toLocaleString(undefined, { maximumFractionDigits: 0 })} 元`
                          : ''}
                        ）为无储能场景。固定曲线跑赢预测驱动规则是已知事实——它正是 LP 优化器工作包（D3）的立项依据。
                      </Typography.Text>
                    </Card>
                  </Space>
                ),
              },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}
