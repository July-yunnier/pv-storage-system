import { useEffect, useState } from 'react'
import { Alert, Card, Space, Table, Tabs, Tag, Typography } from 'antd'
import PageHeader from '../components/PageHeader'
import AccuracyPanel, { methodLabel } from '../components/AccuracyPanel'
import { SiteSelect, useSites } from '../components/SitePicker'
import { getReplayRun, listReplayRuns } from '../api/client'
import type { ReplayForecastMetric, ReplayRunDetail } from '../api/types'

/** 测评中心：算法精度对账与回放实验矩阵。内部工程界面，不面向客户。 */
export default function EvalCenterPage() {
  const { sites, siteId, setSiteId } = useSites()
  const [replay, setReplay] = useState<ReplayRunDetail | null>(null)

  useEffect(() => {
    if (!siteId) return
    listReplayRuns(siteId)
      .then(async (runs) => {
        const done = runs.find((r) => r.status === 'finished')
        setReplay(done ? await getReplayRun(done.run_id) : null)
      })
      .catch(() => setReplay(null))
  }, [siteId])

  const forecastRows = Object.entries(
    (replay?.summary?.forecasts ?? {}) as Record<string, ReplayForecastMetric>,
  ).map(([method, m]) => ({ method, ...m }))

  return (
    <div>
      <PageHeader
        title="测评中心"
        description="算法精度对账与回放实验矩阵 · 内部工程界面，用于算法验收与对比，不作为客户交付内容"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
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
                      extra={
                        replay && (
                          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                            {replay.config.test_window?.[0]} ~{' '}
                            {replay.config.test_window?.[1]} · 气象代理：再分析数据
                          </Typography.Text>
                        )
                      }
                    >
                      {forecastRows.length === 0 ? (
                        <Typography.Text type="secondary">
                          该站点暂无回放实验记录，可在「调度计划」页发起。
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
                      负荷回放矩阵待 LSTM 等学习类算法接入后开放，对比口径与光伏一致。
                    </Typography.Text>
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
