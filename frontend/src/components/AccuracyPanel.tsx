import { useEffect, useState } from 'react'
import { Card, Col, Empty, Row, Statistic, Table, Tag, Typography } from 'antd'
import { getAccuracy } from '../api/client'
import type { AccuracyResponse, ForecastTarget } from '../api/types'

const METHOD_LABEL: Record<string, string> = {
  pv_diurnal_stats: '同刻均值',
  pvlib_pvwatts: '物理法',
  pvlib_blend: '融合',
  pv_intraday_deviation: '日内修正',
  load_diurnal_mean: '同刻均值',
  load_intraday_kalman: '日内卡尔曼',
  load_mes_multitask_attention: 'MES 注意力',
  day_persistence: '昨日延续',
  slot_mean: '同刻均值',
  slot_mean_bias: '同刻均值+偏差修正',
  physics: '物理法',
  blend: '融合',
  oracle: '事后上帝视角',
}

export function methodLabel(name: string): string {
  return METHOD_LABEL[name] ?? name
}

function fmt(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

/**
 * 精度对账区：上半为按方法汇总，下半为逐日明细。
 * 数据源为 forecast_archive 封存 + forecast_accuracy 对账（观测体系）。
 */
export default function AccuracyPanel({
  siteId,
  target,
  days = 30,
}: {
  siteId: string
  target: ForecastTarget
  days?: number
}) {
  const [data, setData] = useState<AccuracyResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    getAccuracy(siteId, target, days)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [siteId, target, days])

  const summaries = Object.values(data?.summary ?? {})
  const best = summaries.reduce<typeof summaries[number] | null>(
    (acc, s) => (acc === null || (s.skill_all_day ?? -1) > (acc.skill_all_day ?? -1) ? s : acc),
    null,
  )

  return (
    <Card
      title="精度对账（封存预测 vs 实测）"
      size="small"
      loading={loading}
      extra={
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          每次预测自动封存，实测到齐后自动对账 · 近 {days} 天
        </Typography.Text>
      }
    >
      {summaries.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="暂无对账记录：站点产生过预测且实测满 96 点/天后自动生成"
        />
      ) : (
        <>
          <Row gutter={16} style={{ marginBottom: 12 }}>
            <Col span={8}>
              <Statistic
                title="最优方法"
                value={best ? methodLabel(best.method) : '—'}
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="最优 skill（全天口径）"
                value={fmt(best?.skill_all_day)}
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="对账天数"
                value={best?.days ?? 0}
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
          </Row>
          <Table
            size="small"
            rowKey={(r) => `${r.method}-${r.target_date}`}
            dataSource={data?.rows ?? []}
            pagination={{ pageSize: 8, size: 'small' }}
            columns={[
              { title: '日期', dataIndex: 'target_date', width: 110 },
              {
                title: '方法',
                dataIndex: 'method',
                render: (m: string) => <Tag>{methodLabel(m)}</Tag>,
              },
              {
                title: 'skill 全天',
                dataIndex: 'skill_all_day',
                render: (v: number | null) => fmt(v),
              },
              {
                title: 'skill 白天',
                dataIndex: 'skill_daylight',
                render: (v: number | null) => fmt(v),
              },
              { title: 'RMSE', dataIndex: 'rmse', render: (v: number | null) => fmt(v, 1) },
              { title: '偏差', dataIndex: 'bias', render: (v: number | null) => fmt(v, 1) },
              { title: '样本点', dataIndex: 'n_points', width: 80 },
            ]}
          />
        </>
      )}
    </Card>
  )
}
