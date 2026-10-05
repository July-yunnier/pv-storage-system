import { useEffect, useMemo, useState } from 'react'
import { Card, Col, Descriptions, Progress, Row, Space, Tag, Timeline, Typography } from 'antd'
import dayjs from 'dayjs'
import type { EChartsOption } from 'echarts'
import PageHeader from '../../components/PageHeader'
import { SiteSelect, useSites } from '../../components/SitePicker'
import EnergyChart from '../../charts/EnergyChart'
import { getLive, getRolling } from '../../api/client'
import type { LiveStatus, RollingState } from '../../api/types'

export default function StoragePage() {
  const { sites, site, siteId, setSiteId } = useSites()
  const [live, setLive] = useState<LiveStatus | null>(null)
  const [rolling, setRolling] = useState<RollingState | null>(null)

  useEffect(() => {
    if (!siteId) return
    getLive(siteId).then(setLive).catch(() => setLive(null))
    getRolling(siteId).then(setRolling).catch(() => setRolling(null))
    const timer = setInterval(() => {
      getLive(siteId).then(setLive).catch(() => undefined)
      getRolling(siteId).then(setRolling).catch(() => undefined)
    }, 30000)
    return () => clearInterval(timer)
  }, [siteId])

  const socOption = useMemo<EChartsOption>(() => {
    const schedule = rolling?.latest_schedule
    if (!schedule?.length) return {}
    return {
      grid: { left: 56, right: 20, top: 30, bottom: 28 },
      xAxis: {
        type: 'category',
        data: schedule.map((p) => dayjs(p.ts).format('HH:mm')),
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        name: 'SOC',
        min: 0,
        max: 1,
        axisLabel: {
          formatter: (v: number | string) => `${Math.round(Number(v) * 100)}%`,
        },
      },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (v) => `${(Number(v) * 100).toFixed(1)}%`,
      },
      series: [
        {
          name: 'SOC',
          type: 'line',
          data: schedule.map((p) => p.soc_end),
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 2.5, color: '#722ed1' },
          areaStyle: { color: 'rgba(114,46,209,0.12)' },
        },
      ],
    }
  }, [rolling])

  const socNow = live?.latest_soc?.value
  const storage = site?.storage

  return (
    <div>
      <PageHeader
        title="储能状态"
        description="储能柜参数、实时 SOC 与日内滚动修正轨迹"
        extra={<SiteSelect sites={sites} value={siteId} onChange={setSiteId} />}
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Row gutter={16}>
          <Col span={8}>
            <Card title="当前 SOC" size="small">
              {socNow != null ? (
                <Progress
                  type="dashboard"
                  percent={Math.round(socNow * 100)}
                  strokeColor="#722ed1"
                  size={120}
                />
              ) : (
                <div
                  style={{
                    height: 120,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography.Text type="secondary">—</Typography.Text>
                </div>
              )}
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {live?.latest_soc
                  ? `更新于 ${dayjs(live.latest_soc.ts).format('HH:mm')}`
                  : '暂无实时数据'}
              </Typography.Text>
            </Card>
          </Col>
          <Col span={16}>
            <Card title="设备参数" size="small">
              <Descriptions
                size="small"
                column={2}
                items={[
                  { key: '1', label: 'PCS 功率', children: `${storage?.pcs_kw ?? '—'} kW` },
                  { key: '2', label: '电池容量', children: `${storage?.capacity_kwh ?? '—'} kWh` },
                  {
                    key: '3',
                    label: 'SOC 工作区间',
                    children: storage
                      ? `${Math.round(storage.soc_min * 100)}% ~ ${Math.round(storage.soc_max * 100)}%`
                      : '—',
                  },
                  {
                    key: '4',
                    label: '往返效率',
                    children: storage ? `${(storage.round_trip_eff * 100).toFixed(0)}%` : '—',
                  },
                ]}
              />
            </Card>
          </Col>
        </Row>

        <Card
          title="今日 SOC 计划轨迹"
          size="small"
          extra={
            rolling && (
              <Space size={12}>
                <Tag color="purple">已修正 {rolling.revisions.length} 次</Tag>
                {rolling.deviation_now != null && (
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    当前光伏偏差 {(rolling.deviation_now * 100).toFixed(1)}%
                  </Typography.Text>
                )}
              </Space>
            )
          }
        >
          {rolling?.latest_schedule?.length ? (
            <EnergyChart option={socOption} height={240} />
          ) : (
            <Typography.Text type="secondary">
              今日暂无调度计划（该站点可能为历史回放站点，不产生线上计划）。
            </Typography.Text>
          )}
        </Card>

        <Card title="滚动修正记录" size="small">
          {rolling?.revisions?.length ? (
            <Timeline
              items={rolling.revisions.map((r) => ({
                color: '#722ed1',
                children: (
                  <>
                    <Typography.Text strong>第 {r.revision} 次修正</Typography.Text>
                    <Typography.Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                      {dayjs(r.created_at).format('HH:mm:ss')} · {r.trigger_reason} ·
                      预期收益 {r.expected_benefit_yuan.toFixed(2)} 元
                    </Typography.Text>
                  </>
                ),
              }))}
            />
          ) : (
            <Typography.Text type="secondary">暂无修正记录。</Typography.Text>
          )}
          {rolling?.cooldown_until && (
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              冷却防抖至 {dayjs(rolling.cooldown_until).format('HH:mm:ss')}（30 分钟内不重复触发）
            </Typography.Text>
          )}
        </Card>
      </Space>
    </div>
  )
}
