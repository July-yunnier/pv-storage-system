import { useEffect, useRef } from 'react'
import { Alert, Card, Col, Row, Space, Statistic, Table, Tag, Typography } from 'antd'
import dayjs from 'dayjs'
import * as echarts from 'echarts'
import { TIER_META } from './tier'
import SourceTag from './SourceTag'
import type { ScheduleAction, SchedulePoint, ScheduleResponse, TierName } from '../api/types'

const ACTION_META: Record<ScheduleAction, { label: string; color: string }> = {
  charge: { label: '充电', color: '#1677ff' },
  discharge: { label: '放电', color: '#52c41a' },
  idle: { label: '静置', color: 'default' },
}

function ScheduleChart({ schedule }: { schedule: SchedulePoint[] }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
    chart.setOption({
      grid: { left: 56, right: 56, top: 32, bottom: 28 },
      legend: { top: 0 },
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const items = params as {
            axisValue: string
            seriesName: string
            value: number
          }[]
          const lines = items
            .filter((p) => p.value !== 0)
            .map(
              (p) =>
                `${p.seriesName}：${p.value}${p.seriesName === 'SOC' ? '%' : ' kW'}`,
            )
          return [items[0]?.axisValue ?? '', ...lines].join('<br/>')
        },
      },
      xAxis: {
        type: 'category',
        data: schedule.map((p) => dayjs(p.ts).format('HH:mm')),
        axisLabel: { interval: 7 },
      },
      yAxis: [
        { type: 'value', name: '功率 (kW)' },
        {
          type: 'value',
          name: 'SOC',
          min: 0,
          max: 100,
          axisLabel: { formatter: '{value}%' },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: '充电功率',
          type: 'bar',
          stack: 'power',
          data: schedule.map((p) => (p.action === 'charge' ? p.power_kw : 0)),
          itemStyle: { color: '#1677ff' },
        },
        {
          name: '放电功率',
          type: 'bar',
          stack: 'power',
          data: schedule.map((p) => (p.action === 'discharge' ? -p.power_kw : 0)),
          itemStyle: { color: '#52c41a' },
        },
        {
          name: 'SOC',
          type: 'line',
          yAxisIndex: 1,
          symbol: 'none',
          data: schedule.map((p) => Math.round(p.soc_end * 1000) / 10),
          lineStyle: { type: 'dashed', width: 1.5, color: '#722ed1' },
        },
      ],
    })
    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.dispose()
    }
  }, [schedule])

  return <div ref={ref} style={{ height: 260 }} />
}

function ScheduleTable({ schedule }: { schedule: SchedulePoint[] }) {
  return (
    <Table
      rowKey="ts"
      size="small"
      pagination={false}
      scroll={{ y: 360 }}
      dataSource={schedule}
      columns={[
        {
          title: '时间',
          dataIndex: 'ts',
          width: 70,
          render: (ts: string) => dayjs(ts).format('HH:mm'),
        },
        {
          title: '动作',
          dataIndex: 'action',
          width: 80,
          render: (action: ScheduleAction) => {
            const meta = ACTION_META[action]
            return <Tag color={meta.color}>{meta.label}</Tag>
          },
        },
        {
          title: '功率 (kW)',
          dataIndex: 'power_kw',
          width: 100,
          render: (v: number) => v.toFixed(1),
        },
        {
          title: 'SOC',
          key: 'soc',
          width: 130,
          render: (_: unknown, p: SchedulePoint) =>
            `${Math.round(p.soc_start * 100)}% → ${Math.round(p.soc_end * 100)}%`,
        },
        {
          title: '电价 (元/kWh)',
          dataIndex: 'price',
          width: 110,
          render: (v: number) => v.toFixed(2),
        },
        {
          title: '档位',
          dataIndex: 'tier',
          width: 80,
          render: (tier: TierName) => {
            const meta = TIER_META[tier] ?? { label: tier, color: '#8c8c8c' }
            return <Tag color={meta.color}>{meta.label}</Tag>
          },
        },
        {
          title: '购电 (kW)',
          dataIndex: 'grid_import_kw',
          width: 100,
          render: (v: number) => v.toFixed(1),
        },
      ]}
    />
  )
}

export default function DispatchResult({
  result,
  createdAt,
}: {
  result: ScheduleResponse
  createdAt?: string | null
}) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {result.data_source === 'forecast' && (
        <Alert
          type="info"
          showIcon
          message="本计划基于预测数据生成，实际收益以实测为准"
        />
      )}
      <Row gutter={16}>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="预期收益（元）"
              value={result.expected_benefit_yuan}
              precision={2}
              valueStyle={{ color: '#3f8600', fontWeight: 600 }}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="基准成本（元）"
              value={result.baseline_cost_yuan}
              precision={2}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="优化后成本（元）"
              value={result.optimized_cost_yuan}
              precision={2}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card>
            <Statistic
              title="等效全循环次数"
              value={result.equivalent_full_cycles}
              precision={2}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Space direction="vertical" size={8} style={{ width: '100%' }}>
          <Space wrap size={8}>
            <Typography.Text strong>策略</Typography.Text>
            <Tag color="blue">{result.strategy}</Tag>
            {result.data_source && (
              <>
                <Typography.Text strong style={{ marginLeft: 16 }}>
                  数据来源
                </Typography.Text>
                <SourceTag source={result.data_source} />
              </>
            )}
            <Typography.Text strong style={{ marginLeft: 16 }}>
              生效约束
            </Typography.Text>
            {result.constraints_applied.map((c) => (
              <Tag key={c}>{c}</Tag>
            ))}
          </Space>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {[
              result.run_id ? `运行编号：${result.run_id}` : null,
              createdAt
                ? `生成时间：${dayjs(createdAt).format('YYYY-MM-DD HH:mm:ss')}`
                : null,
              `计划日期：${result.date}`,
            ]
              .filter(Boolean)
              .join('　')}
          </Typography.Text>
        </Space>
      </Card>

      <Card title="充放电与 SOC 曲线">
        <ScheduleChart schedule={result.schedule} />
      </Card>

      <Card title="调度明细">
        <ScheduleTable schedule={result.schedule} />
      </Card>
    </Space>
  )
}
