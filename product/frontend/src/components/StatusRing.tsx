import { Progress, Space, Tag, Typography } from 'antd'
import type { QualityStatus } from '../api/types'

const STATUS_COLOR: Record<string, string> = {
  complete: '#52c41a',
  partial: '#faad14',
  empty: '#f5222d',
}

const STATUS_LABEL: Record<string, string> = {
  complete: '完整',
  partial: '部分缺失',
  empty: '无数据',
}

function toPercent(v: number): number {
  return Math.round((v <= 1 ? v * 100 : v) * 10) / 10
}

export default function StatusRing({
  label,
  completeness,
  status,
}: {
  label: string
  completeness: number
  status: QualityStatus
}) {
  return (
    <Space direction="vertical" align="center" size={4}>
      <Progress
        type="circle"
        percent={toPercent(completeness)}
        size={88}
        strokeColor={STATUS_COLOR[status] ?? '#faad14'}
      />
      <Space size={4}>
        <Typography.Text type="secondary">{label}</Typography.Text>
        <Tag color={STATUS_COLOR[status]}>{STATUS_LABEL[status] ?? status}</Tag>
      </Space>
    </Space>
  )
}
