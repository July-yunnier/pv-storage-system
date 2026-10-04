import { Tag } from 'antd'
import type { DataSource } from '../api/types'

export default function SourceTag({ source }: { source?: DataSource }) {
  if (!source) return null
  return (
    <Tag color={source === 'forecast' ? 'blue' : 'default'}>
      {source === 'forecast' ? '预测' : '实测'}
    </Tag>
  )
}
