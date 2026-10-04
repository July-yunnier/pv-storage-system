import { Typography } from 'antd'
import type { ReactNode } from 'react'

interface PageHeaderProps {
  title: string
  description?: string
  extra?: ReactNode
}

export default function PageHeader({ title, description, extra }: PageHeaderProps) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-end',
        flexWrap: 'wrap',
        gap: 12,
        marginBottom: 18,
      }}
    >
      <div>
        <div className="ps-page-title">{title}</div>
        {description && (
          <Typography.Text type="secondary" style={{ fontSize: 13 }}>
            {description}
          </Typography.Text>
        )}
      </div>
      {extra}
    </div>
  )
}
