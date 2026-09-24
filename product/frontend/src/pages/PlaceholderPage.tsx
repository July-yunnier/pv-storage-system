import { Card, Empty } from 'antd'
import type { ReactNode } from 'react'
import PageHeader from '../components/PageHeader'

interface PlaceholderPageProps {
  title: string
  description: string
  extra?: ReactNode
}

export default function PlaceholderPage({ title, description, extra }: PlaceholderPageProps) {
  return (
    <>
      <PageHeader title={title} description={description} />
      <Card>
        {extra}
        <Empty description="功能建设中，敬请期待后续迭代" />
      </Card>
    </>
  )
}
