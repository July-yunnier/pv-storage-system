import { useEffect, useState } from 'react'
import { Alert, Button, Card, Spin } from 'antd'
import { ArrowLeftOutlined } from '@ant-design/icons'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import DispatchResult from '../components/DispatchResult'
import { getDispatchRun } from '../api/client'
import type { DispatchRunDetail } from '../api/types'

export default function DispatchDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [searchParams] = useSearchParams()
  const siteId = searchParams.get('site')
  const navigate = useNavigate()
  const [detail, setDetail] = useState<DispatchRunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!siteId || !runId) {
      setError('缺少站点或运行编号参数，请从调度计划页进入')
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    getDispatchRun(siteId, runId)
      .then((data) => {
        if (!cancelled) setDetail(data)
      })
      .catch((e) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : '调度详情加载失败')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [siteId, runId])

  return (
    <>
      <PageHeader
        title="调度详情"
        description={
          detail ? `计划日期：${detail.date}　策略：${detail.strategy}` : undefined
        }
        extra={
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/dispatch')}>
            返回列表
          </Button>
        }
      />
      {loading ? (
        <Card>
          <div style={{ textAlign: 'center', padding: 48 }}>
            <Spin />
          </div>
        </Card>
      ) : error ? (
        <Alert
          type="error"
          showIcon
          message="调度详情加载失败"
          description={error}
          action={
            <Button size="small" onClick={() => navigate('/dispatch')}>
              返回列表
            </Button>
          }
        />
      ) : detail ? (
        <DispatchResult result={detail} createdAt={detail.created_at ?? null} />
      ) : null}
    </>
  )
}
