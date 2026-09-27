import { useEffect, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd'
import { ArrowLeftOutlined, DownloadOutlined, FilePdfOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import { createReport, getReport, getSizingRun, listSites } from '../api/client'
import type {
  Site,
  SizingCandidate,
  SizingResult,
  SizingSensitivityPoint,
} from '../api/types'
import { fmtIrr, fmtWanYuan, fmtYears, fmtYuan } from '../utils/format'

const ASSUMPTION_LABELS: Record<string, string> = {
  battery_price_per_kwh: '电池单价（元/kWh）',
  pcs_price_per_kw: 'PCS 单价（元/kW）',
  install_ratio: '安装及其他费用比例',
  opex_ratio: '年运维比例',
  project_years: '项目年限（年）',
  battery_life_years: '电池寿命（年）',
  replacement_cost_ratio: '更换成本比例',
  discount_rate: '折现率',
  residual_ratio: '残值比例',
  representative_days: '代表日数量',
  annualization_method: '年化方法',
}

type ReportPhase =
  | { phase: 'idle' }
  | { phase: 'working' }
  | { phase: 'done'; url: string }
  | { phase: 'failed'; error: string }

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

function deltaLabel(delta: number): string {
  if (delta === 0) return '基准'
  const pct = Math.round(delta * 100)
  return `${pct > 0 ? '+' : ''}${pct}%`
}

function SensitivityTable({
  title,
  data,
}: {
  title: string
  data: SizingSensitivityPoint[]
}) {
  return (
    <Table
      title={() => <Typography.Text strong>{title}</Typography.Text>}
      rowKey={(r) => String(r.delta)}
      size="small"
      pagination={false}
      dataSource={data}
      columns={[
        {
          title: '情景',
          dataIndex: 'delta',
          render: (v: number) => (
            <Tag color={v === 0 ? 'blue' : 'default'}>{deltaLabel(v)}</Tag>
          ),
        },
        {
          title: '年收益（元）',
          dataIndex: 'annual_benefit_yuan',
          align: 'right',
          render: (v: number) => fmtYuan(v),
        },
        {
          title: '回本（年）',
          dataIndex: 'payback_years',
          align: 'right',
          render: (v: number | null) => fmtYears(v),
        },
        {
          title: 'NPV（元）',
          dataIndex: 'npv_yuan',
          align: 'right',
          render: (v: number) => (
            <span style={{ color: v >= 0 ? '#3f8600' : '#cf1322' }}>{fmtYuan(v)}</span>
          ),
        },
      ]}
    />
  )
}

export default function SizingDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [searchParams] = useSearchParams()
  const siteId = searchParams.get('site')
  const navigate = useNavigate()
  const [result, setResult] = useState<SizingResult | null>(null)
  const [site, setSite] = useState<Site | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [report, setReport] = useState<ReportPhase>({ phase: 'idle' })
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    if (!siteId || !runId) {
      setError('缺少站点或分析编号参数，请从方案分析页进入')
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    getSizingRun(siteId, runId)
      .then((data) => {
        if (!cancelled) setResult(data)
      })
      .catch((e) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : '分析结果加载失败')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    listSites()
      .then((data) => {
        if (!cancelled) setSite(data.find((s) => s.site_id === siteId) ?? null)
      })
      .catch(() => {
        if (!cancelled) setSite(null)
      })
    return () => {
      cancelled = true
    }
  }, [siteId, runId])

  const exportReport = async () => {
    if (!siteId || !runId) return
    setReport({ phase: 'working' })
    try {
      const job = await createReport(siteId, runId)
      for (let i = 0; i < 15; i++) {
        await sleep(2000)
        if (!mountedRef.current) return
        const status = await getReport(job.report_id)
        if (status.status === 'succeeded') {
          if (!mountedRef.current) return
          setReport({
            phase: 'done',
            url: status.download_url ?? `/api/v1/reports/${job.report_id}/download`,
          })
          return
        }
        if (status.status === 'failed') {
          if (!mountedRef.current) return
          setReport({ phase: 'failed', error: status.error ?? '报告生成失败' })
          return
        }
      }
      if (mountedRef.current)
        setReport({ phase: 'failed', error: '报告生成超时，请稍后重试' })
    } catch (e) {
      if (mountedRef.current)
        setReport({
          phase: 'failed',
          error: e instanceof Error ? e.message : '报告生成失败',
        })
    }
  }

  const rec = result?.recommended ?? null
  const currentCandidate =
    result && site
      ? (result.candidates.find(
          (c) =>
            c.pcs_kw === site.storage.pcs_kw &&
            c.capacity_kwh === site.storage.capacity_kwh,
        ) ?? null)
      : null

  const exportButton =
    report.phase === 'done' ? (
      <Button
        type="primary"
        icon={<DownloadOutlined />}
        onClick={() => window.open(report.url, '_blank')}
      >
        下载 PDF
      </Button>
    ) : (
      <Button
        icon={<FilePdfOutlined />}
        loading={report.phase === 'working'}
        disabled={!result}
        onClick={exportReport}
      >
        {report.phase === 'working'
          ? '报告生成中…'
          : report.phase === 'failed'
            ? '重试导出'
            : '导出 PDF 报告'}
      </Button>
    )

  return (
    <>
      <PageHeader
        title="投资分析结果"
        description={
          result
            ? `站点：${site?.name ?? result.site_id}　生成时间：${dayjs(result.created_at).format('YYYY-MM-DD HH:mm:ss')}　代表日：${result.representative_dates.length} 天`
            : undefined
        }
        extra={
          <Space size={8}>
            {exportButton}
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/sizing')}>
              返回列表
            </Button>
          </Space>
        }
      />
      {report.phase === 'failed' && (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          message="报告导出失败"
          description={report.error}
        />
      )}
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
          message="分析结果加载失败"
          description={error}
          action={
            <Button size="small" onClick={() => navigate('/sizing')}>
              返回列表
            </Button>
          }
        />
      ) : result ? (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          {rec && (
            <Card>
              <Space direction="vertical" size={8} style={{ width: '100%' }}>
                <Typography.Title level={4} style={{ margin: 0 }}>
                  结论：推荐 {rec.pcs_kw} kW / {rec.capacity_kwh} kWh
                  <Typography.Text type="secondary" style={{ fontSize: 14, marginLeft: 12 }}>
                    （{rec.duration_h} h 系统）
                  </Typography.Text>
                </Typography.Title>
                <Typography.Text style={{ fontSize: 15 }}>
                  总投资约{' '}
                  <Typography.Text strong>{fmtWanYuan(rec.capex_yuan)}</Typography.Text>{' '}
                  万元，每年净赚{' '}
                  <Typography.Text strong style={{ color: '#3f8600' }}>
                    {fmtYuan(rec.annual_benefit_yuan - rec.annual_opex_yuan)}
                  </Typography.Text>{' '}
                  元，<Typography.Text strong>{fmtYears(rec.payback_years)}</Typography.Text>{' '}
                  年回本，项目期净现值 {fmtWanYuan(rec.npv_yuan)} 万元
                </Typography.Text>
                {currentCandidate &&
                  (currentCandidate.pcs_kw !== rec.pcs_kw ||
                    currentCandidate.capacity_kwh !== rec.capacity_kwh) && (
                    <Typography.Text type="secondary">
                      对比当前安装的 {currentCandidate.pcs_kw} kW /{' '}
                      {currentCandidate.capacity_kwh} kWh（年收益{' '}
                      {fmtYuan(currentCandidate.annual_benefit_yuan)} 元），推荐配置每年多赚{' '}
                      <Typography.Text type="success" strong>
                        {fmtYuan(rec.annual_benefit_yuan - currentCandidate.annual_benefit_yuan)}
                      </Typography.Text>{' '}
                      元
                    </Typography.Text>
                  )}
              </Space>
            </Card>
          )}

          <Card title="推荐方案">
            {!rec ? (
              <Alert
                type="warning"
                showIcon
                message="所有候选方案 NPV 均为负，不建议投资"
              />
            ) : (
              <Space direction="vertical" size={16} style={{ width: '100%' }}>
                <Row gutter={16}>
                  <Col xs={12} md={8}>
                    <Statistic
                      title="总投资（万元）"
                      value={fmtWanYuan(rec.capex_yuan)}
                      valueStyle={{ fontSize: 24, fontWeight: 600 }}
                    />
                  </Col>
                  <Col xs={12} md={8}>
                    <Statistic
                      title="年收益（元/年）"
                      value={fmtYuan(rec.annual_benefit_yuan)}
                      valueStyle={{ color: '#3f8600', fontWeight: 600, fontSize: 24 }}
                    />
                  </Col>
                  <Col xs={12} md={8}>
                    <Statistic
                      title="回本年限（年）"
                      value={fmtYears(rec.payback_years)}
                      valueStyle={{ fontSize: 24, fontWeight: 600 }}
                    />
                  </Col>
                </Row>
                <Descriptions column={{ xs: 2, md: 4 }} size="small">
                  <Descriptions.Item label="年运维（元/年）">
                    {fmtYuan(rec.annual_opex_yuan)}
                  </Descriptions.Item>
                  <Descriptions.Item label="IRR（内部收益率）">
                    {fmtIrr(rec.irr)}
                  </Descriptions.Item>
                  <Descriptions.Item label="NPV（净现值）">
                    <span style={{ color: rec.npv_yuan >= 0 ? '#3f8600' : '#cf1322' }}>
                      {fmtYuan(rec.npv_yuan)}
                    </span>
                  </Descriptions.Item>
                  <Descriptions.Item label="日循环次数">
                    {rec.equivalent_full_cycles_per_day.toFixed(2)}
                  </Descriptions.Item>
                </Descriptions>
              </Space>
            )}
          </Card>

          <Card title="方案对比">
            <Table
              rowKey={(r: SizingCandidate) => `${r.pcs_kw}-${r.duration_h}`}
              size="small"
              pagination={false}
              dataSource={result.candidates}
              rowClassName={(r: SizingCandidate) =>
                rec && r.pcs_kw === rec.pcs_kw && r.duration_h === rec.duration_h
                  ? 'sizing-recommended-row'
                  : ''
              }
              columns={[
                { title: '功率（kW）', dataIndex: 'pcs_kw' },
                { title: '容量（kWh）', dataIndex: 'capacity_kwh' },
                { title: '时长（h）', dataIndex: 'duration_h' },
                {
                  title: '投资（万元）',
                  dataIndex: 'capex_yuan',
                  align: 'right',
                  render: (v: number) => fmtWanYuan(v),
                },
                {
                  title: '年收益（元）',
                  dataIndex: 'annual_benefit_yuan',
                  align: 'right',
                  render: (v: number) => fmtYuan(v),
                },
                {
                  title: '回本（年）',
                  dataIndex: 'payback_years',
                  align: 'right',
                  render: (v: number | null) => fmtYears(v),
                },
                {
                  title: 'IRR',
                  dataIndex: 'irr',
                  align: 'right',
                  render: (v: number | null) => fmtIrr(v),
                },
                {
                  title: 'NPV（元）',
                  dataIndex: 'npv_yuan',
                  align: 'right',
                  defaultSortOrder: 'descend',
                  sorter: (a: SizingCandidate, b: SizingCandidate) =>
                    a.npv_yuan - b.npv_yuan,
                  render: (v: number) => (
                    <span style={{ color: v >= 0 ? '#3f8600' : '#cf1322' }}>
                      {fmtYuan(v)}
                    </span>
                  ),
                },
                {
                  title: '日循环次数',
                  dataIndex: 'equivalent_full_cycles_per_day',
                  align: 'right',
                  render: (v: number) => v.toFixed(2),
                },
              ]}
            />
          </Card>

          <Row gutter={16}>
            <Col xs={24} lg={12}>
              <Card>
                <SensitivityTable title="电价价差敏感性" data={result.sensitivity.price_spread} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card>
                <SensitivityTable title="电池价格敏感性" data={result.sensitivity.battery_price} />
              </Card>
            </Col>
          </Row>

          <Card title="计算假设">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Descriptions column={{ xs: 1, sm: 2, md: 3 }} bordered size="small">
                {Object.entries(result.assumptions).map(([key, value]) => (
                  <Descriptions.Item
                    key={key}
                    label={ASSUMPTION_LABELS[key] ?? key}
                  >
                    {typeof value === 'number'
                      ? value.toLocaleString('zh-CN')
                      : String(value)}
                  </Descriptions.Item>
                ))}
              </Descriptions>
              <Typography.Text>
                代表日：
                {result.representative_dates.length > 0
                  ? result.representative_dates.join('、')
                  : '—'}
              </Typography.Text>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                说明：结果基于代表日调度测算外推至全年，未考虑电池衰减逐年的细微影响与电价政策变化，
                实际收益受天气、负荷波动影响，仅供投资决策参考。
              </Typography.Text>
            </Space>
          </Card>
        </Space>
      ) : null}
    </>
  )
}
