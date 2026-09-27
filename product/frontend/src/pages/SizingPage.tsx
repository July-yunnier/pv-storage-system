import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Col,
  Collapse,
  Empty,
  Form,
  InputNumber,
  Row,
  Select,
  Space,
  Table,
  Typography,
} from 'antd'
import dayjs from 'dayjs'
import { useNavigate } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import { createSizing, listSites, listSizingRuns } from '../api/client'
import type { Site, SizingFinancial, SizingRunSummary } from '../api/types'
import { fmtYuan, fmtYears } from '../utils/format'

const PCS_PRESETS = [100, 200, 300, 500]
const DURATION_PRESETS = [1, 2, 4]

const DEFAULT_FINANCIAL: SizingFinancial = {
  battery_price_per_kwh: 1200,
  pcs_price_per_kw: 300,
  install_ratio: 0.1,
  opex_ratio: 0.02,
  project_years: 15,
  battery_life_years: 10,
  replacement_cost_ratio: 0.6,
  discount_rate: 0.08,
  residual_ratio: 0.05,
}

const FINANCIAL_FIELDS: { key: keyof SizingFinancial; label: string; step?: number }[] = [
  { key: 'battery_price_per_kwh', label: '电池单价（元/kWh）', step: 50 },
  { key: 'pcs_price_per_kw', label: 'PCS 单价（元/kW）', step: 10 },
  { key: 'install_ratio', label: '安装及其他费用比例', step: 0.01 },
  { key: 'opex_ratio', label: '年运维比例', step: 0.005 },
  { key: 'project_years', label: '项目年限（年）', step: 1 },
  { key: 'battery_life_years', label: '电池寿命（年）', step: 1 },
  { key: 'replacement_cost_ratio', label: '更换成本比例', step: 0.05 },
  { key: 'discount_rate', label: '折现率', step: 0.01 },
  { key: 'residual_ratio', label: '残值比例', step: 0.01 },
]

function toNumbers(values: string[]): number[] {
  return values
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v) && v > 0)
}

export default function SizingPage() {
  const { message } = AntdApp.useApp()
  const navigate = useNavigate()
  const [sites, setSites] = useState<Site[]>([])
  const [sitesLoading, setSitesLoading] = useState(true)
  const [selectedSiteId, setSelectedSiteId] = useState<string>()
  const [pcsValues, setPcsValues] = useState<string[]>(PCS_PRESETS.map(String))
  const [durationValues, setDurationValues] = useState<string[]>(
    DURATION_PRESETS.map(String),
  )
  const [repDays, setRepDays] = useState(3)
  const [financial, setFinancial] = useState<SizingFinancial>(DEFAULT_FINANCIAL)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [runs, setRuns] = useState<SizingRunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [runsError, setRunsError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((data) => {
        if (cancelled) return
        setSites(data)
        setSelectedSiteId(data[0]?.site_id)
      })
      .catch((e) => {
        if (!cancelled)
          message.error(e instanceof Error ? e.message : '站点列表加载失败')
      })
      .finally(() => {
        if (!cancelled) setSitesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [message])

  const site = sites.find((s) => s.site_id === selectedSiteId)

  const loadRuns = useCallback(async (siteId: string) => {
    setRunsLoading(true)
    setRunsError(null)
    try {
      setRuns(await listSizingRuns(siteId))
    } catch (e) {
      setRuns([])
      setRunsError(e instanceof Error ? e.message : '历史分析加载失败')
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (selectedSiteId) void loadRuns(selectedSiteId)
    else setRuns([])
  }, [selectedSiteId, loadRuns])

  const submit = async () => {
    if (!site) return
    const pcsKw = toNumbers(pcsValues)
    const durationHours = toNumbers(durationValues)
    if (pcsKw.length === 0 || durationHours.length === 0) {
      message.warning('请至少各保留一个候选 PCS 功率和时长')
      return
    }
    setRunning(true)
    setRunError(null)
    try {
      const result = await createSizing({
        site_id: site.site_id,
        candidates: { pcs_kw: pcsKw, duration_hours: durationHours },
        representative_days: repDays,
        financial,
      })
      message.success('分析完成')
      navigate(`/sizing/${result.run_id}?site=${site.site_id}`)
    } catch (e) {
      setRunError(e instanceof Error ? e.message : '方案分析失败')
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        title="投资"
        description="储能买多大最划算？对比候选配置，算清每年赚多少、几年回本"
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Card title="分析参数">
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              选出想比较的储能规格（功率 × 时长），系统对每种组合模拟全年运行，对比回本年限和净收益
            </Typography.Text>
            <Form layout="inline" style={{ rowGap: 8 }}>
              <Form.Item label="站点" style={{ marginBottom: 0 }}>
                <Select
                  value={selectedSiteId}
                  onChange={setSelectedSiteId}
                  loading={sitesLoading}
                  disabled={sitesLoading || sites.length === 0}
                  placeholder="请选择站点"
                  style={{ minWidth: 200 }}
                  options={sites.map((s) => ({ value: s.site_id, label: s.name }))}
                  notFoundContent={
                    <Empty description="暂无站点" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                  }
                />
              </Form.Item>
              <Form.Item label="候选 PCS 功率（kW）" style={{ marginBottom: 0 }}>
                <Select
                  mode="tags"
                  value={pcsValues}
                  onChange={setPcsValues}
                  style={{ minWidth: 280 }}
                  placeholder="输入数字后回车添加"
                  options={PCS_PRESETS.map((v) => ({
                    value: String(v),
                    label: `${v} kW`,
                  }))}
                />
              </Form.Item>
              <Form.Item label="候选时长（h）" style={{ marginBottom: 0 }}>
                <Select
                  mode="tags"
                  value={durationValues}
                  onChange={setDurationValues}
                  style={{ minWidth: 200 }}
                  placeholder="输入数字后回车添加"
                  options={DURATION_PRESETS.map((v) => ({
                    value: String(v),
                    label: `${v} h`,
                  }))}
                />
              </Form.Item>
              <Form.Item label="代表日数量" style={{ marginBottom: 0 }}>
                <InputNumber
                  min={1}
                  max={7}
                  value={repDays}
                  onChange={(v) => setRepDays(v ?? 3)}
                  style={{ width: 80 }}
                />
              </Form.Item>
              <Form.Item style={{ marginBottom: 0 }}>
                <Button
                  type="primary"
                  onClick={submit}
                  loading={running}
                  disabled={!site}
                >
                  开始分析
                </Button>
              </Form.Item>
            </Form>
            <Collapse
              ghost
              items={[
                {
                  key: 'financial',
                  label: '财务假设（默认值已填，可展开调整）',
                  children: (
                    <Row gutter={[16, 12]}>
                      {FINANCIAL_FIELDS.map((f) => (
                        <Col xs={12} md={8} lg={6} key={f.key}>
                          <Form.Item label={f.label} style={{ marginBottom: 0 }}>
                            <InputNumber
                              value={financial[f.key]}
                              step={f.step}
                              min={0}
                              onChange={(v) =>
                                setFinancial((prev) => ({
                                  ...prev,
                                  [f.key]: v ?? 0,
                                }))
                              }
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Col>
                      ))}
                    </Row>
                  ),
                },
              ]}
            />
            {runError && (
              <Alert
                type="error"
                showIcon
                message="方案分析失败"
                description={runError}
              />
            )}
          </Space>
        </Card>

        <Card title="历史分析">
          {runsError ? (
            <Alert
              type="error"
              showIcon
              message="历史分析加载失败"
              description={runsError}
              action={
                <Button
                  size="small"
                  onClick={() => selectedSiteId && loadRuns(selectedSiteId)}
                >
                  重试
                </Button>
              }
            />
          ) : (
            <Table
              rowKey="run_id"
              size="small"
              loading={runsLoading}
              pagination={{ pageSize: 10, hideOnSinglePage: true }}
              dataSource={runs}
              locale={{ emptyText: <Empty description="暂无分析记录" /> }}
              columns={[
                {
                  title: '生成时间',
                  dataIndex: 'created_at',
                  render: (v: string) => dayjs(v).format('YYYY-MM-DD HH:mm:ss'),
                },
                {
                  title: '推荐配置',
                  key: 'recommended',
                  render: (_: unknown, r: SizingRunSummary) =>
                    r.recommended_pcs_kw != null &&
                    r.recommended_capacity_kwh != null ? (
                      `${r.recommended_pcs_kw} kW / ${r.recommended_capacity_kwh} kWh`
                    ) : (
                      <Typography.Text type="secondary">无推荐</Typography.Text>
                    ),
                },
                {
                  title: 'NPV（元）',
                  dataIndex: 'npv_yuan',
                  align: 'right',
                  render: (v: number | null) => (
                    <span style={{ color: v != null && v >= 0 ? '#3f8600' : '#cf1322' }}>
                      {fmtYuan(v)}
                    </span>
                  ),
                },
                {
                  title: '回本（年）',
                  dataIndex: 'payback_years',
                  align: 'right',
                  render: (v: number | null) => fmtYears(v),
                },
                {
                  title: '操作',
                  key: 'action',
                  width: 80,
                  render: (_: unknown, r: SizingRunSummary) => (
                    <Button
                      type="link"
                      size="small"
                      onClick={() =>
                        navigate(`/sizing/${r.run_id}?site=${r.site_id}`)
                      }
                    >
                      查看
                    </Button>
                  ),
                },
              ]}
            />
          )}
        </Card>
      </Space>
    </>
  )
}
