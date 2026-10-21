import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  App as AntdApp,
  Button,
  Card,
  Col,
  Collapse,
  DatePicker,
  Descriptions,
  Empty,
  Form,
  Input,
  List,
  Modal,
  Progress,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  Upload,
} from 'antd'
import { EditOutlined, InboxOutlined } from '@ant-design/icons'
import dayjs, { Dayjs } from 'dayjs'
import PageHeader from '../components/PageHeader'
import { TIER_META } from '../components/tier'
import {
  getDataQuality,
  getTariff,
  listSites,
  renameSite,
  uploadCsv,
} from '../api/client'
import type {
  CsvUploadResult,
  DataType,
  QualityReport,
  Site,
  Tariff,
  TierName,
} from '../api/types'

const DATA_TYPE_OPTIONS: { value: DataType; label: string }[] = [
  { value: 'pv_power', label: '光伏功率' },
  { value: 'load_power', label: '负荷功率' },
  { value: 'storage_soc', label: '储能SOC' },
  { value: 'grid_power', label: '电网功率' },
  { value: 'price', label: '电能量价格' },
]

function formatHour(h: string | number): string {
  return typeof h === 'number' ? `${h}:00` : h
}

function SiteInfoCard({
  site,
  tariffName,
  loading,
  error,
  onRetry,
  onRenamed,
}: {
  site?: Site
  tariffName?: string
  loading: boolean
  error: string | null
  onRetry: () => void
  onRenamed: (site: Site) => void
}) {
  const { message } = AntdApp.useApp()
  const [editOpen, setEditOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [saving, setSaving] = useState(false)

  const openEdit = () => {
    if (!site) return
    setNewName(site.name)
    setEditOpen(true)
  }

  const submitRename = async () => {
    if (!site) return
    const name = newName.trim()
    if (!name) {
      message.warning('站点名称不能为空')
      return
    }
    setSaving(true)
    try {
      const updated = await renameSite(site.site_id, name)
      onRenamed(updated)
      setEditOpen(false)
      message.success('站点名称已更新')
    } catch (e) {
      message.error(e instanceof Error ? e.message : '更新站点名称失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card title="站点信息">
      {loading ? (
        <div style={{ textAlign: 'center', padding: 32 }}>
          <Spin />
        </div>
      ) : error ? (
        <Alert
          type="error"
          showIcon
          message="站点列表加载失败"
          description={error}
          action={
            <Button size="small" onClick={onRetry}>
              重试
            </Button>
          }
        />
      ) : !site ? (
        <Empty description="暂无站点数据" />
      ) : (
        <Descriptions column={{ xs: 1, sm: 2, md: 3 }} bordered size="small">
          <Descriptions.Item label="名称">
            <Space size={4}>
              {site.name}
              <Button
                type="link"
                size="small"
                icon={<EditOutlined />}
                onClick={openEdit}
              >
                编辑
              </Button>
            </Space>
          </Descriptions.Item>
          <Descriptions.Item label="位置">
            {site.location.province} {site.location.city}（{site.location.lon},{' '}
            {site.location.lat}）
          </Descriptions.Item>
          <Descriptions.Item label="时区">{site.location.timezone}</Descriptions.Item>
          <Descriptions.Item label="光伏容量">
            {site.pv.capacity_kwp} kWp
          </Descriptions.Item>
          <Descriptions.Item label="储能">
            PCS {site.storage.pcs_kw} kW / 容量 {site.storage.capacity_kwh} kWh
          </Descriptions.Item>
          <Descriptions.Item label="SOC 区间">
            {Math.round(site.storage.soc_min * 100)}% ~{' '}
            {Math.round(site.storage.soc_max * 100)}%
          </Descriptions.Item>
          <Descriptions.Item label="关联电价">
            {site.tariff_id}
            {tariffName ? `（${tariffName}）` : ''}
          </Descriptions.Item>
        </Descriptions>
      )}

      <Modal
        title="编辑站点名称"
        open={editOpen}
        onOk={submitRename}
        onCancel={() => setEditOpen(false)}
        confirmLoading={saving}
        okText="保存"
        cancelText="取消"
        destroyOnHidden
      >
        <Input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="请输入新的站点名称"
          maxLength={50}
          onPressEnter={submitRename}
        />
      </Modal>
    </Card>
  )
}

function TariffCard({
  tariff,
  loading,
  error,
}: {
  tariff: Tariff | null
  loading: boolean
  error: string | null
}) {
  return (
    <Card title={tariff ? `电价模板：${tariff.name}` : '电价模板'}>
      {loading ? (
        <div style={{ textAlign: 'center', padding: 32 }}>
          <Spin />
        </div>
      ) : error ? (
        <Alert type="error" showIcon message="电价模板加载失败" description={error} />
      ) : !tariff ? (
        <Empty description="暂无电价数据" />
      ) : (
        <>
          <Table
            rowKey="name"
            size="small"
            pagination={false}
            dataSource={tariff.tiers}
            columns={[
              {
                title: '档位',
                dataIndex: 'name',
                render: (name: TierName) => {
                  const meta = TIER_META[name] ?? { label: name, color: '#8c8c8c' }
                  return <Tag color={meta.color}>{meta.label}</Tag>
                },
              },
              {
                title: '价格（元/kWh）',
                dataIndex: 'price',
                render: (price: number) => price.toFixed(4).replace(/0+$/, '').replace(/\.$/, '.0'),
              },
              {
                title: '时段',
                dataIndex: 'hours',
                render: (hours: (string | number)[]) => hours.map(formatHour).join('、'),
              },
            ]}
          />
          <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
            输配电价及政府性基金：{tariff.transmission_and_fund ?? 0.21} 元/kWh（示例占位）
          </Typography.Paragraph>
        </>
      )}
    </Card>
  )
}

function UploadCard({ site }: { site?: Site }) {
  const { message } = AntdApp.useApp()
  const [dataType, setDataType] = useState<DataType>('pv_power')
  const [priceResolution, setPriceResolution] = useState(15)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<CsvUploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFile = async (file: File) => {
    if (!site) {
      message.warning('请先选择站点')
      return
    }
    if (!file.name.toLowerCase().endsWith('.csv')) {
      message.warning('仅支持 .csv 文件')
      return
    }
    setUploading(true)
    setResult(null)
    setError(null)
    try {
      const res = await uploadCsv(site.site_id, dataType, file, dataType === 'price' ? priceResolution : 15)
      setResult(res)
      message.success(`上传完成：接受 ${res.accepted} 条`)
    } catch (e) {
      const msg = e instanceof Error ? e.message : '上传失败'
      setError(msg)
      message.error(msg)
    } finally {
      setUploading(false)
    }
  }

  return (
    <Card title="数据上传（CSV）">
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Form layout="inline" style={{ rowGap: 8 }}>
          <Form.Item label="数据类型" style={{ marginBottom: 0 }}>
            <Select
              value={dataType}
              onChange={setDataType}
              options={DATA_TYPE_OPTIONS}
              style={{ width: 160 }}
            />
          </Form.Item>
          {dataType === 'price' && <Form.Item label="原始结算间隔" style={{ marginBottom: 0 }}>
            <Select value={priceResolution} onChange={setPriceResolution} style={{ width: 120 }}
              options={[{ value: 15, label: '15 分钟' }, { value: 60, label: '1 小时' }]} />
          </Form.Item>}
        </Form>
        {dataType === 'price' && <Alert type="info" showIcon
          message="价格 CSV：ts,value；单位为人民币/千瓦时，仅填写电能量价格，不含输配和基金附加；支持负电价。" />}
        <Upload.Dragger
          accept=".csv"
          maxCount={1}
          disabled={!site || uploading}
          beforeUpload={(file) => {
            void handleFile(file)
            return false
          }}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽 CSV 文件到此处上传</p>
          <p className="ant-upload-hint">
            上传目标：{site ? site.name : '请先选择站点'}
          </p>
        </Upload.Dragger>
        {uploading && <Spin />}
        {error && (
          <Alert type="error" showIcon message="上传失败" description={error} />
        )}
        {result && (
          <>
            <Alert
              type={result.rejected.length > 0 ? 'warning' : 'success'}
              showIcon
              message={`共 ${result.total_rows} 行，接受 ${result.accepted} 条${
                result.rejected.length > 0 ? `，拒绝 ${result.rejected.length} 条` : ''
              }`}
            />
            {result.rejected.length > 0 && (
              <Collapse
                items={[
                  {
                    key: 'rejected',
                    label: `查看被拒绝的 ${result.rejected.length} 行`,
                    children: (
                      <Table
                        rowKey="row"
                        size="small"
                        pagination={{ pageSize: 10, hideOnSinglePage: true }}
                        dataSource={result.rejected}
                        columns={[
                          { title: '行号', dataIndex: 'row', width: 100 },
                          { title: '拒绝原因', dataIndex: 'reason' },
                        ]}
                      />
                    ),
                  },
                ]}
              />
            )}
          </>
        )}
      </Space>
    </Card>
  )
}

const STATUS_COLOR: Record<string, string> = {
  complete: '#52c41a',
  partial: '#faad14',
  empty: '#f5222d',
}

function QualityCard({ site }: { site?: Site }) {
  const [dataType, setDataType] = useState<DataType>('pv_power')
  const [date, setDate] = useState<Dayjs>(dayjs())
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState<QualityReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  const query = async () => {
    if (!site) return
    setLoading(true)
    setError(null)
    try {
      const data = await getDataQuality(
        site.site_id,
        dataType,
        date.format('YYYY-MM-DD'),
      )
      setReport(data)
    } catch (e) {
      setReport(null)
      setError(e instanceof Error ? e.message : '数据质量查询失败')
    } finally {
      setLoading(false)
    }
  }

  const percent = report
    ? Math.round(
        (report.completeness <= 1 ? report.completeness * 100 : report.completeness) * 10,
      ) / 10
    : 0

  return (
    <Card title="数据质量">
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Form layout="inline" style={{ rowGap: 8 }}>
          <Form.Item label="日期" style={{ marginBottom: 0 }}>
            <DatePicker
              value={date}
              onChange={(d) => d && setDate(d)}
              allowClear={false}
            />
          </Form.Item>
          <Form.Item label="数据类型" style={{ marginBottom: 0 }}>
            <Select
              value={dataType}
              onChange={setDataType}
              options={DATA_TYPE_OPTIONS}
              style={{ width: 160 }}
            />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0 }}>
            <Button type="primary" onClick={query} loading={loading} disabled={!site}>
              查询
            </Button>
          </Form.Item>
        </Form>

        {error && (
          <Alert type="error" showIcon message="查询失败" description={error} />
        )}

        {report && report.status === 'empty' && (
          <Alert type="info" showIcon message="当日无数据" />
        )}

        {report && report.status !== 'empty' && (
          <>
            <Row gutter={24} align="middle">
              <Col>
                <Progress
                  type="circle"
                  percent={percent}
                  strokeColor={STATUS_COLOR[report.status] ?? '#faad14'}
                  size={120}
                />
                <div style={{ textAlign: 'center', marginTop: 8 }}>
                  <Typography.Text type="secondary">完整度</Typography.Text>
                </div>
              </Col>
              <Col flex="auto">
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic title="实测" value={report.measured} />
                  </Col>
                  <Col span={6}>
                    <Statistic title="估计" value={report.estimated} />
                  </Col>
                  <Col span={6}>
                    <Statistic title="异常" value={report.invalid} />
                  </Col>
                  <Col span={6}>
                    <Statistic title="缺失" value={report.missing} />
                  </Col>
                </Row>
              </Col>
            </Row>
            {report.missing_ranges.length > 0 && (
              <List
                size="small"
                header={<Typography.Text strong>缺失时间段</Typography.Text>}
                bordered
                dataSource={report.missing_ranges}
                renderItem={(r) => (
                  <List.Item>
                    {dayjs(r.start).format('MM-DD HH:mm')} ~{' '}
                    {dayjs(r.end).format('MM-DD HH:mm')}
                  </List.Item>
                )}
              />
            )}
          </>
        )}
      </Space>
    </Card>
  )
}

export default function DataPage() {
  const { message } = AntdApp.useApp()
  const [sites, setSites] = useState<Site[]>([])
  const [sitesLoading, setSitesLoading] = useState(true)
  const [sitesError, setSitesError] = useState<string | null>(null)
  const [selectedSiteId, setSelectedSiteId] = useState<string>()
  const [tariff, setTariff] = useState<Tariff | null>(null)
  const [tariffLoading, setTariffLoading] = useState(false)
  const [tariffError, setTariffError] = useState<string | null>(null)

  const loadSites = useCallback(async () => {
    setSitesLoading(true)
    setSitesError(null)
    try {
      const data = await listSites()
      setSites(data)
      setSelectedSiteId((prev) =>
        prev && data.some((s) => s.site_id === prev) ? prev : data[0]?.site_id,
      )
    } catch (e) {
      const msg = e instanceof Error ? e.message : '站点列表加载失败'
      setSites([])
      setSitesError(msg)
      message.error(msg)
    } finally {
      setSitesLoading(false)
    }
  }, [message])

  useEffect(() => {
    void loadSites()
  }, [loadSites])

  const site = sites.find((s) => s.site_id === selectedSiteId)

  useEffect(() => {
    if (!site) {
      setTariff(null)
      setTariffError(null)
      return
    }
    let cancelled = false
    setTariffLoading(true)
    setTariffError(null)
    getTariff(site.tariff_id)
      .then((data) => {
        if (!cancelled) setTariff(data)
      })
      .catch((e) => {
        if (!cancelled) {
          setTariff(null)
          setTariffError(e instanceof Error ? e.message : '电价模板加载失败')
        }
      })
      .finally(() => {
        if (!cancelled) setTariffLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [site])

  const handleRenamed = (updated: Site) => {
    setSites((prev) =>
      prev.map((s) => (s.site_id === updated.site_id ? updated : s)),
    )
  }

  return (
    <>
      <PageHeader
        title="数据管理"
        description="站点配置、电价模板、数据上传与数据质量检查"
        extra={
          <Space size={8}>
            <Typography.Text type="secondary">当前站点</Typography.Text>
            <Select
              value={selectedSiteId}
              onChange={setSelectedSiteId}
              loading={sitesLoading}
              disabled={sitesLoading || sites.length === 0}
              placeholder={sitesError ? '站点列表加载失败' : '请选择站点'}
              style={{ minWidth: 260 }}
              options={sites.map((s) => ({ value: s.site_id, label: s.name }))}
              notFoundContent={<Empty description="暂无站点" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
            />
          </Space>
        }
      />
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <SiteInfoCard
          site={site}
          tariffName={tariff?.name}
          loading={sitesLoading}
          error={sitesError}
          onRetry={loadSites}
          onRenamed={handleRenamed}
        />
        <TariffCard tariff={tariff} loading={tariffLoading} error={tariffError} />
        <UploadCard site={site} />
        <QualityCard site={site} />
      </Space>
    </>
  )
}
