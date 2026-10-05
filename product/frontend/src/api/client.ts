import type {
  AccuracyResponse,
  CreateScheduleRequest,
  CsvUploadResult,
  CurvesResponse,
  DataType,
  DispatchRunDetail,
  ForecastRequest,
  ForecastResponse,
  ForecastTarget,
  LiveSimulateResponse,
  LiveStatus,
  ReplayRunDetail,
  ReplayRunSummary,
  RollingState,
  DispatchRunSummary,
  QualityReport,
  ReportJob,
  ScheduleResponse,
  Site,
  SiteOverview,
  SizingRequest,
  SizingResult,
  SizingRunSummary,
  Tariff,
} from './types'

const BASE_URL = '/api/v1'

export class ApiError extends Error {
  code?: string
  details?: unknown
  requestId?: string

  constructor(message: string, code?: string, details?: unknown, requestId?: string) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.details = details
    this.requestId = requestId
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE_URL}${path}`, init)
  } catch {
    throw new ApiError('网络请求失败，请确认后端服务已启动（http://127.0.0.1:8000）')
  }

  if (!res.ok) {
    let message = `请求失败（HTTP ${res.status}）`
    let code: string | undefined
    let details: unknown
    let requestId: string | undefined
    try {
      const body = await res.json()
      if (body && typeof body === 'object' && body.error) {
        if (typeof body.error.message === 'string' && body.error.message) {
          message = body.error.message
        }
        code = body.error.code
        details = body.error.details
        requestId = body.error.request_id
      } else if (body && typeof body === 'object' && typeof body.detail === 'string') {
        // FastAPI 默认错误结构（后端尚未接入统一错误格式时的兜底）
        message = `${message}：${body.detail}`
      }
    } catch {
      // 响应体不是 JSON 时保留默认错误信息
    }
    throw new ApiError(message, code, details, requestId)
  }

  return (await res.json()) as T
}

export async function listSites(): Promise<Site[]> {
  const data = await request<Site[] | { items?: Site[] }>('/sites')
  if (Array.isArray(data)) return data
  return data.items ?? []
}

export function renameSite(siteId: string, name: string): Promise<Site> {
  return request<Site>(`/sites/${encodeURIComponent(siteId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
}

export async function getTariff(tariffId: string): Promise<Tariff> {
  try {
    return await request<Tariff>(`/tariffs/${encodeURIComponent(tariffId)}`)
  } catch (e) {
    // 后端暂未实现 GET /tariffs/{id} 时，回退到列表接口按 id 查找
    if (e instanceof ApiError && e.message.includes('404')) {
      const list = await request<Tariff[]>('/tariffs')
      const found = list.find((t) => t.tariff_id === tariffId)
      if (found) return found
    }
    throw e
  }
}

export function uploadCsv(
  siteId: string,
  dataType: DataType,
  file: File,
): Promise<CsvUploadResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('data_type', dataType)
  return request<CsvUploadResult>(
    `/sites/${encodeURIComponent(siteId)}/measurements/csv`,
    { method: 'POST', body: form },
  )
}

export function getDataQuality(
  siteId: string,
  dataType: DataType,
  date: string,
): Promise<QualityReport> {
  const params = new URLSearchParams({ data_type: dataType, date })
  return request<QualityReport>(
    `/sites/${encodeURIComponent(siteId)}/data-quality?${params.toString()}`,
  )
}

export function createSchedule(body: CreateScheduleRequest): Promise<ScheduleResponse> {
  return request<ScheduleResponse>('/storage/schedule', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function runForecast(body: ForecastRequest): Promise<ForecastResponse> {
  return request<ForecastResponse>('/forecast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getLive(siteId: string): Promise<LiveStatus> {
  return request<LiveStatus>(`/sites/${encodeURIComponent(siteId)}/live`)
}

export function simulateLive(siteId: string): Promise<LiveSimulateResponse> {
  return request<LiveSimulateResponse>(
    `/sites/${encodeURIComponent(siteId)}/simulate-live`,
    { method: 'POST' },
  )
}

export function getRolling(siteId: string): Promise<RollingState> {
  return request<RollingState>(`/sites/${encodeURIComponent(siteId)}/rolling`)
}

export function runRolling(siteId: string): Promise<RollingState> {
  return request<RollingState>(
    `/sites/${encodeURIComponent(siteId)}/rolling/run`,
    { method: 'POST' },
  )
}

export async function listDispatchRuns(siteId: string): Promise<DispatchRunSummary[]> {
  const data = await request<DispatchRunSummary[] | { items?: DispatchRunSummary[] }>(
    `/sites/${encodeURIComponent(siteId)}/dispatch-runs`,
  )
  if (Array.isArray(data)) return data
  return data.items ?? []
}

export function getDispatchRun(siteId: string, runId: string): Promise<DispatchRunDetail> {
  return request<DispatchRunDetail>(
    `/sites/${encodeURIComponent(siteId)}/dispatch-runs/${encodeURIComponent(runId)}`,
  )
}

export function getCurves(
  siteId: string,
  start: string,
  end: string,
): Promise<CurvesResponse> {
  const params = new URLSearchParams({ start, end })
  return request<CurvesResponse>(
    `/sites/${encodeURIComponent(siteId)}/curves?${params.toString()}`,
  )
}

export function createSizing(body: SizingRequest): Promise<SizingResult> {
  return request<SizingResult>('/storage/sizing', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function listSizingRuns(siteId: string): Promise<SizingRunSummary[]> {
  const data = await request<SizingRunSummary[] | { items?: SizingRunSummary[] }>(
    `/sites/${encodeURIComponent(siteId)}/sizing-runs`,
  )
  if (Array.isArray(data)) return data
  return data.items ?? []
}

export function getSizingRun(siteId: string, runId: string): Promise<SizingResult> {
  return request<SizingResult>(
    `/sites/${encodeURIComponent(siteId)}/sizing-runs/${encodeURIComponent(runId)}`,
  )
}

export function createReport(siteId: string, sizingRunId: string): Promise<ReportJob> {
  return request<ReportJob>('/reports', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ site_id: siteId, sizing_run_id: sizingRunId }),
  })
}

export function getReport(reportId: string): Promise<ReportJob> {
  return request<ReportJob>(`/reports/${encodeURIComponent(reportId)}`)
}

export function getOverview(siteId: string): Promise<SiteOverview> {
  return request<SiteOverview>(`/sites/${encodeURIComponent(siteId)}/overview`)
}

// ---------- 观测体系 ----------

export function getAccuracy(
  siteId: string,
  target?: ForecastTarget,
  days = 30,
): Promise<AccuracyResponse> {
  const params = new URLSearchParams({ days: String(days) })
  if (target) params.set('target', target)
  return request<AccuracyResponse>(
    `/sites/${encodeURIComponent(siteId)}/accuracy?${params.toString()}`,
  )
}

// ---------- 回放实验 ----------

export async function listReplayRuns(siteId: string): Promise<ReplayRunSummary[]> {
  const data = await request<ReplayRunSummary[] | { items?: ReplayRunSummary[] }>(
    `/sites/${encodeURIComponent(siteId)}/replay-runs`,
  )
  if (Array.isArray(data)) return data
  return data.items ?? []
}

export function getReplayRun(runId: string): Promise<ReplayRunDetail> {
  return request<ReplayRunDetail>(`/replay-runs/${encodeURIComponent(runId)}`)
}

export function runReplay(siteId: string, testDays = 30): Promise<ReplayRunSummary> {
  return request<ReplayRunSummary>(`/sites/${encodeURIComponent(siteId)}/replay`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ test_days: testDays }),
  })
}
