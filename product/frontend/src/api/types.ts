export interface SiteLocation {
  lat: number
  lon: number
  timezone: string
  province: string
  city: string
}

export interface SitePv {
  capacity_kwp: number
  tilt_deg: number
  azimuth_deg: number
  system_loss: number
}

export interface SiteLoad {
  annual_kwh: number
  peak_kw: number
  profile_type: string
}

export interface SiteStorage {
  pcs_kw: number
  capacity_kwh: number
  round_trip_eff: number
  soc_min: number
  soc_max: number
  cycles_per_day_max: number
  allow_grid_export: boolean
}

export interface Site {
  site_id: string
  name: string
  location: SiteLocation
  pv: SitePv
  load: SiteLoad
  storage: SiteStorage
  tariff_id: string
  created_at: string
  updated_at: string
}

export type TierName = 'sharp' | 'peak' | 'flat' | 'valley'

export interface TariffTier {
  name: TierName
  price: number
  hours: (string | number)[]
}

export interface Tariff {
  tariff_id: string
  name: string
  mode: string
  transmission_and_fund: number
  tiers: TariffTier[]
}

export type DataType = 'pv_power' | 'load_power' | 'storage_soc' | 'grid_power'

export interface CsvUploadRejected {
  row: number
  reason: string
}

export interface CsvUploadResult {
  site_id: string
  data_type: string
  total_rows: number
  accepted: number
  rejected: CsvUploadRejected[]
}

export interface MissingRange {
  start: string
  end: string
}

export type QualityStatus = 'complete' | 'partial' | 'empty'

export interface QualityReport {
  resolution_minutes: number
  expected_points: number
  measured: number
  estimated: number
  invalid: number
  missing: number
  completeness: number
  missing_ranges: MissingRange[]
  status: QualityStatus
}

export type ScheduleAction = 'charge' | 'discharge' | 'idle'

export type DataSource = 'measurements' | 'forecast'

export interface SchedulePoint {
  ts: string
  action: ScheduleAction
  power_kw: number
  soc_start: number
  soc_end: number
  pv_power_kw: number
  load_power_kw: number
  price: number
  tier: TierName
  grid_import_kw: number
  grid_export_kw: number
  pv_curtailment_kw: number
}

export interface BenefitBreakdown {
  discharge_value_yuan: number
  grid_charge_cost_yuan: number
  discharge_kwh: number
  grid_charge_kwh: number
  surplus_charge_kwh: number
}

export interface ScheduleResponse {
  run_id?: string
  site_id: string
  date: string
  strategy: string
  data_source?: DataSource
  expected_benefit_yuan: number
  baseline_cost_yuan: number
  optimized_cost_yuan: number
  equivalent_full_cycles: number
  schedule: SchedulePoint[]
  constraints_applied: string[]
  benefit_breakdown?: BenefitBreakdown | null
}

export interface CreateScheduleRequest {
  site_id: string
  date: string
  initial_soc: number
  strategy: string
  resolution_minutes: number
  source?: 'auto' | DataSource
}

export interface DispatchRunSummary {
  run_id: string
  site_id: string
  date: string
  strategy: string
  data_source?: DataSource
  initial_soc: number
  expected_benefit_yuan: number
  baseline_cost_yuan: number
  optimized_cost_yuan: number
  equivalent_full_cycles: number
  created_at: string
}

export interface DispatchRunDetail extends ScheduleResponse {
  input_digest?: string
  created_at?: string
}

export interface CurvePoint {
  ts: string
  pv_power_kw: number | null
  load_power_kw: number | null
  net_load_kw: number | null
  price: number | null
  tier: TierName
}

export interface CurvesResponse {
  site_id: string
  start: string
  end: string
  resolution_minutes: number
  points: CurvePoint[]
}

export interface SizingFinancial {
  battery_price_per_kwh: number
  pcs_price_per_kw: number
  install_ratio: number
  opex_ratio: number
  project_years: number
  battery_life_years: number
  replacement_cost_ratio: number
  discount_rate: number
  residual_ratio: number
}

export interface SizingRequest {
  site_id: string
  candidates: {
    pcs_kw: number[]
    duration_hours: number[]
  }
  representative_days: number
  financial: SizingFinancial
}

export interface SizingCandidate {
  pcs_kw: number
  capacity_kwh: number
  duration_h: number
  capex_yuan: number
  annual_benefit_yuan: number
  annual_opex_yuan: number
  payback_years: number | null
  npv_yuan: number
  irr: number | null
  equivalent_full_cycles_per_day: number
}

export interface SizingSensitivityPoint {
  delta: number
  annual_benefit_yuan: number
  payback_years: number | null
  npv_yuan: number
}

export interface SizingResult {
  run_id: string
  site_id: string
  created_at: string
  representative_dates: string[]
  recommended: SizingCandidate | null
  candidates: SizingCandidate[]
  sensitivity: {
    price_spread: SizingSensitivityPoint[]
    battery_price: SizingSensitivityPoint[]
  }
  assumptions: Record<string, unknown>
}

export interface SizingRunSummary {
  run_id: string
  site_id: string
  created_at: string
  recommended_pcs_kw: number | null
  recommended_capacity_kwh: number | null
  npv_yuan: number | null
  payback_years: number | null
}

export type ReportStatus = 'pending' | 'running' | 'succeeded' | 'failed'

export interface ReportJob {
  report_id: string
  status: ReportStatus
  error?: string | null
  download_url?: string | null
}

export type ForecastTarget = 'pv_power' | 'load_power' | 'price'

export interface ForecastRequest {
  site_id: string
  targets: ForecastTarget[]
  horizon_hours: number
  resolution_minutes: number
  options?: {
    include_uncertainty?: boolean
    pv_model?: 'auto' | 'statistical' | 'physics'
    intraday?: boolean
  }
}

export interface ForecastModel {
  name: string
  version: string
}

export interface ForecastPoint {
  ts: string
  value: number
}

export interface TargetForecast {
  target: ForecastTarget
  unit: string
  model: ForecastModel
  points: ForecastPoint[]
  p10: number[] | null
  p90: number[] | null
  accuracy: { metric: string; value: number } | null
  provenance?: {
    provider?: string | null
    fetched_at?: string | null
    factor?: number | null
    window?: string | null
  } | null
  notes: string
}

export interface ForecastResponse {
  site_id: string
  issued_at: string
  resolution_minutes: number
  results: TargetForecast[]
}

export interface OverviewSiteInfo {
  name: string
  pv_capacity_kwp: number
  storage_pcs_kw: number
  storage_capacity_kwh: number
  soc_min: number
  soc_max: number
}

export interface OverviewCurrentSoc {
  ts: string
  value: number
}

export interface OverviewDataStatus {
  date: string
  pv_completeness: number
  load_completeness: number
  pv_status: QualityStatus
  load_status: QualityStatus
}

export interface OverviewRunWindow {
  action: 'charge' | 'discharge'
  start: string
  end: string
  max_power_kw: number
}

export interface OverviewLatestRun {
  run_id: string
  date: string
  strategy: string
  data_source?: DataSource
  expected_benefit_yuan: number
  optimized_cost_yuan: number
  equivalent_full_cycles: number
  created_at: string
  windows: OverviewRunWindow[]
}

export interface SiteOverview {
  site_id: string
  generated_at: string
  site: OverviewSiteInfo
  current_soc: OverviewCurrentSoc | null
  latest_measurement_date: string | null
  data_status: OverviewDataStatus | null
  latest_run: OverviewLatestRun | null
}
