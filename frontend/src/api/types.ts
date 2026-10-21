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
  charge_from_pv_kw?: number
  charge_from_grid_kw?: number
}

export interface PlanSegment {
  start: string
  end: string
  action: 'charge_pv' | 'charge_grid' | 'discharge'
  tier: TierName | null
  avg_power_kw: number
  text: string
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
  explanation?: PlanSegment[] | null
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
  auto_generated?: boolean
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
    load_model?: 'statistical' | 'advanced'
    intraday?: boolean
    explain?: boolean
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
  components?: Record<string, number[]> | null
  provenance?: {
    provider?: string | null
    fetched_at?: string | null
    factor?: number | null
    window?: string | null
    explain?: {
      weather: {
        ts: string
        ghi_wm2: number
        dni_wm2: number
        dhi_wm2: number
        temperature_c: number | null
      }[] | null
      backtest: {
        dates: string[]
        sum_measured_kwh: number
        sum_physics_kwh: number
        factor: number
      } | null
    } | null
  } | null
  notes: string
}

export interface ForecastResponse {
  site_id: string
  issued_at: string
  resolution_minutes: number
  results: TargetForecast[]
}

export interface LiveLatestPoint {
  ts: string
  value: number
}

export interface LiveStatus {
  site_id: string
  now_local: string
  latest_pv: LiveLatestPoint | null
  latest_load: LiveLatestPoint | null
  latest_soc: LiveLatestPoint | null
  latest_grid: LiveLatestPoint | null
  price_now: {
    price: number
    tier: TierName | null
    next_change_at: string | null
    next_tier: TierName | null
  }
  today_plan: {
    run_id: string
    expected_benefit_yuan: number
    created_at: string
  } | null
  pv_points_today: number
  load_points_today: number
  expected_points_today: number
}

export interface LiveSimulateResponse {
  site_id: string
  inserted: Record<string, number>
  up_to_slot: number
}

export interface RollingRevision {
  revision: number
  created_at: string
  initial_soc: number
  expected_benefit_yuan: number
  trigger_reason: string
}

export interface RollingState {
  site_id: string
  date: string
  base_run_id: string | null
  base_expected_benefit_yuan: number | null
  revisions: RollingRevision[]
  latest_schedule: SchedulePoint[] | null
  latest_revision: number
  cooldown_until: string | null
  deviation_now: number | null
  last_trigger_reason: string | null
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

// ---------- 观测体系：预测精度对账 ----------

export interface AccuracyRow {
  site_id: string
  target: ForecastTarget
  method: string
  target_date: string
  skill_all_day: number | null
  skill_daylight: number | null
  rmse: number | null
  mae: number | null
  bias: number | null
  n_points: number
}

export interface AccuracyMethodSummary {
  method: string
  days: number
  skill_all_day: number | null
  skill_daylight: number | null
  rmse: number | null
  bias: number | null
}

export interface AccuracyResponse {
  site_id: string
  days: number
  rows: AccuracyRow[]
  summary: Record<string, AccuracyMethodSummary>
}

// ---------- 回放实验 ----------

export interface ReplayRunSummary {
  run_id: string
  site_id: string
  config: {
    test_days: number
    test_window: [string, string]
    pv_methods: string[]
    strategies: string[]
    weather_proxy?: string
  }
  status: 'running' | 'finished' | 'failed'
  created_at: string
  finished_at: string | null
}

export interface ReplayForecastMetric {
  days: number
  skill_all_day: number | null
  skill_daylight: number | null
  rmse: number | null
  bias: number | null
}

export interface ReplayStrategyMetric {
  days: number
  total_cost_yuan: number
  total_saving_yuan: number
  avg_daily_saving_yuan: number
  max_peak_import_kw: number
  total_discharge_kwh: number
  pv_self_use_ratio: number
  soc_violations: number
}

export interface ReplayDayResult {
  date: string
  pv_method: string
  strategy: string
  metrics: {
    cost_yuan: number
    saving_vs_d0_yuan: number
    grid_import_kwh: number
    peak_import_kw: number
    pv_self_use_ratio: number
    curtailment_kwh: number
    discharge_kwh: number
    end_soc: number
    soc_violations: number
  }
}

export interface ReplayRunDetail extends ReplayRunSummary {
  summary: {
    forecasts: Record<string, ReplayForecastMetric>
    strategies: Record<string, ReplayStrategyMetric>
  } | null
  day_results: ReplayDayResult[]
}

// ---------- 电价中心 ----------

export type PriceModelKind = 'auto' | 'lgbm' | 'statistical' | 'tou'

export interface PriceForecastPoint {
  ts: string
  p10: number | null
  p50: number
  p90: number | null
  tou_price: number | null
  tier: TierName | null
}

export interface PriceArbitrage {
  charge_start: string
  discharge_start: string
  slot_hours: number
  charge_price: number
  discharge_price: number
  gross_spread: number
  net_spread: number
  round_trip_eff: number
  note: string
}

export interface PriceForecastSummary {
  mean: number
  max: number
  min: number
  peak_valley_spread: number
  base_tou_spread: number | null
  max_hours: string[]
  min_hours: string[]
  arbitrage: PriceArbitrage | null
  holidays: string[]
}

export interface PriceForecastView {
  site_id: string
  issued_at: string
  horizon_hours: number
  resolution_minutes: number
  model: string
  method_note: string
  data_note: string
  uncertainty: boolean
  points: PriceForecastPoint[]
  summary: PriceForecastSummary
  accuracy: {
    metric: string
    value: number
    caliber: Record<string, unknown>
    evaluated_on: string
  } | null
}

export interface PriceHistoryView {
  site_id: string
  unit: string
  resolution: 'hourly' | 'daily'
  points: { ts: string; price: number; source: string }[]
  stats: {
    count: number
    mean: number | null
    max: number | null
    min: number | null
    negative_hours: number
    spike_hours: number
  }
  bounds: { first: string; last: string } | null
}
