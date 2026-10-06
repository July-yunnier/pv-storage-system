export function fmtYuan(v: number | null | undefined): string {
  if (v == null) return '—'
  return Math.round(v).toLocaleString('zh-CN')
}

export function fmtWanYuan(v: number | null | undefined): string {
  if (v == null) return '—'
  return (v / 10000).toLocaleString('zh-CN', { maximumFractionDigits: 1 })
}

export function fmtYears(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toFixed(1)
}

export function fmtIrr(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)}%`
}
