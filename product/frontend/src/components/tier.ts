import type { TierName } from '../api/types'

export const TIER_META: Record<TierName, { label: string; color: string }> = {
  sharp: { label: '尖峰', color: '#cf1322' },
  peak: { label: '高峰', color: '#fa8c16' },
  flat: { label: '平段', color: '#8c8c8c' },
  valley: { label: '低谷', color: '#389e0d' },
}
