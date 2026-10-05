import { useEffect, useState } from 'react'
import { Select } from 'antd'
import { listSites } from '../api/client'
import type { Site } from '../api/types'

/** 站点列表 + 选中态，各算法页共用。 */
export function useSites() {
  const [sites, setSites] = useState<Site[]>([])
  const [siteId, setSiteId] = useState<string>()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((data) => {
        if (cancelled) return
        setSites(data)
        setSiteId((prev) =>
          prev && data.some((s) => s.site_id === prev) ? prev : data[0]?.site_id,
        )
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const site = sites.find((s) => s.site_id === siteId)
  return { sites, site, siteId, setSiteId, loading }
}

export function SiteSelect({
  sites,
  value,
  onChange,
}: {
  sites: Site[]
  value?: string
  onChange: (v: string) => void
}) {
  return (
    <Select
      value={value}
      onChange={onChange}
      options={sites.map((s) => ({ value: s.site_id, label: s.name }))}
      style={{ minWidth: 200 }}
      placeholder="选择站点"
    />
  )
}
