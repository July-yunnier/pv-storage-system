#!/usr/bin/env python3
"""Read raw XLSX without Excel dependencies and reconcile delivered Solar8 CSV.

Usage: python backend/scripts/audit_solar8_source.py /path/to/site8.xlsx
"""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/official_experiment/solar8_diagnosis'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('xlsx',type=Path);args=ap.parse_args()
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(args.xlsx) as archive:
        strings=[''.join(v.itertext()) for v in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        rows=ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns)
        records=[]
        for row in rows[1:]:
            values=[]
            for cell in row:
                value=cell.find('s:v',ns)
                value=value.text if value is not None else ''.join(cell.itertext())
                if cell.attrib.get('t')=='s':value=strings[int(value)]
                values.append(value)
            records.append(values)
    raw=pd.DataFrame(records,columns=['excel_ts','tsi','dni','ghi','temperature_c','pressure','humidity','power_mw'])
    for col in raw.columns:raw[col]=pd.to_numeric(raw[col],errors='coerce')
    raw['ts']=(pd.Timestamp('1899-12-30')+pd.to_timedelta(raw.excel_ts,unit='D')).dt.round('min').dt.tz_localize('Asia/Shanghai')
    raw['power_kw']=raw.power_mw*1000
    delivered=pd.read_csv(ROOT/'reports/official_experiment/p0a_ampdnet/data/Solar8_processed_3.csv')
    delivered.ts=pd.to_datetime(delivered.ts)
    merged=delivered.merge(raw[['ts','power_kw']],on='ts',how='left',suffixes=('','_raw'))
    merged['original_present']=merged.power_kw_raw.notna()
    merged[['ts','original_present']].to_csv(OUT/'solar8_original_presence.csv',index=False)
    # Three-component geometry cannot be recovered from scalar copies.
    daylight=raw[(raw.ts.dt.hour>=6)&(raw.ts.dt.hour<19)].copy()
    fit=daylight[daylight.tsi>1]
    full=pd.date_range(raw.ts.min(),raw.ts.max(),freq='15min')
    missing=full.difference(raw.ts)
    groups=[]
    for ts in missing:
        if not groups or ts-groups[-1][-1]!=pd.Timedelta(minutes=15):groups.append([ts])
        else:groups[-1].append(ts)
    report={'source_xlsx':str(args.xlsx),'source_sha256':hashlib.sha256(args.xlsx.read_bytes()).hexdigest(),
        'raw_rows':len(raw),'raw_daylight_rows':len(daylight),'delivered_rows':len(delivered),
        'missing_full_timestamps':len(missing),'missing_daylight_timestamps':int((~merged.original_present).sum()),
        'raw_irradiance_correlation':float(fit.dni.corr(fit.tsi)),
        'raw_dni_tsi_ratio_mean':float((fit.dni/fit.tsi).mean()),
        'raw_dni_tsi_ratio_std':float((fit.dni/fit.tsi).std()),
        'raw_ghi_tsi_ratio_mean':float((fit.ghi/fit.tsi).mean()),
        'raw_ghi_tsi_ratio_std':float((fit.ghi/fit.tsi).std()),
        'matched_label_max_abs_kw':float(abs(merged.loc[merged.original_present,'power_kw']-merged.loc[merged.original_present,'power_kw_raw']).max()),
        'gaps':[{'start':g[0].isoformat(),'end':g[-1].isoformat(),'points':len(g)} for g in groups],
        'limits':'Raw file is supplied dataset, not independent sensor authenticity validation; no cloud images.'}
    (OUT/'raw_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
