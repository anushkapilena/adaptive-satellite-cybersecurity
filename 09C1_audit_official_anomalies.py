"""Independent Phase 9C audit. Read-only; does not modify any Phase 9A/9B/9C files."""
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'ESA-Mission1'
P = ROOT / 'results/phase9c'
conf = json.loads((ROOT / 'results/phase9a1/config.json').read_text(encoding='utf-8'))
chans = conf['channels']
seq_len = int(conf['sequence_length'])
step_ns = int(conf['grid_seconds']) * 1_000_000_000
max_age = pd.Timedelta(seconds=int(conf['max_observation_age_seconds']))
means = np.asarray(conf['scaler_mean'], dtype=np.float64)
scales = np.asarray(conf['scaler_scale'], dtype=np.float64)
splits = {k: (pd.Timestamp(v[0]).value, pd.Timestamp(v[1]).value) for k,v in conf['split_boundaries'].items()}
labels = pd.read_csv(DATA / 'labels.csv')
labels['StartTime'] = pd.to_datetime(labels['StartTime'], utc=True)
labels['EndTime'] = pd.to_datetime(labels['EndTime'], utc=True)
types = pd.read_csv(DATA / 'anomaly_types.csv')
labels = labels.merge(types[['ID','Category']], on='ID', how='left', validate='many_to_one')
labels['Category'] = labels['Category'].astype(str).str.strip()
positive = labels[(labels['Category'] == 'Anomaly') & labels['Channel'].isin(chans)]
events = positive.groupby('ID').agg(start=('StartTime','min'), end=('EndTime','max'))
all_intervals = [(int(r.StartTime.value),int(r.EndTime.value),str(r.ID)) for r in labels.itertuples(index=False)]
by_event = {}
for eid, group in positive.groupby('ID'):
    by_event[str(eid)] = {ch: [(int(r.StartTime.value), int(r.EndTime.value)) for r in group[group['Channel']==ch].itertuples(index=False)] for ch in chans}

# Force timestamp fields to string: pandas' default numeric inference may lose nanoseconds.
m = pd.read_csv(P/'window_manifest.csv', dtype={'start_ns':'string','end_ns':'string','event_id':'string','split':'string'})
a = pd.read_csv(P/'event_audit.csv')
errors = []
def check(condition, message):
    if not condition:
        errors.append(message)
        if len(errors) <= 20:
            print('ERROR:', message)

npzs = {}
for split in ('train','validation','test'):
    with np.load(P/f'{split}_official_anomalies.npz', allow_pickle=False) as z:
        npzs[split] = {k:z[k].copy() for k in ('X','start_ns','end_ns','event_id')}
    z = npzs[split]
    n = len(z['X'])
    check(z['X'].shape == (n,seq_len,len(chans)),f'{split}: incorrect X shape')
    check(all(len(z[k]) == n for k in ('start_ns','end_ns','event_id')),f'{split}: array lengths')
    check(bool(np.isfinite(z['X']).all()),f'{split}: nonfinite scaled values')
    check(bool(np.all(z['end_ns']-z['start_ns'] == (seq_len-1)*step_ns)),f'{split}: window duration')
    check(bool(np.all((z['start_ns']>=splits[split][0]) & (z['end_ns']<=splits[split][1]))),f'{split}: split boundary')
    sub = m[m['split']==split]
    check(len(sub)==n,f'{split}: manifest count')
    if len(sub)==n:
        check(np.array_equal(sub['start_ns'].map(int).to_numpy(dtype=np.int64),z['start_ns']),f'{split}: manifest start mismatch')
        check(np.array_equal(sub['end_ns'].map(int).to_numpy(dtype=np.int64),z['end_ns']),f'{split}: manifest end mismatch')
        check(np.array_equal(sub['event_id'].astype(str).to_numpy(),z['event_id'].astype(str)),f'{split}: manifest event mismatch')

check(len(a)==len(events), 'event audit does not cover all official selected-channel anomaly IDs')
check(len(m)==sum(len(z['X']) for z in npzs.values()),'total manifest length mismatch')
check(not m.duplicated(['split','event_id','start_ns','end_ns']).any(),'duplicate manifest rows')
check(bool((a['saved_windows']<=100).all()),'event cap exceeded')

# Independent checks against original official labels, not builder's own audit decisions.
for r in m.itertuples(index=False):
    eid = str(r.event_id)
    first, last = int(r.start_ns), int(r.end_ns)
    check(eid in by_event, f'unknown event {eid}')
    if eid not in by_event:
        continue
    ev = events.loc[eid]
    check(int(ev.start.value)<=first<=last<=int(ev.end.value),f'{eid}: outside event envelope')
    check(any(a<=first and last<=b for intervals in by_event[eid].values() for a,b in intervals),f'{eid}: no whole-window official target label')
    check(not any(other!=eid and a<=last and b>=first for a,b,other in all_intervals),f'{eid}: overlapping different official event')
    check((last-first)==(seq_len-1)*step_ns,f'{eid}: wrong time span')
    expected_channels = [ch for ch in chans if any(a<=first and last<=b for a,b in by_event[eid][ch])]
    check(str(r.target_channels)==';'.join(expected_channels),f'{eid}: target_channels mismatch')

for split, z in npzs.items():
    for eid in set(z['event_id'].astype(str)):
        check(int((z['event_id'].astype(str)==eid).sum()) == int(a.loc[a['event_id']==eid,'saved_windows'].sum()),f'{eid}: audit count mismatch')

# Recompute EVERY saved scaled window directly from raw ZIP telemetry, on exact saved
# timestamps, with same age-limited ffill. This is an independent source-value audit.
series = {}
for ch in chans:
    print('Loading raw',ch,flush=True)
    with ZipFile(DATA/'channels'/f'{ch}.zip') as archive:
        with archive.open(ch) as f:
            df = pd.read_pickle(f)  # Trusted ESA source only
    idx = pd.to_datetime(df.index,utc=True)
    s = pd.Series(pd.to_numeric(df[ch],errors='coerce').to_numpy(dtype=np.float32),index=idx).sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).last()
    series[ch] = s
    del df

max_abs = 0.0
compared = 0
for split,z in npzs.items():
    for i in range(len(z['X'])):
        first = pd.Timestamp(int(z['start_ns'][i]),tz='UTC')
        grid = pd.date_range(start=first,periods=seq_len,freq=f'{conf["grid_seconds"]}s')
        raw = np.column_stack([
            series[ch].loc[first-max_age:grid[-1]].reindex(grid,method='ffill',tolerance=max_age).to_numpy(dtype=np.float32)
            for ch in chans
        ])
        check(bool(np.isfinite(raw).all()),f'{split} window {i}: raw values unavailable')
        expected = ((raw.astype(np.float64)-means)/scales).astype(np.float32)
        difference = float(np.max(np.abs(expected-z['X'][i])))
        max_abs = max(max_abs,difference)
        if not np.allclose(expected,z['X'][i],rtol=1e-6,atol=2e-5,equal_nan=False):
            check(False,f'{split} window {i}: scaled telemetry differs, max={difference:.8g}')
        compared += 1
    print(split,'windows audited:',len(z['X']))

print('\n========== PHASE 9C.1 AUDIT ==========')
print('Saved windows:',sum(len(z['X']) for z in npzs.values()))
print('Official event IDs:',len(events))
print('Events with saved windows:',len(set(m['event_id'].astype(str))))
print('Raw-source windows independently compared:',compared)
print('Maximum absolute value difference:',max_abs)
print('Errors:',len(errors))
print('PHASE_9C1_AUDIT_PASSED' if not errors else 'PHASE_9C1_AUDIT_FAILED')
if errors:
    raise SystemExit(1)
