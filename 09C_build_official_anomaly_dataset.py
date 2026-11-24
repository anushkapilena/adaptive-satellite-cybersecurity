"""Phase 9C: conservative, event-aware ESA Mission 1 anomaly windows.

Only official anomaly-positive windows are emitted. This is NOT a clean-negative
corpus or an ESA-ADB benchmark reproduction. Original Phase 9A/9B files are untouched.
"""
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'ESA-Mission1'
OUT = ROOT / 'results' / 'phase9c'
OUT.mkdir(parents=True, exist_ok=True)
config = json.loads((ROOT / 'results/phase9a1/config.json').read_text(encoding='utf-8'))
CHANNELS = config['channels']
SEQ_LEN = int(config['sequence_length'])
GRID_SECONDS = int(config['grid_seconds'])
MAX_AGE = pd.Timedelta(seconds=int(config['max_observation_age_seconds']))
MAX_WINDOWS_PER_EVENT = 100  # Equal event sampling weight; full timeline remains for later evaluation.

labels = pd.read_csv(DATA / 'labels.csv')
labels['StartTime'] = pd.to_datetime(labels['StartTime'], utc=True)
labels['EndTime'] = pd.to_datetime(labels['EndTime'], utc=True)
assert (labels['StartTime'] <= labels['EndTime']).all()
types = pd.read_csv(DATA / 'anomaly_types.csv')
labels = labels.merge(types[['ID', 'Category']], on='ID', how='left', validate='many_to_one')
labels['Category'] = labels['Category'].astype(str).str.strip()
anomaly = labels[(labels['Category'] == 'Anomaly') & labels['Channel'].isin(CHANNELS)].copy()
assert len(anomaly) > 0, 'No selected-channel official anomaly labels'

# Event metadata is computed from the original label intervals, not parquet context.
events = anomaly.groupby('ID').agg(start=('StartTime', 'min'), end=('EndTime', 'max')).sort_values('start')
splits = {name: (pd.Timestamp(pair[0]), pd.Timestamp(pair[1])) for name, pair in config['split_boundaries'].items()}

def event_split(start, end):
    eligible = [name for name, (a, b) in splits.items() if a <= start and end <= b]
    if len(eligible) != 1:
        raise RuntimeError(f'Event not contained in exactly one split: {start}, {end}: {eligible}')
    return eligible[0]

# Load trusted official ZIP pickle only. Do not use untrusted pickle files.
series = {}
for channel in CHANNELS:
    print('Loading', channel, flush=True)
    with ZipFile(DATA / 'channels' / f'{channel}.zip') as z:
        with z.open(channel) as f:
            df = pd.read_pickle(f)
    idx = pd.to_datetime(df.index, utc=True)
    s = pd.Series(pd.to_numeric(df[channel], errors='coerce').to_numpy(dtype=np.float32),
                  index=idx, name=channel).sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).last()
    series[channel] = s
    del df

means = np.asarray(config['scaler_mean'], dtype=np.float64)
scales = np.asarray(config['scaler_scale'], dtype=np.float64)
assert len(means) == len(CHANNELS) == len(scales) and (scales > 0).all()

# Use exact official labelled intervals for each channel. A window must be fully
# inside a target labelled interval on at least one channel; do not label gaps
# within a long event's min/max envelope as anomalous.
def interval_contains_whole_window(intervals, first, last):
    return any(a <= first and last <= b for a, b in intervals)

# Conservative exclusion: any other official event (any category, any channel)
# touching a window disqualifies it, to avoid ambiguous attribution.
all_intervals = [(r.StartTime, r.EndTime, str(r.ID)) for r in labels.itertuples(index=False)]
by_event = {}
for event_id, group in anomaly.groupby('ID'):
    by_event[str(event_id)] = {
        ch: [(r.StartTime, r.EndTime) for r in group[group['Channel'] == ch].itertuples(index=False)]
        for ch in CHANNELS
    }

buffers = {name: {'X': [], 'start_ns': [], 'end_ns': [], 'event_id': []} for name in splits}
manifest_rows = []
audit_rows = []

for event_id, ev in events.iterrows():
    event_id = str(event_id)
    split = event_split(ev.start, ev.end)
    grid = pd.date_range(start=ev.start, end=ev.end, freq=f'{GRID_SECONDS}s')
    aligned = pd.DataFrame(index=grid)
    for ch in CHANNELS:
        local = series[ch].loc[ev.start - MAX_AGE:ev.end]
        aligned[ch] = local.reindex(grid, method='ffill', tolerance=MAX_AGE)
    values = aligned[CHANNELS].to_numpy(dtype=np.float32)
    valid = np.isfinite(values).all(axis=1)
    candidates = []
    discarded_other = 0
    discarded_unlabelled = 0
    # Non-overlapping event-anchored windows. Separate event grids are NOT a
    # continuous mission timeline and cannot establish first-alert latency.
    for i in range(0, len(grid) - SEQ_LEN + 1, SEQ_LEN):
        j = i + SEQ_LEN
        if not valid[i:j].all():
            continue
        first, last = grid[i], grid[j - 1]
        target_channels = [ch for ch in CHANNELS if interval_contains_whole_window(by_event[event_id][ch], first, last)]
        if not target_channels:
            discarded_unlabelled += 1
            continue
        if any(other_id != event_id and a <= last and b >= first for a, b, other_id in all_intervals):
            discarded_other += 1
            continue
        candidates.append((i, j, target_channels))
    if len(candidates) > MAX_WINDOWS_PER_EVENT:
        selected_idx = np.linspace(0, len(candidates)-1, MAX_WINDOWS_PER_EVENT, dtype=int)
        candidates = [candidates[k] for k in selected_idx]
    for i, j, target_channels in candidates:
        scaled = ((values[i:j].astype(np.float64) - means) / scales).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise RuntimeError(f'Nonfinite scaled window: {event_id}')
        b = buffers[split]
        b['X'].append(scaled)
        b['start_ns'].append(grid[i].value)
        b['end_ns'].append(grid[j-1].value)
        b['event_id'].append(event_id)
        manifest_rows.append({'split': split, 'event_id': event_id,
                              'start_ns': grid[i].value, 'end_ns': grid[j-1].value,
                              'start_time': grid[i].isoformat(), 'end_time': grid[j-1].isoformat(),
                              'official_label': 'operational_anomaly',
                              'target_channels': ';'.join(target_channels),
                              'label_policy': 'whole_window_inside_at_least_one_channel_label',
                              'grid_anchor': 'event_start'})
    audit_rows.append({'event_id': event_id, 'split': split, 'grid_points': len(grid),
                       'valid_grid_points': int(valid.sum()), 'saved_windows': len(candidates),
                       'excluded_other_event_windows': discarded_other,
                       'excluded_not_fully_labelled_windows': discarded_unlabelled})
    print(f'{event_id}: {split}, saved={len(candidates)}', flush=True)

for name, b in buffers.items():
    X = np.stack(b['X']) if b['X'] else np.empty((0, SEQ_LEN, len(CHANNELS)), dtype=np.float32)
    np.savez_compressed(OUT / f'{name}_official_anomalies.npz',
                        X=X, start_ns=np.asarray(b['start_ns'], dtype=np.int64),
                        end_ns=np.asarray(b['end_ns'], dtype=np.int64),
                        event_id=np.asarray(b['event_id'], dtype='U64'))
    print(name, X.shape, 'unique events:', len(set(b['event_id'])))

pd.DataFrame(manifest_rows).to_csv(OUT / 'window_manifest.csv', index=False)
pd.DataFrame(audit_rows).to_csv(OUT / 'event_audit.csv', index=False)
metadata = {'phase': '9C', 'channels': CHANNELS, 'sequence_length': SEQ_LEN,
            'grid_seconds': GRID_SECONDS, 'max_observation_age_seconds': int(MAX_AGE.total_seconds()),
            'max_windows_per_event': MAX_WINDOWS_PER_EVENT,
            'label': 'official operational anomaly, not verified cyberattack',
            'source_scaler': 'results/phase9a1/config.json',
            'limitations': ['Event-anchored grids differ from Phase 9A chunk grids.',
                            'This is a positive-only window subset, not continuous event-level evaluation.',
                            'Window count is capped per event; not suitable for first-alert latency.',
                            'Unlabelled readings are not verified normal.',
                            'No claim of comparability with original ESA-ADB benchmark.']}
(OUT / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('\nSaved to', OUT)
print('Events with zero saved windows:', sum(r['saved_windows'] == 0 for r in audit_rows))
print('PHASE 9C BUILDER COMPLETE (requires independent audit)')
