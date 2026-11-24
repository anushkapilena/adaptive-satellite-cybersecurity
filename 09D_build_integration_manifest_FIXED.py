"""Phase 9D: verified, non-destructive experiment index (not an ESA benchmark reproduction).

Run from the project root using your project virtual environment.
Writes only results/phase9d. No input NPZ is modified or duplicated.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'
OUT = R / 'phase9d'
SPLITS = ('train', 'validation', 'test')
SEQ_LEN, NCHANNELS = 20, 3
EXPECTED_LABELS = {0: 'normal_candidate', 1: 'operational_anomaly', 2: 'telemetry_manipulation', 3: 'historical_replay'}


def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_X(d, path):
    require('X' in d and d['X'].ndim == 3 and d['X'].shape[1:] == (SEQ_LEN, NCHANNELS), f'Bad X shape: {path}')
    require(d['X'].dtype == np.float32 and np.isfinite(d['X']).all(), f'Bad X dtype/values: {path}')
    return len(d['X'])


cfg = json.loads((R / 'phase9a1' / 'config.json').read_text(encoding='utf-8'))
require(cfg['channels'] == ['channel_14', 'channel_21', 'channel_29'], 'Unexpected channel order')
require(cfg['sequence_length'] == SEQ_LEN and cfg['grid_seconds'] == 90, 'Unexpected window/grid configuration')
bounds = {s: (pd.Timestamp(cfg['split_boundaries'][s][0]).value, pd.Timestamp(cfg['split_boundaries'][s][1]).value) for s in SPLITS}
phase9c_manifest = pd.read_csv(R / 'phase9c' / 'window_manifest.csv', dtype={'event_id': str})
require(not phase9c_manifest.duplicated(['split', 'event_id', 'start_ns', 'end_ns']).any(), 'Duplicate official anomaly manifest keys')

main = []
controls = []
summary = []
for split in SPLITS:
    print('Checking', split, flush=True)
    normal_file = f'phase9a1/{split}_normal.npz'
    attack_file = f'phase9b/{split}_attacks.npz'
    official_file = f'phase9c/{split}_official_anomalies.npz'
    n = load_npz(R / normal_file)
    a = load_npz(R / attack_file)
    o = load_npz(R / official_file)
    nn, na, no = check_X(n, normal_file), check_X(a, attack_file), check_X(o, official_file)
    require(len(n['start_ns']) == len(n['end_ns']) == nn, 'Normal timestamp length mismatch')
    require(len(a['y']) == len(a['target_start_ns']) == len(a['target_end_ns']) == na, 'Attack metadata length mismatch')
    require(len(o['start_ns']) == len(o['end_ns']) == len(o['event_id']) == no, 'Official metadata length mismatch')
    require(set(np.unique(a['y']).tolist()) == {0, 2, 3}, 'Unexpected attack class labels')
    require((n['start_ns'] < n['end_ns']).all() and (o['start_ns'] < o['end_ns']).all(), 'Invalid window timestamps')
    lo, hi = bounds[split]
    for name, starts, ends in [('normal', n['start_ns'], n['end_ns']), ('attack', a['target_start_ns'], a['target_end_ns']), ('official', o['start_ns'], o['end_ns'])]:
        require(((starts >= lo) & (ends <= hi) & (starts < ends)).all(), f'{split} {name}: outside split boundaries')
    require((n['end_ns'] - n['start_ns'] == 19 * 90 * 10**9).all(), 'Normal window spacing mismatch')
    require((a['target_end_ns'] - a['target_start_ns'] == 19 * 90 * 10**9).all(), 'Attack window spacing mismatch')
    require((o['end_ns'] - o['start_ns'] == 19 * 90 * 10**9).all(), 'Official window spacing mismatch')

    m = pd.read_csv(R / 'phase9b' / f'{split}_attack_manifest.csv', dtype={'sample_id': str, 'replay_source_start_ns': 'string', 'replay_source_end_ns': 'string', 'attack_start_ns': 'string', 'attack_end_ns': 'string'})
    require(len(m) == na, f'{split}: attack manifest length mismatch')
    require(m['sample_id'].notna().all() and m['sample_id'].is_unique, f'{split}: duplicate/empty attack IDs')
    require(np.array_equal(m['label'].to_numpy(dtype=np.int64), a['y']), f'{split}: attack label/order mismatch')
    targets = m['target_sequence_idx'].to_numpy(dtype=np.int64)
    require(((targets >= 0) & (targets < nn)).all(), f'{split}: invalid target indices')
    require(np.array_equal(n['start_ns'][targets], a['target_start_ns']) and np.array_equal(n['end_ns'][targets], a['target_end_ns']), f'{split}: attack target timestamps mismatch')
    require(np.array_equal(m['target_start_ns'].to_numpy(dtype=np.int64), a['target_start_ns']), f'{split}: attack manifest start mismatch')
    require(np.array_equal(m['target_end_ns'].to_numpy(dtype=np.int64), a['target_end_ns']), f'{split}: attack manifest end mismatch')
    control_indices = np.flatnonzero(a['y'] == 0)
    require(np.array_equal(a['X'][control_indices], n['X'][targets[control_indices]]), f'{split}: control source mismatch')
    replay_indices = np.flatnonzero(a['y'] == 3)
    require(m.loc[replay_indices, 'replay_source_idx'].notna().all(), f'{split}: missing replay sources')
    replay_src = m.loc[replay_indices, 'replay_source_idx'].to_numpy(dtype=np.int64)
    require(((replay_src >= 0) & (replay_src < nn)).all(), f'{split}: replay source index out of range')
    require(np.array_equal(m.loc[replay_indices, 'replay_source_start_ns'].astype('int64').to_numpy(), n['start_ns'][replay_src]), f'{split}: replay source start mismatch')
    require(np.array_equal(m.loc[replay_indices, 'replay_source_end_ns'].astype('int64').to_numpy(), n['start_ns'][replay_src] + 7*90*10**9), f'{split}: replay source segment end mismatch')
    require(np.array_equal(a['X'][replay_indices, 12:20, :], n['X'][replay_src, 0:8, :]), f'{split}: replay segment values mismatch')
    require((n['end_ns'][replay_src] <= a['target_start_ns'][replay_indices] - 24*3600*10**9).all(), f'{split}: replay source is not sufficiently historical')

    for i in range(nn):
        main.append((split, 0, EXPECTED_LABELS[0], normal_file, i, int(n['start_ns'][i]), int(n['end_ns'][i]), '', i, '', 'original_normal_candidate'))
    for i in range(na):
        label = int(a['y'][i])
        record = (split, label, EXPECTED_LABELS[label], attack_file, i, int(a['target_start_ns'][i]), int(a['target_end_ns'][i]), '', int(targets[i]), str(m.iloc[i]['sample_id']), 'derived_from_phase9a1_target')
        if label == 0:
            controls.append(record)
        else:
            main.append(record)
    sub = phase9c_manifest[phase9c_manifest['split'] == split].reset_index(drop=True)
    require(len(sub) == no, f'{split}: official manifest count mismatch')
    require(np.array_equal(sub['start_ns'].to_numpy(dtype=np.int64), o['start_ns']), f'{split}: official manifest start/order mismatch')
    require(np.array_equal(sub['end_ns'].to_numpy(dtype=np.int64), o['end_ns']), f'{split}: official manifest end/order mismatch')
    require(np.array_equal(sub['event_id'].astype(str).to_numpy(), o['event_id'].astype(str)), f'{split}: official event/order mismatch')
    require((sub['official_label'] == 'operational_anomaly').all(), f'{split}: official label mismatch')
    for i in range(no):
        main.append((split, 1, EXPECTED_LABELS[1], official_file, i, int(o['start_ns'][i]), int(o['end_ns'][i]), str(o['event_id'][i]), '', '', 'official_positive_only'))
    summary.append({'split': split, 'normal_candidates': nn, 'operational_anomalies': no, 'manipulation': int((a['y'] == 2).sum()), 'replay': int((a['y'] == 3).sum()), 'excluded_duplicate_controls': len(control_indices), 'official_unique_events': len(set(o['event_id'].tolist()))})

columns = ['split', 'label', 'class_name', 'source_npz', 'source_row', 'start_ns', 'end_ns', 'event_id', 'normal_target_row', 'attack_sample_id', 'lineage']
master = pd.DataFrame(main, columns=columns)
paired = pd.DataFrame(controls, columns=columns)
require(not master.duplicated(['split', 'source_npz', 'source_row']).any(), 'Duplicate source rows in master')
require(len(master) == 58000 + 162, f'Unexpected master size: {len(master)}')
require(len(paired) == 900, f'Unexpected paired-control count: {len(paired)}')
require(master.groupby('split')['label'].nunique().eq(4).all(), 'A split is missing a class')
require((master.loc[master['label'] == 1, 'event_id'].astype(str).str.len() > 0).all(), 'Official event IDs absent')

OUT.mkdir(parents=True, exist_ok=True)
master.to_csv(OUT / 'master_manifest.csv', index=False)
paired.to_csv(OUT / 'paired_normal_controls.csv', index=False)
pd.DataFrame(summary).to_csv(OUT / 'split_summary.csv', index=False)
(OUT / 'config.json').write_text(json.dumps({
    'phase': '9D', 'mode': 'manifest_only_no_npz_copies', 'classes': EXPECTED_LABELS,
    'channels': cfg['channels'], 'sequence_length': SEQ_LEN,
    'input_files_are_unchanged': True,
    'paired_controls': 'Separate file; duplicates of Phase 9A.1 normal candidates',
    'limitations': [
        'Normal candidates are not verified clean.',
        'Phase 9B attack variants share target windows with normal controls; preserve grouping in experiments.',
        'Operational anomalies are positive-only and event-anchored, not continuous timeline evaluation.',
        'This does not prove all model classes are distinguishable or establish benchmark latency comparability.',
        'No model training or test-set tuning has been performed.'
    ]
}, indent=2), encoding='utf-8')
print('\n=== PHASE 9D SUMMARY ===')
print(pd.DataFrame(summary).to_string(index=False))
print('Main manifest rows:', len(master))
print('Separate paired normal controls:', len(paired))
print('Output:', OUT)
print('PHASE_9D_BUILD_PASSED (independent audit pending)')
