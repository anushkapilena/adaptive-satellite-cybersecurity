"""Independent, read-only Phase 9D.1 audit. Run from the project root.
Does not import the Phase 9D builder or write/modify any project files.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / 'results'
SPLITS = ('train', 'validation', 'test')
NAMES = {0:'normal_candidate', 1:'operational_anomaly', 2:'telemetry_manipulation', 3:'historical_replay'}
FIELDS = ['split','label','class_name','source_npz','source_row','start_ns','end_ns','event_id','normal_target_row','attack_sample_id','lineage']
NS = 1_000_000_000

def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)

def read_index(path):
    # Read timestamp columns as exact decimal strings, not float.
    df = pd.read_csv(path, dtype={'start_ns':'string','end_ns':'string','event_id':'string','attack_sample_id':'string','normal_target_row':'string','source_npz':'string','class_name':'string','lineage':'string','split':'string'})
    check(list(df.columns) == FIELDS, f'{path.name}: unexpected columns')
    for c in ('start_ns','end_ns'):
        check(df[c].notna().all(), f'{path.name}: missing {c}')
        df[c] = df[c].map(int).astype('int64')
    df['label'] = pd.to_numeric(df['label'], errors='raise').astype('int64')
    df['source_row'] = pd.to_numeric(df['source_row'], errors='raise').astype('int64')
    return df

def array(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

master = read_index(ROOT / 'phase9d/master_manifest.csv')
paired = read_index(ROOT / 'phase9d/paired_normal_controls.csv')
summary = pd.read_csv(ROOT / 'phase9d/split_summary.csv').set_index('split')
config = json.loads((ROOT / 'phase9d/config.json').read_text(encoding='utf-8'))
base_config = json.loads((ROOT / 'phase9a1/config.json').read_text(encoding='utf-8'))
check(len(master) == 58162 and len(paired) == 900, 'Unexpected overall record counts')
check(not master.duplicated(['split','source_npz','source_row']).any(), 'Repeated master source reference')
check(not paired.duplicated(['split','source_npz','source_row']).any(), 'Repeated control source reference')
check(not master[['split','source_npz','source_row']].merge(paired[['split','source_npz','source_row']],on=['split','source_npz','source_row']).shape[0], 'Main and control references overlap')
check(config['mode'] == 'manifest_only_no_npz_copies', 'Unexpected integration mode')
check(config['channels'] == ['channel_14','channel_21','channel_29'], 'Channel order mismatch')
check(set(master['split']) == set(SPLITS), 'Unexpected splits')
check(set(master['label']) == set(NAMES), 'Unexpected classes')
check((master['class_name'] == master['label'].map(NAMES)).all(), 'Class name mismatch')
check((paired['label'] == 0).all(), 'Paired controls are not all class 0')
check((master['end_ns']-master['start_ns'] == 19*90*NS).all(), 'Master window length mismatch')
check((paired['end_ns']-paired['start_ns'] == 19*90*NS).all(), 'Paired window length mismatch')

for split in SPLITS:
    print(f'Auditing {split}...', flush=True)
    mm = master.loc[master['split']==split]
    pp = paired.loc[paired['split']==split]
    n = array(ROOT / f'phase9a1/{split}_normal.npz')
    a = array(ROOT / f'phase9b/{split}_attacks.npz')
    o = array(ROOT / f'phase9c/{split}_official_anomalies.npz')
    raw_manifest = pd.read_csv(ROOT / f'phase9b/{split}_attack_manifest.csv', dtype={'sample_id':'string','replay_source_start_ns':'string','replay_source_end_ns':'string','target_start_ns':'string','target_end_ns':'string'})
    official_manifest = pd.read_csv(ROOT / 'phase9c/window_manifest.csv', dtype={'event_id':'string','start_ns':'string','end_ns':'string'})
    official_manifest = official_manifest.loc[official_manifest['split']==split]
    lo,hi = (pd.Timestamp(x).value for x in base_config['split_boundaries'][split])
    check(((mm['start_ns'] >= lo)&(mm['end_ns'] <= hi)).all(), f'{split}: main outside split')
    check(((pp['start_ns'] >= lo)&(pp['end_ns'] <= hi)).all(), f'{split}: controls outside split')
    check(all(x['X'].ndim==3 and x['X'].shape[1:]==(20,3) and np.isfinite(x['X']).all() for x in (n,a,o)),f'{split}: source array invalid')
    check(len(raw_manifest)==len(a['X']), f'{split}: attack metadata length')
    check(np.array_equal(raw_manifest['label'].to_numpy(dtype='int64'),a['y']), f'{split}: attack label alignment')
    targets = raw_manifest['target_sequence_idx'].to_numpy(dtype='int64')
    check(((targets>=0)&(targets<len(n['X']))).all(),f'{split}: invalid target indices')
    check(np.array_equal(a['target_start_ns'],n['start_ns'][targets]) and np.array_equal(a['target_end_ns'],n['end_ns'][targets]),f'{split}: attack target lineage')
    check(np.array_equal(raw_manifest['target_start_ns'].map(int).to_numpy(dtype='int64'),a['target_start_ns']),f'{split}: attack manifest start')
    check(np.array_equal(raw_manifest['target_end_ns'].map(int).to_numpy(dtype='int64'),a['target_end_ns']),f'{split}: attack manifest end')
    check(np.array_equal(a['X'][:,:12,:],n['X'][targets,:12,:]),f'{split}: synthetic prefix changed')
    for label,src,data,starts,ends in [
        (0,f'phase9a1/{split}_normal.npz',n,n['start_ns'],n['end_ns']),
        (1,f'phase9c/{split}_official_anomalies.npz',o,o['start_ns'],o['end_ns']),
        (2,f'phase9b/{split}_attacks.npz',a,a['target_start_ns'],a['target_end_ns']),
        (3,f'phase9b/{split}_attacks.npz',a,a['target_start_ns'],a['target_end_ns'])]:
        rows=mm.loc[mm['label']==label]
        ix=rows['source_row'].to_numpy(dtype='int64')
        check((rows['source_npz']==src).all(),f'{split}: class {label} source path')
        check(((ix>=0)&(ix<len(data['X']))).all(),f'{split}: class {label} invalid source row')
        check(np.array_equal(rows['start_ns'].to_numpy(dtype='int64'),starts[ix]),f'{split}: class {label} start mismatch')
        check(np.array_equal(rows['end_ns'].to_numpy(dtype='int64'),ends[ix]),f'{split}: class {label} end mismatch')
        if label in (2,3):
            check((a['y'][ix]==label).all(),f'{split}: class {label} source label mismatch')
            check(np.array_equal(rows['normal_target_row'].map(int).to_numpy(dtype='int64'),targets[ix]),f'{split}: class {label} target lineage mismatch')
            check((rows['attack_sample_id'].to_numpy(dtype=str)==raw_manifest['sample_id'].iloc[ix].to_numpy(dtype=str)).all(),f'{split}: class {label} attack ID mismatch')
        if label==1:
            check((rows['event_id'].to_numpy(dtype=str)==o['event_id'][ix].astype(str)).all(),f'{split}: official event ID mismatch')
        if label==0:
            check(np.array_equal(np.sort(ix),np.arange(len(n['X']))),f'{split}: normal coverage incomplete')
    check(len(official_manifest)==len(o['X']),f'{split}: official manifest length')
    check(np.array_equal(official_manifest['start_ns'].map(int).to_numpy(dtype='int64'),o['start_ns']),f'{split}: official start order')
    check(np.array_equal(official_manifest['end_ns'].map(int).to_numpy(dtype='int64'),o['end_ns']),f'{split}: official end order')
    check((official_manifest['event_id'].to_numpy(dtype=str)==o['event_id'].astype(str)).all(),f'{split}: official event order')
    control_rows=np.flatnonzero(a['y']==0)
    check(len(pp)==len(control_rows)==300,f'{split}: control count')
    check(np.array_equal(np.sort(pp['source_row'].to_numpy(dtype='int64')),control_rows),f'{split}: control source rows')
    pi=pp['source_row'].to_numpy(dtype='int64')
    check((pp['source_npz']==f'phase9b/{split}_attacks.npz').all(),f'{split}: control source path')
    check(np.array_equal(a['X'][pi],n['X'][targets[pi]]),f'{split}: controls differ from originals')
    check(np.array_equal(pp['start_ns'].to_numpy(dtype='int64'),a['target_start_ns'][pi]),f'{split}: control start mismatch')
    check(np.array_equal(pp['end_ns'].to_numpy(dtype='int64'),a['target_end_ns'][pi]),f'{split}: control end mismatch')
    check(np.array_equal(pp['normal_target_row'].map(int).to_numpy(dtype='int64'),targets[pi]),f'{split}: control target indices')
    check((pp['attack_sample_id'].to_numpy(dtype=str)==raw_manifest['sample_id'].iloc[pi].to_numpy(dtype=str)).all(),f'{split}: control attack IDs')
    replay=np.flatnonzero(a['y']==3)
    source=raw_manifest['replay_source_idx'].iloc[replay].to_numpy(dtype='int64')
    check(((source>=0)&(source<len(n['X']))).all(),f'{split}: replay source bounds')
    check(np.array_equal(a['X'][replay,12:,:],n['X'][source,:8,:]),f'{split}: replay segment differs')
    check(np.array_equal(raw_manifest['replay_source_start_ns'].iloc[replay].map(int).to_numpy(dtype='int64'),n['start_ns'][source]),f'{split}: replay source start')
    check(np.array_equal(raw_manifest['replay_source_end_ns'].iloc[replay].map(int).to_numpy(dtype='int64'),n['start_ns'][source]+7*90*NS),f'{split}: replay source segment end')
    check((n['end_ns'][source]<=a['target_start_ns'][replay]-24*3600*NS).all(),f'{split}: replay historical separation')
    counts=mm['label'].value_counts().to_dict()
    for label,expected in [(0,len(n['X'])),(1,len(o['X'])),(2,300),(3,300)]:
        check(counts.get(label,0)==expected,f'{split}: class {label} count')
    for col,expected in [('normal_candidates',len(n['X'])),('operational_anomalies',len(o['X'])),('manipulation',300),('replay',300),('excluded_duplicate_controls',300),('official_unique_events',len(np.unique(o['event_id'])))]:
        check(int(summary.loc[split,col])==expected,f'{split}: summary {col} mismatch')
    print(f'  PASSED: {len(mm):,} master rows; {len(pp)} paired controls; {len(np.unique(o["event_id"]))} official events')

check(len(master)==sum(len(master.loc[master['split']==s]) for s in SPLITS),'Split totals mismatch')
print('\nPHASE_9D1_INDEPENDENT_AUDIT_PASSED')
print('Note: this validates manifest/source consistency, not label truth, attack realism, or event-level detection latency.')
