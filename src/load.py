"""Strict loading of the supplied, fixed NSL-KDD split."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = 'duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell su_attempted num_root num_file_creations num_shells num_access_files num_outbound_cmds is_host_login is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate'.split()
CAT = ['protocol_type', 'service', 'flag']
FAMILIES = {'normal': 'normal', 'DoS': 'back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb', 'Probe': 'satan ipsweep nmap portsweep mscan saint', 'R2L': 'guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named', 'U2R': 'buffer_overflow loadmodule rootkit perl sqlattack xterm ps'}
FAMILY_MAP = {name: family for family, names in FAMILIES.items() for name in names.split()}
EXPECTED = [dict(normal=67343, DoS=45927, Probe=11656, R2L=995, U2R=52), dict(normal=9711, DoS=7460, Probe=2421, R2L=2885, U2R=67)]
HASHES = ['0b732256db3e34094774988f8f902966db7fde73d4984c002d6726d201f2523c', 'fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84']

def load_data():
    frames = []
    for i, name in enumerate(['KDDTrain+.txt', 'KDDTest+.txt']):
        path = ROOT / 'data' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[i], f'File integrity changed: {path}'
        with path.open() as f:
            first = f.readline()
        frame = pd.read_csv(path, sep='\t' if '\t' in first else ',', header=None)
        assert frame.shape == ([125973, 22544][i], 43), f'{name}: actual shape {frame.shape}'
        frame.columns = FEATURES + ['label', 'difficulty']
        for col in CAT + ['label']:
            frame[col] = frame[col].str.strip()
        assert not frame.astype(str).apply(lambda s: s.str.lower().eq('attack4').any()).any(), 'Corrupted Attack4 copy'
        frame['label'] = frame.label.str.lower()
        unknown = set(frame.label) - FAMILY_MAP.keys()
        assert not unknown, f'Unmapped labels: {sorted(unknown)}'
        frame['family'] = frame.label.map(FAMILY_MAP)
        assert frame.family.value_counts().to_dict() == EXPECTED[i]
        numeric = frame[[c for c in FEATURES if c not in CAT] + ['difficulty']].to_numpy(float)
        assert np.isfinite(numeric).all() and (numeric >= 0).all()
        assert frame.difficulty.between(0, 21).all() and (frame.difficulty % 1 == 0).all()
        frame['y'] = frame.label.ne('normal').astype(int)
        frames.append(frame)
    train, test = frames
    assert train.label.nunique() == 23  # 22 attack names plus normal
    assert test.service.eq('imap4').sum() == 306
    test['seen'] = test.label.isin(train.label.unique()).astype(int)
    assert test.seen.eq(0).sum() == 3750 and test.loc[test.seen.eq(0), 'label'].nunique() == 17
    expected_unseen=dict(mscan=996,apache2=737,processtable=685,snmpguess=331,saint=319,mailbomb=293,snmpgetattack=178,httptunnel=133,named=17,ps=15,sendmail=14,xterm=13,xlock=9,xsnoop=4,worm=2,sqlattack=2,udpstorm=2)
    assert test.loc[test.seen.eq(0),'label'].value_counts().to_dict() == expected_unseen
    return train, test
