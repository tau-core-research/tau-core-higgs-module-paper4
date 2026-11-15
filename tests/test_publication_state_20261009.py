"""Publication dependency guard, not Nature validation."""
import hashlib
import json
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[1]
def test_current_state_scope():
    p = json.loads((ROOT/'data/derived/publication_state_20261009.json').read_text())
    assert p['empirical_scores_changed'] is False
    assert p['nature_occupation_proved'] is False
    assert p['tau_specific_evidence'] is False
    assert p['mass_candidate_retained'] is True
    for item in p['manuscripts']:
        text = (ROOT/item['source']).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == item['source_sha256']
        parts = re.findall(r'% BEGIN SOURCE INSTRUMENT RECONCILIATION 20261009(.*?)% END SOURCE INSTRUMENT RECONCILIATION 20261009',text,re.S)
        assert len(parts) == 1
        if item['role'] == 'OIR1-owner':
            assert 'label{prop:oir1}' in parts[0] and 'OIR1(a)' in parts[0]
            assert 'OIR1(b)' in parts[0] and 'OIR1(c)' in parts[0]
            assert 'hypothetical' in parts[0] and 'unresolved' in parts[0]
        elif item['role'] == 'MSI1-owner':
            assert 'label{prop:msi1}' in parts[0] and ('N_{' + chr(92) + 'min}') in parts[0]
            assert '192' in parts[0] and 'zero joint hits' in parts[0]
        else:
            assert 'nameddest=oir1' in parts[0] and 'nameddest=msi1' in parts[0]
            assert len(parts[0].split()) < 210
            assert 'section{' not in parts[0] and 'begin{proof}' not in parts[0]
