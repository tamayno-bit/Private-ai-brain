"""Provenance-gated, immutable dataset snapshots. No proprietary-teacher harvesting."""
import json,hashlib
from pathlib import Path
from .tokenizer import ByteTokenizer

def validate_record(record):
    if not isinstance(record,dict):raise ValueError('Expected JSON object')
    if record.get('consent') is not True:raise ValueError('Explicit dataset consent required')
    if record.get('license') not in ('own','cc0','public-domain','cc-by-4.0','permitted-research'):
        raise ValueError('Dataset license must be documented and allowed')
    if not isinstance(record.get('source'),str) or not record['source'].strip():raise ValueError('Document source')
    if not isinstance(record.get('text'),str) or not record['text'].strip():raise ValueError('Nonempty text required')
    if len(record['text'].encode('utf-8'))>1024*1024:raise ValueError('Example too large')
    return record

def load_records(path):
    content=Path(path).read_bytes();rows=[]
    for n,line in enumerate(content.splitlines(),1):
        if not line.strip():continue
        try:rows.append(validate_record(json.loads(line)))
        except Exception as exc:raise ValueError(f'Invalid dataset row {n}: {exc}') from exc
    if not rows:raise ValueError('Dataset has no records')
    return rows,hashlib.sha256(content).hexdigest()

def token_stream(rows):
    tok=ByteTokenizer();sequence=[]
    for r in rows:sequence.extend(tok.encode(r['text'],bos=True,eos=True))
    return sequence

def split_stream(ids,context):
    # Deterministic sequence split; holdout does not share token windows with training.
    pivot=int(len(ids)*0.9)
    if pivot<context+2 or len(ids)-pivot<context+2:
        raise ValueError('Dataset too small for a disjoint validation split at current context')
    return ids[:pivot],ids[pivot:]
