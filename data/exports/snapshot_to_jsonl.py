import collections
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
outdir = snapshot.with_suffix('')
outdir.mkdir(exist_ok=False)
db = sqlite3.connect(snapshot)
db.row_factory = sqlite3.Row
sample = dict(db.execute('SELECT * FROM corpus_sample_sets').fetchone())
if isinstance(sample.get('stats'), str):
    sample['stats'] = json.loads(sample['stats'])
repos = {r['id']:dict(r) for r in db.execute('SELECT * FROM repositories')}
annotations = {}
for row in db.execute('SELECT * FROM llm_annotations'):
    a = dict(row)
    if a['parsed_result'] is not None:
        a['parsed_result'] = json.loads(a['parsed_result'])
    assert a['corpus_id'] not in annotations
    annotations[a['corpus_id']] = a
items = {r['corpus_id']:dict(r) for r in db.execute('SELECT * FROM corpus_sample_items')}
counts = collections.Counter()
distribution = collections.Counter()
nonempty = 0
with gzip.open(outdir/'annotations.jsonl.gz','wt',encoding='utf-8') as f:
    for row in db.execute('SELECT * FROM corpus ORDER BY id'):
        corpus = dict(row)
        cid = corpus['id']
        item = items[cid]
        a = annotations.get(cid)
        status = a['status'] if a else 'not_attempted'
        counts[status] += 1
        labels = a['parsed_result']['annotations'] if a and status=='succeeded' else None
        if labels:
            nonempty += 1
            for label in labels:
                distribution[(label['aspect'],label['class'])] += 1
        record = {'corpus_id':cid,'sample_name':sample['name'],
                  'repository':repos[item['repository_id']]['full_name'],
                  'status':status,'annotations':labels,'corpus':corpus,
                  'sample_item':item,'annotation_record':a}
        f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
with gzip.open(outdir/'lexicon_hits.jsonl.gz','wt',encoding='utf-8') as f:
    for row in db.execute('SELECT * FROM lexicon_sample_hits'):
        r=dict(row)
        if isinstance(r.get('evidence'),str):r['evidence']=json.loads(r['evidence'])
        f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
summary={'sample':sample,'total_corpus':sum(counts.values()),'status_counts':dict(counts),
         'nonempty_succeeded':nonempty,'empty_succeeded':counts['succeeded']-nonempty,
         'total_labels':sum(distribution.values()),
         'aspect_sentiment':[{'aspect':a,'sentiment':s,'count':n} for (a,s),n in sorted(distribution.items())],
         'scope':'Cumulative sample results for rust-aspects-v2 / aspect-sentiment-zh-v8 / glm-5.3-flash; not exclusive to one run',
         'snapshot':json.loads(Path(str(snapshot)+'.json').read_text(encoding='utf-8'))}
(outdir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[]
for p in sorted(outdir.iterdir()):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    manifest.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(outdir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'directory':str(outdir),'counts':dict(counts),'total_labels':sum(distribution.values()),'files':manifest}),flush=True)
