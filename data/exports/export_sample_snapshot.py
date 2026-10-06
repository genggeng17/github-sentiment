import datetime
import decimal
import hashlib
import json
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, text
from config import Settings

sample = 'rust-all-aspects-v2-10k'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
path = Path('data/exports') / ('labeling-' + stamp + '.sqlite3')
path.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(Settings.from_env().database_url)
counts = {}

def convert(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return str(v)
    return v

with engine.connect().execution_options(isolation_level='REPEATABLE READ') as src:
    src.exec_driver_sql('START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY')
    sid = src.execute(text('SELECT id FROM corpus_sample_sets WHERE name=:name'), {'name':sample}).scalar_one()
    queries = {
        'corpus_sample_sets':'SELECT * FROM corpus_sample_sets WHERE id=:sid',
        'corpus_sample_items':'SELECT * FROM corpus_sample_items WHERE sample_set_id=:sid',
        'repositories':'SELECT * FROM repositories WHERE id IN (SELECT repository_id FROM corpus_sample_items WHERE sample_set_id=:sid)',
        'corpus':'SELECT c.* FROM corpus c JOIN corpus_sample_items i ON i.corpus_id=c.id WHERE i.sample_set_id=:sid',
        'llm_annotations':"SELECT a.* FROM llm_annotations a JOIN corpus_sample_items i ON i.corpus_id=a.corpus_id WHERE i.sample_set_id=:sid AND a.taxonomy_version='rust-aspects-v2' AND a.prompt_version='aspect-sentiment-zh-v8' AND a.model_name='glm-5.3-flash'",
        'lexicon_sample_hits':'SELECT * FROM lexicon_sample_hits WHERE sample_set_id=:sid',
        'pipeline_runs':"SELECT * FROM pipeline_runs WHERE id='31180520-e94c-4b06-99b8-51f642d8b23f'",
    }
    with sqlite3.connect(path) as dest:
        for table, sql in queries.items():
            rows = src.execution_options(stream_results=True).execute(text(sql), {'sid':sid})
            columns = list(rows.keys())
            # Analysis snapshot: preserve original values and IDs, not MySQL constraints.
            dest.execute('CREATE TABLE "'+table+'" ('+','.join('"'+c+'"' for c in columns)+')')
            insert = 'INSERT INTO "'+table+'" VALUES ('+','.join('?' for _ in columns)+')'
            count = 0
            for batch in rows.partitions(500):
                dest.executemany(insert, [tuple(convert(v) for v in row) for row in batch])
                count += len(batch)
            rows.close()
            counts[table] = count
            print(table, count, flush=True)
        dest.execute('CREATE INDEX annotation_corpus ON llm_annotations(corpus_id)')
        dest.execute('CREATE INDEX corpus_id_index ON corpus(id)')
        dest.execute('CREATE INDEX sample_corpus ON corpus_sample_items(corpus_id)')
        dest.execute('CREATE TABLE snapshot_metadata (key TEXT PRIMARY KEY,value TEXT)')
        metadata = {'sample':sample,'snapshot_utc':stamp,'scope':'sample cumulative annotations, prompt v8, taxonomy v2, glm-5.3-flash; not exclusive to a single run','counts':counts}
        dest.executemany('INSERT INTO snapshot_metadata VALUES (?,?)', [(k,json.dumps(v,ensure_ascii=False)) for k,v in metadata.items()])
    src.rollback()
sha = hashlib.sha256()
with path.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024), b''):
        sha.update(block)
manifest = {'path':str(path),'bytes':path.stat().st_size,'sha256':sha.hexdigest(),'counts':counts,'sample':sample,'snapshot_utc':stamp}
Path(str(path)+'.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('MANIFEST '+json.dumps(manifest),flush=True)
