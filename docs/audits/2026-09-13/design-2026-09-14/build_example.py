"""Build a local illustrative retrieval record; no database or model calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
row = next(
    value for line in (HERE.parent / 'cleaning-random-1500.jsonl').read_text(encoding='utf-8').splitlines()
    if (value := json.loads(line))['id'] == 672649
)
field = row['clean_text']
rules = [
    {'rule_id': 'ownership.topic.lifetime', 'pattern': r'\blifetimes?\b', 'role': 'candidate'},
    {'rule_id': 'ownership.experience.hard_time', 'pattern': r'\bhard time\b', 'role': 'ranking_only_requires_topic'},
]
evidence = []
for rule in rules:
    for match in re.finditer(rule['pattern'], field, re.IGNORECASE):
        evidence.append({'rule_id': rule['rule_id'], 'field': 'clean_text',
                         'region': 'target_prose', 'match': match.group(),
                         'start': match.start(), 'end': match.end()})
assert any(e['rule_id'] == rules[0]['rule_id'] for e in evidence)
assert all(field[e['start']:e['end']] == e['match'] for e in evidence)
example = {
    'example_only': True,
    'note': '本地真实文本的存储示例，未执行数据库检索；run_id为演示值。原导出不含corpus.content_hash，因此未填，真实入库必须从corpus读取。规则仅演示ownership，非完整词典。',
    'rules_example': rules,
    'source_snapshot': {'corpus_id': row['id'], 'source_type': row['source_type'],
                        'source_id': row['source_id'], 'cleaning_version': row['cleaning_version'],
                        'clean_text': field},
    'retrieval_hit': {
        'run_id': 'EXAMPLE_ONLY', 'corpus_id': row['id'], 'candidate_aspect': 'ownership',
        'corpus_content_hash': None,
        'field_hashes_json': {'clean_text': hashlib.sha256(field.encode('utf-8')).hexdigest()},
        'matched_rule_ids_json': sorted({e['rule_id'] for e in evidence}),
        'evidence_json': evidence, 'rank_score': None,
    },
    'span_convention': 'Python Unicode character offsets; [start, end)',
    'label_status': '词典候选不等于标签；实际标注继续由llm_annotations保存。',
}
(HERE / 'retrieval-hit-example.json').write_text(json.dumps(example, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
spec = importlib.util.spec_from_file_location('validation', ROOT / 'llm_labeler/validation.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
outputs = [line.removeprefix('输出：') for line in (HERE / 'prompt-v8-candidate.txt').read_text(encoding='utf-8').splitlines() if line.startswith('输出：')]
parsed = [validation.validate_annotation(output) for output in outputs]
covered = {ann['aspect'] for output in parsed for ann in output['annotations']}
assert len(outputs) == 19
assert covered == validation.ASPECTS
print(json.dumps({'valid_prompt_examples': len(outputs), 'aspects_covered': len(covered), 'verified_match_spans': len(evidence), 'model_evaluation_performed': False}, ensure_ascii=False))
