import ast
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent
INPUT = Path(r'C:\Users\20803\AppData\Local\Temp\codex-rust-db-audit')
REPO = ROOT.parents[2]

def read(name):
    return json.loads((INPUT / name).read_text(encoding='utf-8'))

def write_jsonl(name, rows):
    with (ROOT / name).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

meta = read('metadata.json')
corpus = {r['id']: r for r in read('corpus_sample.json')}
selection = read('sample_selection.json')
old = {r['id']: r for r in meta if r['version'].endswith('v4')}
new = {r['id']: r for r in meta if r['version'].endswith('v7')}
overlap = read('overlap.json')
sources = {r['id']: r for r in read('source_metadata.json')}
actual_v2 = {r['v1_id']: r for r in read('actual_clean_v2.json')}
historical = {r['corpus_id']: r for r in map(json.loads, (REPO / 'data/exports/rust-v1-5000.jsonl').open(encoding='utf-8'))}
verified = [json.loads(s) for s in read('result-020.json')['stdout'].splitlines()[1:] if s]
for r in verified:
    sources[r['id']] = {'id': r['id'], 'author': r['author'], 'url': r['url']}

def labels(r):
    return r['labels']['annotations']

def stats(rows):
    rows = list(rows)
    n = len(rows)
    ne = sum(bool(labels(r)) for r in rows)
    polar = sum(any(a['class'] != 'neutral' for a in labels(r)) for r in rows)
    return {'n': n, 'nonempty': ne, 'rate': ne / n, 'polar': polar, 'neutral_only': ne - polar,
            'aspects': dict(collections.Counter(a['aspect'] for r in rows for a in labels(r))),
            'classes': dict(collections.Counter(a['class'] for r in rows for a in labels(r)))}

summary = {'date': '2026-09-13', 'v4': stats(old.values()), 'v7': stats(new.values()),
           'overlap_v4': stats(old[i] for i in overlap['ids']),
           'overlap_v7': stats(new[i] for i in overlap['ids']),
           'transitions': {k: len(v) for k, v in overlap['transitions'].items()},
           'historical_input_matches': sum(corpus[i]['model_input'] == historical[i]['model_input'] for i in overlap['ids'])}
summary['by_source'] = {s: {'v4': stats(r for r in old.values() if r['source_type'] == s),
                           'v7': stats(r for r in new.values() if r['source_type'] == s)}
                        for s in sorted({r['source_type'] for r in meta})}
weighted = {}
for fields in [('source_type',), ('repo_id',), ('repo_id', 'source_type')]:
    group = {}
    for name, rows in [('old', old.values()), ('new', new.values())]:
        counts = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            key = tuple(r[f] for f in fields)
            counts[key][0] += 1
            counts[key][1] += bool(labels(r))
        group[name] = counts
    assert all(group['new'][k][0] for k in group['old'])
    weighted['+'.join(fields)] = sum(n * group['new'][k][1] / group['new'][k][0]
                                     for k, (n, _) in group['old'].items()) / len(old)
summary['v7_standardized_to_v4'] = weighted

random_rows = [corpus[i] for i in selection['random_v7_ids']]
bot_names = {'bors', 'homu', 'rust-highfive', 'rustbot', 'bors-servo', 'highfive'}
bot_ids = [i for i in selection['random_v7_ids'] if sources[i]['author'] and
           ('[bot]' in sources[i]['author'] or sources[i]['author'].lower() in bot_names)]
flags = {
    'fenced_block': lambda t: bool(re.search(r'(?m)^\s*(```|~~~)', t)),
    'quote': lambda t: bool(re.search(r'(?m)^\s*>', t)),
    'under_40_chars': lambda t: len(t.strip()) < 40,
    'checkbox': lambda t: bool(re.search(r'(?m)^\s*[-*]\s+\[[ xX]\]', t)),
    'html_details': lambda t: '<details' in t.lower() or '<summary' in t.lower(),
}
flag_counts = {k: sum(fn(r['clean_text']) for r in random_rows) for k, fn in flags.items()}
placeholder_only = [i for i in selection['random_v7_ids'] if not re.sub(
    r'\[(?:CODE_BLOCK_REMOVED|STACK_TRACE_REMOVED|TECHNICAL_OUTPUT_REMOVED): \d+ lines\]', '', actual_v2[i]['v2_clean']).strip()]
summary['cleaning_audit'] = {'seed': 20260913, 'random_n': 1500, 'random_nonempty': 85,
    'flags': flag_counts, 'bot_candidate_n': len(bot_ids),
    'bot_candidate_nonempty': sum(bool(labels(new[i])) for i in bot_ids),
    'actual_v2_matches': len(actual_v2), 'same_raw': sum(r['same_raw'] for r in actual_v2.values()),
    'changed_by_v2': sum(corpus[i]['clean_text'] != actual_v2[i]['v2_clean'] for i in actual_v2),
    'placeholder_only_ids': placeholder_only}

case_notes = {
 672977: ('明确漏标', 'api_extensibility / positive', '作者明确说策略易用、默认值好，并解释如何允许自定义实现；符合 v7 对 API 易用性和扩展能力的定义。'),
 541166: ('明确漏标', 'readability_maintainability / negative', '作者明确说现有代码复杂，建议去掉过时兼容代码并改成更直观的实现。'),
 1291197: ('明确漏标', 'package_manager / negative', '直接讨论 default-features=false 引起的功能退化，并说明自己此前被这类变更坑过。'),
 672649: ('明确漏标；仅有 v7', 'ownership / negative', '作者明确描述与生命周期斗争后放弃，属于相关使用体验，不是仅在代码中出现 lifetime。'),
 503943: ('漏标候选', 'diagnostics_debugging / negative 或 neutral，极性待裁决', '作者描述回溯缺少源错误相关帧的诊断体验；即使对负面极性有争议，也应审查为何整个方面为空。'),
 1141879: ('neutral 漏标候选', 'package_manager / neutral', '作者实质讨论默认 feature 与破坏性变更，不应因缺少情绪词就判为未涉及方面。'),
 1180314: ('neutral 漏标候选', 'api_extensibility / neutral；runtime_performance / neutral', '同时询问接口拆分或开关的设计，以及运行性能影响是否可忽略，属于有实质内容的问题。'),
 807718: ('方面漏标；极性边界', '至少 runtime_performance；极性另行裁决', '正文直接讨论性能回归和 nightly 改善，不能判完全未涉及方面；旧版 API 标签及整体极性不直接当作标准答案。'),
 504745: ('边界案例；仅有 v7', 'community / neutral 候选', '提出项目加入组织与社区发展的建议；用来检验 community 是否错误要求一定出现评价词。'),
 549493: ('漏标候选；仅有 v7', 'tooling_documentation / negative 候选', '作者反复遇到已发布指南中的运行示例不可用，涉及文档可用性。'),
 618144: ('合理纠偏', '空数组', '只有运行 rustfmt 的修改标题，缺少对工具质量或使用体验的讨论。'),
 563051: ('合理纠偏', '空数组', '只有一次插入问题的标题；不足以推断库生态质量或成熟度。'),
 552664: ('合理纠偏', '空数组', '编译器崩溃与日志本身不等同于对诊断质量或调试体验的评价。'),
 511610: ('合理纠偏候选', '空数组；不支持旧 type_system / negative', 'Miri 运行失败及技术原因猜测，不足以推断对 Rust 类型系统的负面态度。'),
}
case_rows = []
for i, (kind, suggestion, reason) in case_notes.items():
    case_rows.append({'corpus_id': i, 'review_kind': kind, 'suggested_annotation': suggestion,
                     'reason': reason, 'v4': labels(old[i]) if i in old else None,
                     'v7': labels(new[i]), **sources.get(i, {}),
                     'model_input': corpus[i]['model_input']})

write_jsonl('paired-1158.jsonl', [{'corpus_id': i, 'v4': labels(old[i]), 'v7': labels(new[i]),
    'source_type': corpus[i]['source_type'], 'repository_id': old[i]['repo_id'],
    'historical_input_matches': corpus[i]['model_input'] == historical[i]['model_input'],
    'model_input': corpus[i]['model_input']} for i in overlap['ids']])
write_jsonl('review-cases.jsonl', case_rows)
write_jsonl('cleaning-random-1500.jsonl', [{**corpus[i], **sources[i],
    'v7_annotations': labels(new[i]), 'bot_account_candidate': i in bot_ids,
    **actual_v2[i]} for i in selection['random_v7_ids']])
write_jsonl('annotations-metadata.jsonl', meta)
(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'sample-selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ['# 标注变化案例复核', '', '以下是本次助手审阅意见，用于人工裁决和回归测试，不是双人盲标金标准。空标签案例核对了数据库结果；重点案例另外检查了原始响应。', '']
for r in case_rows:
    lines += [f"## {r['corpus_id']}：{r['review_kind']}", '', f"来源：[GitHub 原文]({r.get('url', '')})", '',
              f"旧版：`{json.dumps(r['v4'], ensure_ascii=False)}`；新版：`{json.dumps(r['v7'], ensure_ascii=False)}`", '',
              f"建议：{r['suggested_annotation']}。{r['reason']}", '', '原始模型输入：', '', '````text', r['model_input'], '````', '']
(ROOT / '案例复核.md').write_text('\n'.join(lines), encoding='utf-8')

# Run the actual cleaner implementation without importing database dependencies.
tree = ast.parse((REPO / 'corpus_builder.py').read_text(encoding='utf-8'))
nodes = []
for n in tree.body:
    if isinstance(n, ast.Import) and all(a.name in ('re', 'unicodedata') for a in n.names): nodes.append(n)
    elif isinstance(n, ast.Assign): nodes.append(n)
    elif isinstance(n, ast.FunctionDef) and n.name != 'make_corpus_row': nodes.append(n)
ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'corpus_builder.py', 'exec'), ns)
synthetic = {
 '代码框中的自然语言': 'The result is:\n```text\nThe new API is much harder to use.\n```',
 '堆栈后没有空行的体验描述': 'stack backtrace:\n   0: fail\nThis crashes every day and blocks my work.\n',
 '短诊断': 'The compiler suggests:\n```\nhelp: add an explicit lifetime\n```',
 '基准对比': 'Benchmark results:\n```text\nbefore: 20 ms\nafter: 400 ms\n```',
 '误匹配技术日志的连续自然语言': 'Cargo is much slower now.\nRunning this blocks the editor.\nChecking the project takes ten minutes.',
}
cleaner_tests = [{'name': k, 'input': v, 'actual_output': ns['clean_text'](v)} for k, v in synthetic.items()]
write_jsonl('cleaner-behavior-tests.jsonl', cleaner_tests)
assert all(ns['clean_text'](corpus[i]['target_text']) == actual_v2[i]['v2_clean'] for i in actual_v2)

blind_ids = random.Random(20260913).sample(sorted(selection['random_v7_ids']), 500)
write_jsonl('blind-review-500.jsonl', [{'corpus_id': i, 'model_input': corpus[i]['model_input'],
    'human_annotations': None, 'evidence': None, 'uncertain_reason': None} for i in blind_ids])

source_table = '\n'.join(f"| {s} | {d['v4']['n']:,} | {d['v4']['rate']:.2%} | {d['v7']['n']:,} | {d['v7']['rate']:.2%} |" for s, d in summary['by_source'].items())
aspect_table = '\n'.join(f"| {a} | {summary['overlap_v4']['aspects'].get(a, 0)} | {summary['overlap_v7']['aspects'].get(a, 0)} |" for a in sorted(summary['v4']['aspects']))
report = f'''# GitHub Rust 语料与提示词只读审核

审核日期：2026-09-13。服务器项目提交：aa0979a。数据库：github_sentiment。

## 结论

1. v7 的确存在明显漏标，包含符合现有定义的正面 API 评价、负面维护体验和生命周期体验；不是只有 neutral 被漏掉。
2. v4 也有普通 Bug、工具名和技术日志触发的误标。建议定向修订方面门槛并建立盲标测试，不直接回退 v4，也不把某个非空比例设为优化目标。
3. 两批不是无交集：有 1,158 条双方成功标注的相同 corpus_id。全部 model_input 与本地旧版导出逐字一致。两次使用 clean-v1，清洗版本变化不能解释这部分下降。
4. 当前语料需要有选择的降噪和作者来源过滤，但不能直接全量套用 clean-v2。现有 clean-v2 有内容误删风险，且评论上下文不足不应通过删评论解决。

## 范围与口径

从数据库读取两版成功标注的全部轻量元数据，共 101,230 条；读取交集 1,158 条和从 v7 成功标注集合简单随机抽取的 1,500 条正文，去重后 2,633 条。随机种子 20260913。

另读取随机 1,500 条对应的作者、URL 和实际 clean-v2 记录；全部 1,500 对原文相同，数据库 clean-v2 与当前清洗函数本地重算结果全部一致。审阅了随机选出的 45 条“有→空”完整输入，另检查了 35 条短空标签文本，并保存 14 条重点案例供裁决。案例筛选不等于人工总体准确率评估。

全程没有更改数据库记录、提示词或服务器代码，也没有调用收费标注接口。本地生成审计文件。一次大表聚合查询超时后改为限时、小批量主键读取；尝试取消原查询时服务器报告线程已不存在，不影响下列成功查询和本地统计结果。

非空比例 = annotations 至少含一个方面的成功标注数 / 成功标注总数。neutral 也算非空；失败或未完成记录不算空数组。

## 实际分布

| 版本 | 成功 | 失败 | 非空 | 非空比例 | 至少一项正/负情感 | 仅 neutral |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 4,972 | 23 | 1,413 | 28.42% | 973 | 440 |
| v7 | 96,258 | 11 | 5,069 | 5.27% | 4,462 | 607 |

这次完整数据库读数不是先前观察的约 33% 和 7%；可能是批次进度或口径不同。旧版 4,995 条抽样、新版 96,269 条抽样均已能对应成功或失败结果。v3 另有 7,179 条成功结果，本次比较没有混入 v3。

旧样本 rust-v1：每仓库上限 100，种子 20260804，无记录的长度上限。新样本 rust-clean-v1-max4000-perrepo2000-v1：每仓库上限 2,000，种子 20260816，clean-v1，model_input 不超过 4,000 字符。

服务器 corpus 中有 clean-v1 884,234 条、clean-v2 883,966 条。这些通常是同一来源的不同清洗版本，不能相加后当成独立原始帖子数。

| 来源 | v4 成功数 | v4 非空率 | v7 成功数 | v7 非空率 |
|---|---:|---:|---:|---:|
{source_table}

按旧版“仓库×来源类型”权重对新版非空率重新加权，结果为 {weighted['repo_id+source_type']:.2%}，与原始 5.27% 接近。因此仅这些已观测样本构成差异不足以解释大幅下降；重加权也不代替配对和人工验证。

## 相同语料配对结果

两批抽样原始交集为 1,164 条，其中 1,158 条双方均标注成功。以下配对统计排除了其余 6 条，避免把失败当成空标签。

| v4 → v7 | 数量 |
|---|---:|
| 有标签 → 空数组 | 296 |
| 有标签 → 有标签 | 68 |
| 空数组 → 有标签 | 10 |
| 空数组 → 空数组 | 784 |

同一组 1,158 条：v4 非空 364 条（31.43%），v7 非空 78 条（6.74%），下降 24.70 个百分点。

neutral 标签项从 111 降为 13，negative 从 203 降为 58，positive 从 72 降为 17。这里是标签项数，多方面语料可以贡献多项。不能据此把所有消失标签认定为漏标。

| 方面 | v4 标签项数 | v7 标签项数 |
|---|---:|---:|
{aspect_table}

## 语义审核

重点案例见 [案例复核.md](案例复核.md)，可回到 GitHub 原文核对。

- 672977：明确说 API 易用且默认值好，v4 api_extensibility/positive，v7 空。按 v7 自身定义也应保留。
- 541166：明确说代码复杂、希望改得更直观，v4 readability_maintainability/negative，v7 空。
- 1291197：明确讨论默认 feature 导致功能退化，并表达此前受影响的负面体验，v4 package_manager/negative，v7 空。
- 672649：作者说自己与生命周期斗争后最终放弃，v7 空；该条来自新样本随机抽查，没有 v4 结果。
- 1141879、1180314：分别包含默认 feature 的实质说明、接口设计及性能问题，适合检验 neutral 门槛。
- 618144、563051、552664：只有工具操作、普通 Bug 标题或崩溃日志；v7 去掉旧标签有合理依据。

已核对重点案例的 raw_response，v7 空数组来自模型原始输出，而非校验器把有效标签清空。配对排除了 corpus 输入内容和样本组成差异，但两次请求发生在不同日期，且 v4 批量、v7 单条、思考模式配置不同。仅凭历史数据仍不能把全部变化精确归因到某一条提示规则或排除服务端模型变化。

提示词中总规则允许“陈述、评价、比较、建议”，而 diagnostics_debugging、libraries_frameworks、community 等标签定义又要求“只有作者评价……才标注”。应统一成：先判实质方面讨论，再判情感；事实与问题可以有方面而没有极性。另需正例明确说明：PR 作者对改进的评价同样属于作者态度；不要把所有 Bug/技术材料附近的自然语言一起当成噪声。

## 清洗审核

以下为 v7 成功集合中的随机 1,500 条，不是全库普查，各项可重叠。

| 项目 | 数量 | 占比 | 含义 |
|---|---:|---:|---|
| 含 fenced block | 219 | 14.60% | 代码、日志、建议、自然语言混合，不能全部视为垃圾 |
| 含 Markdown 引用 | 99 | 6.60% | 需保留作者归属，避免复制被引用者情感 |
| TARGET 少于 40 字符 | 160 | 10.67% | 短不等于无效，需检查引用和上下文 |
| 机器人账户候选 | 49 | 3.27% | 按 [bot] 后缀及已知机器人名称识别，需作为作者来源标记 |
| 含任务复选框 | 35 | 2.33% | 通用清单可能误触发工具/文档方面 |
| 含 details/summary 标签 | 4 | 0.27% | 展示结构可以处理，但不要删有意义摘要 |
| clean-v2 改变正文 | 230 | 15.33% | 已比对数据库实际版本 |
| clean-v2 后只剩占位符 | 6 | 0.40% | 无需再次送模型；不自动意味着原始 6 条都有情感 |

随机样本有 85 条非空（5.67%）。49 条机器人候选全部为空；在这 1,500 条内仅排除机器人，非空比例变为 85/1,451 = 5.86%，远不足以解释与旧版的差距。不要把清洗当作提高非空率的手段。

实际机器人例子：262634 是 bors 的入队通知，351414 是 rust-highfive 自动分配审阅者消息。样本还包含 AI 审阅账户 chatgpt-codex-connector[bot] 的评论（1345270）。如果研究目标是人类开发者态度，应通过来源元数据区分人类、机器人、AI，而不是把这些内容当成人类观点。

发现完全相同的短 TARGET 跨不同上下文重复，如 bors r+、@homu r+、Thanks!。现有内容哈希包含上下文，因此不会把所有这些文本合并。对机械模板可单独去重；不要对所有短评论跨线程强制去重，否则会丢掉上下文区别。

### clean-v2 的具体风险

1. `_collapse_fenced_blocks` 无论长短、语言和内容一律折叠。短 help 提示、基准前后数值、自然语言评价和 suggestion 文档都会消失。
2. `_collapse_stack_traces` 遇到非空、非标题行会持续吞掉正文；没有空行分隔时，真实体验说明也会被删。
3. `_TECHNICAL_LINE_PATTERNS` 对 Cargo/Running/Checking 等行首匹配过宽。连续三行正常英语也能被当作长日志折叠。
4. `CorpusBuilder.build` 检查的是 target_text 是否为空，不检查清洗后是否只剩占位符。应在模型输入资格层标记“无可标注正文”，保留原始记录与理由。

已运行五个构造性测试，完整输入和实际输出在 cleaner-behavior-tests.jsonl。它们证明规则存在误删路径，不表示这些构造句子来自数据库，也不提供全库误删率。

数据库中的实际例子：1327260→2253491、1340256→2266487、1349258→2275489 的文档建议被折叠成一个占位符；814617→1740848 的纯日志被折叠则可能合理。是否有可标注方面仍需依据作者与任务定义判断。

### 建议的清洗边界

- 保留 raw_text、完整 target_text；新增清洗版本和排除原因，不覆盖历史证据。
- 把纯机器人状态、纯控制命令、空模板字段放入可审计的排除层；对混有人类评价的文本按段处理。
- 长日志可折叠；短诊断、比较数值、自然语言、API 限制和所有权相关解释应保留。
- 保留引用的说话人边界；对有明确回复关系的短评论补入直接被回复文本。没有回复链证据时标记上下文不足，不能凭标题臆测立场。
- 4,000 字符筛选会排除较长的完整讨论。之后应先做保真清洗，再定义长度政策，并单独记录被排除的长文本比例。
- 作者字段当前在原始表，需要纳入抽样/训练导出。训练与测试按父 Issue/PR 分组，避免同一讨论串泄漏；若做未来预测，再按时间划分。

## 推荐下一轮测试

1. 从现有 1,158 条配对中审阅全部 306 条空/非空分歧；另从 784 条双空和 68 条双非空中抽样，不能只盯旧版有标签。
2. 使用已导出的 blind-review-500.jsonl 做两人独立盲标，隐藏两版结果并裁决。此文件是随机抽查集的一个固定子集；有少量案例已在本报告中展示，正式盲标应由未看过模型答案的标注者执行。
3. 建立开发集与未参与调规则的锁定测试集，按父讨论串分组。对同一输入和相同调用方式重新比较 v4 语义规则、v7、候选修订版；必要时单独控制思考模式与批大小。
4. 在相同原文上独立做清洗消融：原 clean-v1 / 修复后的新清洗。不要同时改变提示词和清洗后只比较总比例。
5. 分别报告方面 precision/recall/F1、方面+极性端到端 F1、neutral→空比例、每方面支持数与各来源分组结果。对同一方面集合评估极性，避免只在模型成功抽出方面的样本上算情感成绩。
6. 对罕见方面补充边界集（facts、questions、explicit sentiment、pure logs、quotes、negation），并做同义改写与无关日志附加测试。不同模型的一致结果可用于辅助复核，不能替代金标准。

## 交付文件

- summary.json：完整统计与口径。
- paired-1158.jsonl：相同输入两版配对。
- 案例复核.md / review-cases.jsonl：14 个具体案例与审阅意见。
- cleaning-random-1500.jsonl：随机正文、作者、v7 标签及实际 clean-v2 对照。
- blind-review-500.jsonl：不含模型答案的人工标注起点。
- cleaner-behavior-tests.jsonl：五个实际执行的清洗边界测试。
- annotations-metadata.jsonl / sample-selection.json：可复算元数据与抽样 ID。
- build_report.py：本次报告生成脚本；重新生成还需本机临时目录中的原始只读查询结果。
'''
(ROOT / '审核报告.md').write_text(report, encoding='utf-8')
print(json.dumps({'output': str(ROOT), 'summary': summary}, ensure_ascii=False))
