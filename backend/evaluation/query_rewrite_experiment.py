# -*- coding: utf-8 -*-
"""
query_rewrite_experiment.py — deterministic query-rewrite experiment (evaluation-only).

Does NOT modify FOOK_rag_chatbot.py / FOOK_build_rag_kb.py / the KB / any existing evaluation
file. Imports retrieve() / _relevant() / _rerank() / RAG_MIN_SCORE / RAG_LEXICAL_ALPHA /
RAG_CANDIDATE_POOL directly from the production module and chains them manually — production
code itself is never touched.

Makes LIVE OpenAI embedding calls (text-embedding-3-small) via the real retrieve() function.
Every "wide" call below uses top_k = full KB size (536) so that a single retrieve() call gives
both (a) the exact production-scale Top-10 candidate pool used by the real pipeline and (b) the
chunk's true full-corpus rank in one shot — no separate narrow + wide calls needed.

Evidence-chunk identity is established by matching FULL chunk text against a small set of
distinctive substrings drawn from each question's `note` field in rag_eval_questions_v2.json,
independently confirmed against backend/data/rag-source/*.txt via grep BEFORE any rewrite text
was written (see conversation record / report for the exact grep commands and line numbers).
This identifies WHICH chunk is correct evidence; it says nothing about how to word a rewrite,
so it does not create the circularity the task asks to avoid.
"""
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
from FOOK_rag_chatbot import (
    retrieve, _relevant, _rerank, _load_kb,
    RAG_MIN_SCORE, RAG_LEXICAL_ALPHA, RAG_CANDIDATE_POOL, RAG_CONTEXT_MAX,
)

HERE = os.path.dirname(os.path.abspath(__file__))
QPATH = os.path.join(HERE, 'rag_eval_questions_v2.json')
OUT_JSON = os.path.join(HERE, 'query_rewrite_experiment_results.json')

questions = json.load(open(QPATH, encoding='utf-8'))
qmap = {q['id']: q for q in questions}
positives = [q for q in questions if q['type'] == 'rag']
oos = [q for q in questions if q['type'] == 'out_of_scope']
assert len(positives) == 30 and len(oos) == 21

kb, mat = _load_kb()
KB_N = len(kb)
print(f"KB size = {KB_N}", flush=True)

HARD_IDS = ['itch1', 'prot2', 'snack1', 'fish1', 'transplant1', 'veg1', 'snack2']


# ── evidence-chunk match rules (documented, grep-confirmed BEFORE rewrite design) ───────────
def _has_all(text, subs):
    return all(s in text for s in subs)


def _has_any(text, subs):
    return any(s in text for s in subs)


EVIDENCE_RULES = {
    # grep: 혈액투석_영양식생활관리_2권.txt:3668 "Key Message: 요독성성 가려움증의 치료..." /
    # :3660 "투석환자의 40–80%에서 발생" — require "요독성" + a body-content marker so the bare
    # TOC heading line ("40. 몸이 너무 가려워요. (요독성 가려움증)", line 187, no body markers)
    # does NOT match.
    'itch1': lambda t: '요독성' in t and _has_any(t, ['부갑상선', '유병율', '40', '호산구']),
    # grep: 콩팥병_환자를_위한_안내서.txt:5874 "...을 1.0-1.2 g/kg/day로 늘려야 합니다."
    # also accept the paraphrase at :6230 "단백 섭취량을 1-1.2 g/kg/day까지 증가시킬 수 있습니다."
    'prot2': lambda t: ('1.0-1.2 g/kg/day' in t and '늘려야' in t) or ('1-1.2 g/kg/day' in t and '증가' in t),
    # grep: 혈액투석환자를_위한_간식.txt:21 "[슈가토스트]"
    'snack1': lambda t: '슈가토스트' in t,
    # two acceptable sources per v2 changelog: 콩팥병_환자를_위한_안내서:4024 "닭고기, 생선, 달걀과
    # 같은 육식을 너무 많이 드시지 마십시오" OR 혈액투석_영양식생활관리_2권 식품구성자전거 "고기, 생선,
    # 달걀, 콩류(매일 3-4회..." — accept either.
    'fish1': lambda t: ('닭고기, 생선, 달걀' in t) or ('고기, 생선, 달걀, 콩류' in t),
    # grep: 콩팥병_환자를_위한_안내서.txt:2812 "...에 3 L 이상의 물을 필요로 할 수 있습니다."
    'transplant1': lambda t: '3 L 이상의 물을 필요로 할 수 있습니다' in t,
    # grep: 혈액투석_영양식생활관리_2권.txt:440 "...과일과 채소 섭취량을 적절하게 섭"
    'veg1': lambda t: '과일과 채소 섭취량을 적절하게' in t,
    # grep: 혈액투석환자를_위한_간식.txt:7 "...100g 내외의 과일, 1/2~1컵 정도의 우유를 섭취하도록 권장"
    'snack2': lambda t: '100g 내외의 과일' in t,
}

# ── EVIDENCE_INDICES for the other 23 positive questions (0-based index into that question's
# OWN baseline Top-10 pool, i.e. rank = index+1) — reused as-is from reranking_experiment.py
# (backend/evaluation/reranking_experiment.py), which established this ground truth by reading
# full untruncated text of every candidate against each question's expected_source + note, on
# the SAME current (post-TOC-filter, 536-chunk) KB. Not re-derived here to avoid duplicating
# that manual text-reading work; reused read-only, no existing evaluation file is modified.
EVIDENCE_INDICES_23 = {
    'k1': 0, 'k2': 4, 'p1': 5, 'p2': 0, 'na1': 0, 'na2': 1,
    'prot1': 2, 'water1': 8, 'water2': 3, 'eat1': 4, 'eat2': 0,
    'fruit1': 3, 'fruit2': 9, 'ckd1': 1, 'ckd2': 3,
    'p3': 0, 'trip1': 0, 'alb1': 0, 'bone1': 0,
    'early1': 8, 'energy1': 3, 'kfruit1': 0, 'acidosis1': 0,
}
assert set(EVIDENCE_INDICES_23) | set(HARD_IDS) == {q['id'] for q in positives}


def wide(query):
    """One live retrieve() call at full-corpus top_k. Returns the full sorted hit list."""
    return retrieve(query, top_k=KB_N)


def find_rank(hits, rule):
    """1-based rank of the first hit whose text satisfies rule(text); None if none match."""
    for i, h in enumerate(hits):
        if rule(h['text']):
            return i + 1, h['score']
    return None, None


def find_rank_by_text(hits, target_text):
    for i, h in enumerate(hits):
        if h['text'] == target_text:
            return i + 1, h['score']
    return None, None


results = {'RAG_MIN_SCORE': RAG_MIN_SCORE, 'RAG_LEXICAL_ALPHA': RAG_LEXICAL_ALPHA,
           'RAG_CANDIDATE_POOL': RAG_CANDIDATE_POOL, 'RAG_CONTEXT_MAX': RAG_CONTEXT_MAX,
           'KB_N': KB_N}


def log(*a):
    print(*a, flush=True)
    with open(os.path.join(HERE, 'query_rewrite_experiment.log'), 'a', encoding='utf-8') as f:
        print(*a, file=f)


def save():
    json.dump(results, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — fresh baseline for the 7 hard cases (live retrieve, full corpus)
# ════════════════════════════════════════════════════════════════════════════════════════════
log("=== PHASE 1: fresh baseline, 7 hard cases ===")
phase1 = {}
evidence_text = {}  # qid -> canonical evidence chunk text (locked once, reused for all rewrites)
for qid in HARD_IDS:
    q = qmap[qid]
    hits = wide(q['question'])
    rank, score = find_rank(hits, EVIDENCE_RULES[qid])
    top1 = hits[0]
    entry = {
        'question': q['question'],
        'evidence_rank': rank,
        'evidence_score': score,
        'top1_score': top1['score'],
        'top1_source': top1['source'],
        'gap_vs_top1': (top1['score'] - score) if score is not None else None,
    }
    if rank is not None:
        evidence_text[qid] = hits[rank - 1]['text']
        entry['evidence_source'] = hits[rank - 1]['source']
    phase1[qid] = entry
    log(f"[baseline] {qid}: rank={rank} score={score} top1={top1['score']:.4f}")
results['phase1_baseline'] = phase1
save()


# ════════════════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — rewrite design (max 2 per question, patterns B/C/D only)
# Reasoning is written to be generalizable, not reverse-engineered from evidence-text wording.
# itch1's C-pattern term ("소양증") was grep-confirmed in KB body prose BEFORE any rewrite was
# drafted (혈액투석_영양식생활관리_2권.txt:2182, a DIFFERENT passage than itch1's own evidence
# chunk — see report). All D-pattern suffixes below use generic quantity/frequency vocabulary
# ("권장량", "기준", "섭취량", "빈도") deliberately chosen to NOT reuse any distinctive phrase
# found inside each question's own evidence chunk (e.g. avoids "100g"/"우유" for snack2, avoids
# "g/kg/day" for prot2, avoids "슈가토스트"/dish names for snack1) — self-critique on this point
# is in the report.
# ════════════════════════════════════════════════════════════════════════════════════════════
REWRITES = {
    'itch1': [
        {'pattern': 'B', 'text': '혈액투석 환자에게 나타나는 가려움증은 왜 생기나요?',
         'reasoning': "Domain-prefix reframe of the exact same casual intent (why do I itch "
                      "since starting dialysis) — adds '혈액투석 환자' framing, drops no/adds "
                      "no facts, keeps the everyday word '가려움증'."},
        {'pattern': 'C', 'text': '혈액투석 환자의 소양증 원인',
         'reasoning': "Clinical-register synonym for '가려움' -> '소양증'. Verified via "
                      "`grep 소양증 data/rag-source/*.txt` BEFORE drafting this rewrite: the "
                      "term appears in real explanatory prose at 혈액투석_영양식생활관리_2권.txt"
                      ":2182 ('...요독증과 관련된 생화학적 이상, 피부 소양증과 관절통증등의 "
                      "신체증상이 있습니다'), a passage OTHER than itch1's own answer chunk — "
                      "so this is genuine independent KB vocabulary, not a copy of the target "
                      "answer's wording."},
    ],
    'prot2': [
        {'pattern': 'B', 'text': '혈액투석 환자의 하루 단백질 섭취 권장량은 얼마인가요?',
         'reasoning': "Domain-prefix reframe of the same intent (how much protein), no new "
                      "facts introduced."},
        {'pattern': 'D', 'text': '투석을 받고 있는 환자는 단백질을 얼마나 섭취해야 하나요? '
                                  '권장량 체중 기준',
         'reasoning': "Generic quantity-question suffix ('recommended amount, per body "
                      "weight') — deliberately avoids quoting the evidence's actual unit "
                      "phrasing 'g/kg/day' to not reverse-engineer the answer."},
    ],
    'snack1': [
        {'pattern': 'B', 'text': '혈액투석 환자를 위한 간식 조리법에는 어떤 것이 있나요?',
         'reasoning': "Domain-prefix reframe of the same intent (snack recipes), no new facts."},
        {'pattern': 'D', 'text': '혈액투석 환자가 먹을 수 있는 간식 레시피를 알려주세요. '
                                  '재료 조리법',
         'reasoning': "Generic recipe-question suffix ('ingredients, cooking method') — true "
                      "of any recipe request, deliberately avoids naming specific dishes (e.g. "
                      "'슈가토스트') found in the evidence chunk."},
    ],
    'fish1': [
        {'pattern': 'B', 'text': '혈액투석 환자가 생선은 얼마나 드셔도 되나요?',
         'reasoning': "Domain-prefix reframe of the same intent, no new facts."},
        {'pattern': 'D', 'text': '생선은 얼마나 드셔도 되나요? 섭취 빈도 기준',
         'reasoning': "Generic frequency-question suffix — avoids quoting the evidence's "
                      "specific '3-4회'/'식품구성자전거' wording."},
    ],
    'transplant1': [
        {'pattern': 'B', 'text': '신장이식 환자의 이식 초기 수분 섭취량은 얼마나 되나요?',
         'reasoning': "Domain-prefix reframe adapted to the patient population the question is "
                      "actually about (신장이식 환자, not 혈액투석 환자 — the B pattern's intent "
                      "is 'reframe with a clinical patient-population prefix', and this question "
                      "is genuinely about transplant patients, not dialysis patients), keeps "
                      "the '초기'(early period) nuance from the original, no new facts."},
        {'pattern': 'D', 'text': '신장이식을 받은 직후에는 물을 얼마나 마셔야 하나요? '
                                  '하루 섭취량 기준',
         'reasoning': "Generic quantity-question suffix — avoids quoting the evidence's "
                      "specific '3 L' figure."},
    ],
    'veg1': [
        {'pattern': 'B', 'text': '혈액투석 환자는 채소를 얼마나 먹어도 되나요?',
         'reasoning': "Domain-prefix reframe of the same intent, no new facts."},
        {'pattern': 'D', 'text': '채소는 얼마나 먹어도 되나요? 하루 권장 섭취량',
         'reasoning': "Generic quantity-question suffix — deliberately avoids the evidence "
                      "chunk's actual co-occurring word '과일'(fruit) even though it is "
                      "topically adjacent, specifically to not reverse-engineer from the found "
                      "answer text."},
    ],
    'snack2': [
        {'pattern': 'B', 'text': '혈액투석 환자는 간식으로 과일을 하루에 얼마나 먹어도 되나요?',
         'reasoning': "Domain-prefix reframe of the same intent, no new facts."},
        {'pattern': 'D', 'text': '간식으로 과일은 하루에 얼마나 먹어도 되나요? 1일 권장량 기준',
         'reasoning': "Generic quantity-question suffix — avoids quoting the evidence's "
                      "specific '100g'/'우유' figures."},
    ],
}

log("=== PHASE 2/3: rewrite retrieval (live), 7 hard cases ===")
phase3 = {}
for qid in HARD_IDS:
    q = qmap[qid]
    entry = {'rewrites': []}
    for rw in REWRITES[qid]:
        hits = wide(rw['text'])
        # rank tracked by exact text match against the SAME evidence chunk locked in phase 1
        if qid in evidence_text:
            rank, score = find_rank_by_text(hits, evidence_text[qid])
        else:
            rank, score = find_rank(hits, EVIDENCE_RULES[qid])
        top1 = hits[0]
        r = {
            'pattern': rw['pattern'], 'text': rw['text'], 'reasoning': rw['reasoning'],
            'evidence_rank': rank, 'evidence_score': score,
            'top1_score': top1['score'], 'top1_source': top1['source'],
            'in_pool10': (rank is not None and rank <= RAG_CANDIDATE_POOL),
            'recall_at_3': (rank is not None and rank <= 3),
            'recall_at_5': (rank is not None and rank <= 5),
            'recall_at_10': (rank is not None and rank <= RAG_CANDIDATE_POOL),
        }
        entry['rewrites'].append(r)
        log(f"[rewrite] {qid} [{rw['pattern']}] '{rw['text']}' -> rank={rank} score={score} "
            f"top1={top1['score']:.4f}")
    phase3[qid] = entry
results['phase2_3_rewrites'] = phase3
save()


# ════════════════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — full production-pipeline simulation for rewrites that measurably helped
# (i.e. moved the evidence chunk's full-corpus rank to <= RAG_CANDIDATE_POOL, since only those
# could possibly reach the LLM context under the real pipeline's retrieve(top_k=10) call).
# Chains retrieve() -> _relevant() -> _rerank() EXACTLY as answer() does (see FOOK_rag_chatbot.py
# answer() docstring) — production code itself is never modified, only imported and called here.
#
# Design decision (documented per task instruction): the lexical-bonus term inside _rerank() is
# computed against the ORIGINAL question text, not the rewritten text. Reasoning: production has
# no concept of "this query was rewritten" — a real future integration would rewrite the query
# ONLY for the embedding-retrieval call, then still want to judge "does this candidate actually
# match what the user asked" using the user's real wording for the lexical-overlap signal. Using
# the rewrite text for the bonus would let generic/added words in the rewrite (e.g. a D-pattern
# suffix like '권장량 체중 기준') inflate lexical-overlap scores against chunks that only overlap
# with the rewrite's added vocabulary, not the user's actual question — precisely the kind of
# vocabulary-coincidence risk flagged in the out-of-scope safety findings below. Scoring against
# the original question keeps the rerank stage anchored to genuine user intent.
# ════════════════════════════════════════════════════════════════════════════════════════════
log("=== PHASE 4: production-pipeline simulation for in-pool rewrites ===")
phase4 = {}
for qid in HARD_IDS:
    q = qmap[qid]
    orig_question = q['question']
    for rw in REWRITES[qid]:
        rank_full = None
        for r in phase3[qid]['rewrites']:
            if r['text'] == rw['text']:
                rank_full = r['evidence_rank']
        if rank_full is None or rank_full > RAG_CANDIDATE_POOL:
            continue  # can't possibly reach the real candidate pool; skip production sim
        hits10 = retrieve(rw['text'], top_k=RAG_CANDIDATE_POOL)   # exact production call shape
        relevant = _relevant(hits10)                               # real gate, RAG_MIN_SCORE
        reranked = _rerank(orig_question, relevant)                # real rerank, orig-question lexical bonus
        top5 = reranked[:RAG_CONTEXT_MAX]
        if qid in evidence_text:
            pos_in_gate = next((i + 1 for i, h in enumerate(relevant) if h['text'] == evidence_text[qid]), None)
            pos_in_final = next((i + 1 for i, h in enumerate(top5) if h['text'] == evidence_text[qid]), None)
        else:
            pos_in_gate = pos_in_final = None
        key = f"{qid}::{rw['pattern']}"
        phase4[key] = {
            'qid': qid, 'pattern': rw['pattern'], 'rewrite_text': rw['text'],
            'pool10_size': len(hits10), 'gate_survivors': len(relevant),
            'evidence_passed_gate': pos_in_gate is not None,
            'rank_in_gate_survivors': pos_in_gate,
            'rank_in_final_top5': pos_in_final,
            'in_final_top5': pos_in_final is not None,
        }
        log(f"[prodsim] {qid} [{rw['pattern']}]: gate_survivors={len(relevant)} "
            f"passed_gate={pos_in_gate is not None} final_top5={pos_in_final is not None} "
            f"(rank_in_final={pos_in_final})")
results['phase4_production_sim'] = phase4
save()


# ════════════════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — out-of-scope gate-safety check (ALL 21, mechanical rewrite rule)
# The ONLY rewrite pattern that is truly "mechanical" for arbitrary/unknown-topic input is B
# (domain-prefix) applied as a FIXED, content-blind template — C needs a per-topic synonym a
# real system can't know in advance, D needs per-topic keywords for the same reason. So the
# mechanical-safety check below applies ONE fixed template to every question, uniformly, exactly
# as a real non-NLU implementation would: MECH_B(q) = "혈액투석 환자의 " + q. No per-question
# tailoring anywhere in this phase.
# ════════════════════════════════════════════════════════════════════════════════════════════
def mech_b(question):
    return f"혈액투석 환자의 {question}"


log("=== PHASE 5: out-of-scope gate-safety check (21/21, mechanical rewrite B) ===")
phase5 = []
for q in oos:
    hits_before = retrieve(q['question'], top_k=RAG_CANDIDATE_POOL)
    gate_before = len(_relevant(hits_before)) > 0
    top1_before = hits_before[0]['score']

    rw_text = mech_b(q['question'])
    hits_after = retrieve(rw_text, top_k=RAG_CANDIDATE_POOL)
    gate_after = len(_relevant(hits_after)) > 0
    top1_after = hits_after[0]['score']

    entry = {
        'id': q['id'], 'question': q['question'], 'rewrite': rw_text,
        'top1_score_before': top1_before, 'top1_score_after': top1_after,
        'delta': top1_after - top1_before,
        'gate_pass_before': gate_before, 'gate_pass_after': gate_after,
        'newly_passes_gate': (not gate_before) and gate_after,
    }
    phase5.append(entry)
    flag = " !!! NEWLY PASSES GATE !!!" if entry['newly_passes_gate'] else ""
    log(f"[oos-safety] {q['id']}: before={top1_before:.4f}({gate_before}) "
        f"after={top1_after:.4f}({gate_after}) delta={entry['delta']:+.4f}{flag}")
results['phase5_oos_safety'] = phase5
save()


# ════════════════════════════════════════════════════════════════════════════════════════════
# PHASE 6 — full 30-positive-question regression check, mechanical rewrite B, production sim
# Chosen pattern for broad application: B (mechanical, fixed-template) — the only pattern that
# generalizes to arbitrary input without per-question tailoring (see Phase 5 docstring); this is
# the pattern actually being weighed for "apply to all RAG questions" in the recommendation.
# ════════════════════════════════════════════════════════════════════════════════════════════
log("=== PHASE 6: full 30-positive regression check, mechanical rewrite B ===")


def prod_sim(query_for_retrieve, query_for_rerank, evidence_matcher):
    """Runs the exact production chain and returns rank info for the matched evidence chunk."""
    hits10 = retrieve(query_for_retrieve, top_k=RAG_CANDIDATE_POOL)
    relevant = _relevant(hits10)
    reranked = _rerank(query_for_rerank, relevant)
    top5 = reranked[:RAG_CONTEXT_MAX]
    pos_gate10 = next((i + 1 for i, h in enumerate(hits10) if evidence_matcher(h['text'])), None)
    pos_gate = next((i + 1 for i, h in enumerate(relevant) if evidence_matcher(h['text'])), None)
    pos_final5 = next((i + 1 for i, h in enumerate(top5) if evidence_matcher(h['text'])), None)
    return {
        'top1_score': hits10[0]['score'], 'gate_pass': len(relevant) > 0,
        'evidence_rank_in_pool10': pos_gate10, 'evidence_rank_after_gate': pos_gate,
        'evidence_rank_final5': pos_final5, 'in_final_top5': pos_final5 is not None,
        'recall_at_1_pool': pos_gate10 == 1, 'recall_at_3_pool': (pos_gate10 is not None and pos_gate10 <= 3),
        'recall_at_5_pool': (pos_gate10 is not None and pos_gate10 <= 5),
        'recall_at_10_pool': pos_gate10 is not None,
    }


# lock evidence-text matchers for the 23 non-hard positive questions from a fresh baseline
# retrieve(top_k=10) call + the reused EVIDENCE_INDICES_23 ground-truth index (see Phase 2 note)
easy23_evidence_text = {}
easy23_baseline_pool = {}
for qid, idx in EVIDENCE_INDICES_23.items():
    q = qmap[qid]
    hits10 = retrieve(q['question'], top_k=RAG_CANDIDATE_POOL)
    easy23_baseline_pool[qid] = hits10
    easy23_evidence_text[qid] = hits10[idx]['text']

phase6 = {}
for q in positives:
    qid = q['id']
    orig_question = q['question']
    rw_text = mech_b(orig_question)

    if qid in evidence_text:
        matcher = (lambda t, _txt=evidence_text[qid]: t == _txt)
    elif qid in easy23_evidence_text:
        matcher = (lambda t, _txt=easy23_evidence_text[qid]: t == _txt)
    else:
        matcher = EVIDENCE_RULES.get(qid, lambda t: False)

    before = prod_sim(orig_question, orig_question, matcher)
    after = prod_sim(rw_text, orig_question, matcher)  # lexical bonus vs ORIGINAL, per Phase-4 decision

    phase6[qid] = {'question': orig_question, 'rewrite': rw_text, 'before': before, 'after': after}
    log(f"[regress30] {qid}: before(top5={before['in_final_top5']},@10={before['recall_at_10_pool']}) "
        f"after(top5={after['in_final_top5']},@10={after['recall_at_10_pool']})")
results['phase6_regression30'] = phase6
save()

log("=== DONE ===")
