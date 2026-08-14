# -*- coding: utf-8 -*-
"""
reranking_experiment.py — lightweight-reranking comparison experiment (evaluation-only).

Reuses the LIVE Top-10 embedding-candidate pool captured once by reranking_fetch_pool.py
(backend/evaluation/reranking_top10_pool.json) for all 30 positive ('rag') questions and all
21 'out_of_scope' questions. NO new embedding/API calls happen in this file — every method
below (dedup / source-diversity cap / lexical bonus / combinations) is a pure, deterministic
re-ordering of the SAME cached candidates, per the task's hard constraint.

Evidence-rank ground truth (EVIDENCE_INDICES below) was established by hand: reading the full,
untruncated text of every expected-source candidate in each question's Top-10 pool (not just
score-based source-matching), cross-checked against prior work (evidence_miss_diagnosis_results
.json for itch1/prot2/snack1/fish1/transplant1, rag_evidence_inspection_v2.md's evidence_relevant
judgments for the rest). This full-text reading caught real evidence buried past the ~200-char
preview a naive substring scan would have missed (e.g. p1, prot1, water1, fruit2, energy1 — see
reranking_evidence_notes.md for the worked example of each), and also surfaced TWO genuine
misses not on the previously-known list of 5 (veg1, snack2) — both confirmed absent from the
Top-10 pool by a full-corpus (top_k=536) live search. Indices are 0-based positions in the
ORIGINAL embedding-score-sorted pool (index 0 = embedding rank 1).

Does NOT modify FOOK_rag_chatbot.py / FOOK_build_rag_kb.py / the KB / any existing evaluation
file. Read-only consumer of reranking_top10_pool.json.
"""
import json, os, re, difflib

POOL_PATH = os.path.join(os.path.dirname(__file__), 'reranking_top10_pool.json')

# ── evidence ground truth (see docstring) ───────────────────────────────────────────────────
# qid -> list of 0-based indices in the ORIGINAL top-10 pool that contain genuine, adequate
# evidence for the question (empty list = evidence confirmed NOT present in the top-10 pool).
EVIDENCE_INDICES = {
    'k1': [0], 'k2': [4], 'p1': [5], 'p2': [0], 'na1': [0], 'na2': [1, 6],
    'prot1': [2, 7], 'prot2': [], 'water1': [8], 'water2': [3], 'eat1': [4], 'eat2': [0, 2],
    'snack1': [], 'fruit1': [3], 'fruit2': [9], 'ckd1': [1], 'ckd2': [3], 'veg1': [],
    'fish1': [], 'p3': [0], 'trip1': [0], 'alb1': [0], 'bone1': [0], 'itch1': [],
    'early1': [8], 'snack2': [], 'energy1': [3], 'transplant1': [], 'kfruit1': [0, 4],
    'acidosis1': [0],
}
KNOWN_5_FAILURES = {'itch1', 'prot2', 'snack1', 'fish1', 'transplant1'}
NEWLY_FOUND_FAILURES = {'veg1', 'snack2'}  # confirmed absent from top10 during this task's rigor pass


# ── normalization helpers (no heavy NLP; whitespace/punct/dash normalization only) ─────────
_DASH_RE = re.compile(r'[‐‑‒–—−~－]')
_WS_RE = re.compile(r'\s+')
_PUNCT_RE = re.compile(r'[.,!?()\[\]{}·:;"\'“”‘’…/\\|~\-‐-―]')


def norm_strict(s):
    """whitespace-stripped, dash-normalized — used for exact/near-duplicate text comparison."""
    s = _DASH_RE.sub('-', s)
    return _WS_RE.sub('', s)


def norm_light(s):
    """whitespace-collapsed, punctuation-stripped, lowercased — used for lexical overlap."""
    s = _PUNCT_RE.sub(' ', s)
    s = _WS_RE.sub(' ', s).strip().lower()
    return s


# ── Method B: near-duplicate dedup ──────────────────────────────────────────────────────────
DEDUP_RATIO_THRESHOLD = 0.5          # difflib.SequenceMatcher ratio on normalized text
DEDUP_SUBSTRING_MIN_LEN = 100        # min shared length for substring-containment to count


def is_near_duplicate(text_a, text_b):
    na, nb = norm_strict(text_a), norm_strict(text_b)
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if shorter and shorter in longer and len(shorter) >= DEDUP_SUBSTRING_MIN_LEN:
        return True
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return ratio >= DEDUP_RATIO_THRESHOLD


def apply_dedup(pool):
    """Returns surviving original indices, in original score order (first-seen wins, i.e. the
    higher-scoring member of any near-duplicate pair is kept)."""
    kept = []
    for i, c in enumerate(pool):
        if any(is_near_duplicate(c['text'], pool[j]['text']) for j in kept):
            continue
        kept.append(i)
    return kept


# ── Method C: source-diversity cap ──────────────────────────────────────────────────────────
def apply_diversity_cap(indices, pool, cap):
    """indices: ordered list of original indices (score order) to re-order.
    Walks the list; once a source has filled `cap` slots, further candidates from that source
    are deferred to the tail (still score-ordered among themselves), letting the next-best
    different-source candidate backfill earlier."""
    primary, deferred = [], []
    counts = {}
    for i in indices:
        src = pool[i]['source']
        counts.setdefault(src, 0)
        if counts[src] < cap:
            primary.append(i)
            counts[src] += 1
        else:
            deferred.append(i)
    return primary + deferred


# ── Method D: lexical-overlap bonus ─────────────────────────────────────────────────────────
# Korean josa/morphology make exact tokenization unreliable, so this strips a small curated set
# of common trailing particles/sentence-ending suffixes from whitespace-split question words,
# then checks SUBSTRING containment of each remaining content-word in the (lightly normalized)
# chunk text — no morphological analyzer, no new dependency.
_STOP_SUFFIXES = sorted([
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로', '와', '과', '도', '만',
    '나요', '까요', '습니까', '합니까', '되나요', '인가요', '하나요', '해요', '이나', '한가요',
    '해야', '해도', '해서', '해서요', '드셔도', '드릴까요',
], key=len, reverse=True)


def question_content_tokens(question):
    q = _PUNCT_RE.sub(' ', question).strip()
    words = _WS_RE.split(q)
    toks = []
    for w in words:
        w2 = w
        for suf in _STOP_SUFFIXES:
            if w2.endswith(suf) and len(w2) - len(suf) >= 2:
                w2 = w2[: -len(suf)]
                break
        if len(w2) >= 2:
            toks.append(w2)
    # de-dup while preserving order
    seen = set(); out = []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def lexical_bonus(tokens, text):
    if not tokens:
        return 0.0
    nt = norm_light(text)
    hits = sum(1 for t in tokens if t in nt)
    return hits / len(tokens)


def apply_lexical_rerank(indices, pool, question, alpha):
    tokens = question_content_tokens(question)
    scored = []
    for rank_pos, i in enumerate(indices):
        bonus = lexical_bonus(tokens, pool[i]['text'])
        final = pool[i]['score'] + alpha * bonus
        scored.append((final, rank_pos, i))  # rank_pos as stable tiebreaker (preserves order on ties)
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, _, i in scored]


# ── evidence-rank lookup after reranking ────────────────────────────────────────────────────
def evidence_rank_after(order, evidence_idx_set):
    if not evidence_idx_set:
        return None
    for pos, orig_i in enumerate(order):
        if orig_i in evidence_idx_set:
            return pos + 1  # 1-based
    return None


def recall_at_k(ranks, k):
    n = len(ranks)
    hit = sum(1 for r in ranks if r is not None and r <= k)
    return hit / n if n else 0.0


def main():
    data = json.load(open(POOL_PATH, encoding='utf-8'))
    rag = data['rag']
    oos = data['out_of_scope']
    RAG_MIN_SCORE = data['RAG_MIN_SCORE']

    methods = {}  # method_name -> {qid: rank_or_None}

    # baseline (embedding-only, original order == indices 0..9)
    baseline = {}
    for e in rag:
        idx = list(range(len(e['pool'])))
        baseline[e['id']] = evidence_rank_after(idx, set(EVIDENCE_INDICES[e['id']]))
    methods['baseline'] = baseline

    # Method B: dedup
    dedup_ranks = {}
    dedup_detail = {}
    for e in rag:
        order = apply_dedup(e['pool'])
        dedup_ranks[e['id']] = evidence_rank_after(order, set(EVIDENCE_INDICES[e['id']]))
        dedup_detail[e['id']] = {'surviving_original_indices': order, 'n_dropped': len(e['pool']) - len(order)}
    methods['dedup'] = dedup_ranks

    # Method C: source-diversity cap (cap=2, cap=3)
    for cap in (2, 3):
        ranks = {}
        for e in rag:
            order = apply_diversity_cap(list(range(len(e['pool']))), e['pool'], cap)
            ranks[e['id']] = evidence_rank_after(order, set(EVIDENCE_INDICES[e['id']]))
        methods[f'diversity_cap{cap}'] = ranks

    # Method D: lexical bonus (alpha = 0.05, 0.15)
    for alpha in (0.05, 0.15):
        ranks = {}
        for e in rag:
            order = apply_lexical_rerank(list(range(len(e['pool']))), e['pool'], e['question'], alpha)
            ranks[e['id']] = evidence_rank_after(order, set(EVIDENCE_INDICES[e['id']]))
        methods[f'lexical_a{alpha}'] = ranks

    # Method E: combination(s) — decided after seeing B/C/D (see report). Here: dedup -> lexical
    # (alpha=0.05) -> diversity cap(2), and dedup -> lexical(alpha=0.05) alone (no cap), since
    # cap variants are evaluated separately below for comparison.
    for combo_name, cap in (('combo_dedup_lex_cap2', 2), ('combo_dedup_lex_cap3', 3)):
        ranks = {}
        for e in rag:
            order = apply_dedup(e['pool'])
            order = apply_lexical_rerank(order, e['pool'], e['question'], 0.05)
            order = apply_diversity_cap(order, e['pool'], cap)
            ranks[e['id']] = evidence_rank_after(order, set(EVIDENCE_INDICES[e['id']]))
        methods[combo_name] = ranks

    ranks_dedup_lex_nocap = {}
    for e in rag:
        order = apply_dedup(e['pool'])
        order = apply_lexical_rerank(order, e['pool'], e['question'], 0.05)
        ranks_dedup_lex_nocap[e['id']] = evidence_rank_after(order, set(EVIDENCE_INDICES[e['id']]))
    methods['combo_dedup_lex_nocap'] = ranks_dedup_lex_nocap

    # ── summary metrics ──────────────────────────────────────────────────────────────────
    qids = [e['id'] for e in rag]
    summary = {}
    for name, ranks in methods.items():
        rlist = [ranks[q] for q in qids]
        summary[name] = {
            'recall@1': round(recall_at_k(rlist, 1), 4),
            'recall@3': round(recall_at_k(rlist, 3), 4),
            'recall@5': round(recall_at_k(rlist, 5), 4),
            'recall@10': round(recall_at_k(rlist, 10), 4),
        }

    # ── regression check vs baseline (Top-5 window, the production-relevant one) ──────────
    regressions = {}
    for name, ranks in methods.items():
        if name == 'baseline':
            continue
        improved, worsened, unchanged = [], [], []
        for q in qids:
            b = baseline[q]
            m = ranks[q]
            b_in5 = b is not None and b <= 5
            m_in5 = m is not None and m <= 5
            if b_in5 == m_in5 and b == m:
                unchanged.append(q)
            elif (not b_in5) and m_in5:
                improved.append({'id': q, 'before': b, 'after': m})
            elif b_in5 and (not m_in5):
                worsened.append({'id': q, 'before': b, 'after': m})
            elif b_in5 and m_in5 and b != m:
                # both still in top5 but rank shuffled; note as neutral-but-changed
                unchanged.append(q)
            else:
                unchanged.append(q)
        regressions[name] = {'improved': improved, 'worsened': worsened,
                              'n_improved': len(improved), 'n_worsened': len(worsened),
                              'n_unchanged': len(unchanged)}

    # ── known-5 + newly-found-2 failure table ───────────────────────────────────────────
    focus_qs = ['itch1', 'prot2', 'snack1', 'fish1', 'transplant1', 'veg1', 'snack2']
    focus_table = {}
    for q in focus_qs:
        focus_table[q] = {name: ranks[q] for name, ranks in methods.items()}

    # ── out-of-scope safety check ───────────────────────────────────────────────────────
    oos_report = []
    for e in oos:
        top1_score = e['pool'][0]['score']
        top1_relevant_baseline = top1_score >= RAG_MIN_SCORE
        # would lexical bonus push any candidate above the gate that wasn't already?
        tokens = question_content_tokens(e['question'])
        max_bonus_alpha015 = 0.0
        best_after = top1_score
        flagged = None
        for c in e['pool']:
            b = lexical_bonus(tokens, c['text'])
            after = c['score'] + 0.15 * b
            if after > best_after:
                best_after = after
            if b > max_bonus_alpha015:
                max_bonus_alpha015 = b
            if c['score'] < RAG_MIN_SCORE <= after:
                flagged = {'source': c['source'], 'orig_score': c['score'], 'bonus': round(b, 3),
                           'after_score': round(after, 4)}
        oos_report.append({
            'id': e['id'], 'question': e['question'], 'top1_score': round(top1_score, 4),
            'gate_pass_baseline': top1_relevant_baseline,
            'max_lexical_bonus_any_candidate': round(max_bonus_alpha015, 3),
            'best_score_after_lexical_a015': round(best_after, 4),
            'gate_pass_after_lexical_a015': best_after >= RAG_MIN_SCORE,
            'flagged_gate_flip': flagged,
        })

    out = {
        'RAG_MIN_SCORE': RAG_MIN_SCORE,
        'n_rag_questions': len(qids),
        'evidence_indices': EVIDENCE_INDICES,
        'summary_recall': summary,
        'regressions_vs_baseline_top5': regressions,
        'focus_table_known5_and_new2': focus_table,
        'oos_safety_check': oos_report,
        'method_ranks_all_questions': methods,
        'dedup_detail': dedup_detail,
    }
    outpath = os.path.join(os.path.dirname(__file__), 'reranking_experiment_results.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\n--- regressions (top5 window) ---')
    for name, r in regressions.items():
        print(f"{name:28s} improved={r['n_improved']} worsened={r['n_worsened']} unchanged={r['n_unchanged']}"
              + (f"  WORSENED={[w['id'] for w in r['worsened']]}" if r['worsened'] else ''))
    print(f'\nwrote {outpath}')


if __name__ == '__main__':
    main()
