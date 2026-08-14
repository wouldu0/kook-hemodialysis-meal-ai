# Lightweight Reranking Experiment — RAG Top-10 Candidate Pool (post-TOC-filter KB, 536 chunks)

Evaluation-only comparison. Production `FOOK_rag_chatbot.py` / `FOOK_build_rag_kb.py` / the KB
were **not** touched. All numbers below come from `reranking_experiment.py`, operating on ONE
live `retrieve(question, top_k=10)` call per question (captured once by `reranking_fetch_pool.py`
into `reranking_top10_pool.json` and reused for every method — no re-embedding anywhere in this
experiment). `RAG_MIN_SCORE=0.30` (unchanged).

Evidence-rank ground truth was established by reading the **full, untruncated** text of every
expected-source candidate in each question's Top-10 pool (not source-name matching, not a
200-char preview) — a naive-substring or truncated-preview pass would have wrongly scored p1,
prot1, water1, fruit2 and energy1 as misses; the real evidence was simply past the preview
window or used a different Unicode dash character (‐ vs -) than a first-draft marker expected.
This full-text pass also surfaced **two additional genuine Top-10 misses not on the previously
known list of 5** — `veg1` and `snack2` — both confirmed absent even at full-corpus rank (30 and
54 respectively) by a direct top_k=536 search. So the true Top-10 miss set for this pass is
**7**, not 5: `itch1, prot2, snack1, fish1, transplant1` (previously known) + `veg1, snack2`
(newly confirmed in this task).

## Baseline (embedding-only, Evidence Recall@1/3/5/10, n=30)

| @1 | @3 | @5 | @10 |
|---|---|---|---|
| 0.333 (10/30) | 0.433 (13/30) | 0.633 (19/30) | 0.767 (23/30) |

7/30 questions have their real evidence confirmed absent from the Top-10 pool entirely — a hard
ceiling no reranking of that pool can fix (see "남은 한계" below).

## Method B — Dedup

Rule: normalize text (strip whitespace, unify dash variants), collapse a candidate into an
earlier one if: (a) normalized text is identical, (b) one is a substring of the other with
≥100 shared chars, or (c) `difflib.SequenceMatcher` ratio ≥ 0.5. Keep the higher-scoring
representative; re-rank survivors by original embedding score.

**Result: 0 near-duplicate chunks found across all 30 pools (300 candidates total).** Recall@1/3/5
identical to baseline (0.333 / 0.433 / 0.633). No improved or worsened questions.

Why: the KB's chunker (`FOOK_build_rag_kb.py`, `CHUNK_SIZE=450`, `CHUNK_OVERLAP=80`) only
overlaps *adjacent* chunks within the same source by ~80/450≈18% of characters — nowhere near
the 0.5 similarity threshold — and repeated facts across *different* chapters (e.g. prot1's
"<0.8 g/kg" limit appears at both pool rank 3 and rank 8) are phrased differently enough each
time that they are legitimately distinct evidence, not duplicates. Dedup is a no-op on this KB
at this pool size. Not harmful, but not useful either.

## Method C — Source-diversity cap

Rule: walk the Top-10 in score order; once a source has filled `cap` slots, defer further
same-source candidates to the tail (letting the next different-source candidate effectively
move up). Tried cap=2 and cap=3.

| variant | @1 | @3 | @5 | @10 | worsened (top-5 window) |
|---|---|---|---|---|---|
| cap=2 | 0.333 | 0.433 | 0.567 | 0.767 | k2 (5→7), water2 (4→8) |
| cap=3 | 0.333 | 0.433 | 0.600 | 0.767 | water2 (4→8) |

**No question was ever improved by this method; it only ever hurts.** Confirmed concrete
regression cases (exactly the risk the task asked to check for): k2's evidence chunk (source
`콩팥병_환자를_위한_안내서`) sits at pool rank 5, but that source already fills 2 of the first 4
slots, so cap=2 defers it behind a `만성콩팥병_환자_식사요법` chunk that was never in genuine
contention. water2 is worse: its evidence chunk (rank 4, `콩팥병_환자를_위한_안내서`) gets bumped
past by 4+ same-source chunks under both cap values. Recall@5 actively drops (0.633→0.567/0.600).
Diversity capping is net-negative here — this KB's genuinely-best answers are frequently
clustered in one source document (each source book covers a different question breadth), so
"diversity" is fighting the actual score signal, not correcting a real bias.

## Method D — Lexical bonus

Formula: `final_score = embedding_score + alpha * lexical_bonus(question, chunk_text)`, where
`lexical_bonus` = fraction of the question's content words (whitespace-split, common Korean
particle/sentence-ending suffixes stripped via a small curated list — no morphological
analyzer) found as a **substring** anywhere in the lightly-normalized (punctuation-stripped,
lowercased) chunk text. Bounded in [0, 1]. Embedding scores in this pool range ≈0.19–0.67.

| alpha | @1 | @3 | @5 | @10 | improved | worsened |
|---|---|---|---|---|---|---|
| 0.05 | 0.333 | **0.533** | 0.633 | 0.767 | fruit2 (10→2) | ckd2 (4→6) |
| 0.15 | 0.367 | **0.567** | **0.700** | 0.767 | p1 (6→4), water1 (9→3), fruit2 (10→1), early1 (9→5) | eat1 (5→6), ckd2 (4→6) |

alpha=0.05 is the only method in this whole experiment with a clean, meaningful Recall@3 gain
(+10pp) and only one regression. alpha=0.15 pushes further (Recall@5 +6.7pp beyond alpha=0.05)
but at the cost of a second regression and — more importantly — a real out-of-scope safety
issue (see below).

Regression mechanics, read from the actual chunk text:
- **ckd2** (both alphas): the promoted chunk is titled almost verbatim "콩팥이 하는 일이
  무엇인가요?" (question: "콩팥이 우리 몸에서 하는 일이 무엇인가요?") — genuinely on-topic
  content, just from `혈액투석_영양식생활관리_2권`, a source not in this question's (narrower,
  single-source) `expected_source` list. This is arguably a metric-labeling artifact more than a
  quality regression — the promoted answer may in fact be *better*, but I scored it as a
  regression to stay strict against the question set as written, per the instruction not to
  loosen judgment to chase a better number.
- **eat1** (alpha=0.15 only): a chunk about "하루에 허용된 수분량을 효과적으로 사용하는 방법"
  (water-budget tips, off-topic) gets boosted because it happens to contain the generic stripped
  tokens "어떻게"/"해야", pushing it just ahead of the real "외식하고 싶은데 무엇을 먹을 수
  있나요?" chapter chunk (rank 5→6). This is a genuine lexical-bonus false positive from a
  too-generic content word slipping past the (deliberately light) stopword list.

## Method E — Combination

Only dedup and the lexical bonus showed any signal (dedup = neutral no-op, lexical = the one
real winner); diversity-cap was net-negative in isolation, so per the instruction to only combine
elements that individually helped, it is included here **only as a documented negative control**,
not because it was expected to help.

| pipeline | @1 | @3 | @5 | @10 | worsened |
|---|---|---|---|---|---|
| dedup → lexical(0.05), no cap | 0.333 | 0.533 | 0.633 | 0.767 | ckd2 |
| dedup → lexical(0.05) → cap=2 | 0.333 | 0.500 | 0.567 | 0.767 | k2, water2, ckd2 |
| dedup → lexical(0.05) → cap=3 | 0.333 | 0.533 | 0.600 | 0.767 | water2, ckd2 |

Since dedup is a no-op here, `dedup → lexical(0.05), no cap` is numerically identical to plain
`lexical_a0.05` — confirming dedup adds nothing to combine. Adding either diversity-cap variant
on top strictly **reduces** Recall@5 and adds regressions, exactly reproducing Method C's
standalone problem inside the pipeline. No combination beats plain lexical(alpha=0.05) alone.

## 주요 실패 문항 Before→After (itch1 / prot2 / snack1 / transplant1 / fish1 + 신규 확인 veg1 / snack2)

All ranks below are position within the fixed Top-10 pool (or "—" = confirmed not in pool,
i.e. genuinely outside the 10 embedding candidates given to every method). **No method changes
any of these**, because none of them have their evidence anywhere in the pool to begin with —
reranking only reorders what's already retrieved.

| id | baseline | dedup | cap=2 | cap=3 | lex(0.05) | lex(0.15) | combo(best) | Top-10 pool status |
|---|---|---|---|---|---|---|---|---|
| itch1 | — | — | — | — | — | — | — | confirmed **not** in Top-10 (see below — true rank ~22) |
| prot2 | — | — | — | — | — | — | — | confirmed not in Top-10 (not even in Top-30, per prior diagnosis) |
| snack1 | — | — | — | — | — | — | — | confirmed not in Top-10 (full-corpus rank 224/539 pre-filter) |
| transplant1 | — | — | — | — | — | — | — | confirmed not in Top-10 (true rank ~18) |
| fish1 | — | — | — | — | — | — | — | not in Top-10; even its best full-corpus score (0.29) sits below RAG_MIN_SCORE=0.30, so production refuses regardless |
| veg1 (new) | — | — | — | — | — | — | — | confirmed not in Top-10; true full-corpus rank ≈30 (`혈액투석_영양식생활관리_2권`), score 0.353 |
| snack2 (new) | — | — | — | — | — | — | — | confirmed not in Top-10; true full-corpus rank 54 (`혈액투석환자를_위한_간식`), score 0.327 |

### itch1 ceiling — explicitly diagnosed as requested

Verified live on the current post-TOC-filter, post-live-retrieve pool: itch1's real evidence
chunk ("요독성 가려움증" section) is **not** among the 10 embedding candidates returned for this
query — all 10 candidates are generic dialysis/CKD-overview chunks, none containing the itch
explanation. This matches the prior diagnosis's ~rank 22–23 finding (pre-filter rank 23; the
TOC-filter rebuild removing 3 pure-TOC chunks, one of which used to occupy itch1's own pool at
old rank 4, very plausibly shifts it to ~22 post-filter, consistent with the task brief). **No
reranking method that only reorders a fixed Top-10 embedding pool can retrieve a chunk that was
never fetched into that pool in the first place.** This is a hard ceiling of the two-stage
"embed top-10, then rerank" design, not a reranking failure — fixing it needs either a larger
initial retrieval window (e.g. embed top-30, then rerank down to 5) or a query-rewrite step,
neither of which this task's mandate covers (the mandate explicitly says not to reintroduce
query rewriting here).

## Regression — full 30-question count per method (top-5 window: "was evidence in Top-5 before, is it still/now in Top-5")

| method | improved | worsened | unchanged | worsened questions (reason) |
|---|---|---|---|---|
| dedup | 0 | 0 | 30 | — |
| diversity_cap2 | 0 | 2 | 28 | k2 (4+ same-source chunks outrank it, deferred to rank 7); water2 (same mechanism, deferred to rank 8) |
| diversity_cap3 | 0 | 1 | 29 | water2 (deferred to rank 8 even at the looser cap) |
| lexical_a0.05 | 1 | 1 | 28 | ckd2 (promoted chunk is on-topic but from an out-of-list source — see Method D notes) |
| lexical_a0.15 | 4 | 2 | 24 | eat1 (generic "어떻게/해야" tokens boost an off-topic water-budget chunk); ckd2 (same as above) |
| combo dedup→lex0.05→cap2 | 1 | 3 | 26 | k2, water2 (cap mechanism), ckd2 (lexical mechanism) |
| combo dedup→lex0.05→cap3 | 1 | 2 | 27 | water2 (cap), ckd2 (lexical) |
| combo dedup→lex0.05 (no cap) | 1 | 1 | 28 | ckd2 — identical to plain lexical_a0.05 since dedup is a no-op |

## Out-of-scope safety check (n=21)

Checked every out-of-scope question's full Top-10 pool for whether the lexical bonus could push
a genuinely-irrelevant chunk's score across `RAG_MIN_SCORE=0.30` (the exact risk the task asked
to verify, not assume away).

- **alpha=0.05: zero gate flips.** No out-of-scope question's best achievable score (embedding +
  0.05×bonus, taken over all 10 candidates) crosses 0.30 when it wasn't already going to on
  embedding score alone.
- **alpha=0.15: 2 concrete gate flips**, both confirmed by reading the actual promoted chunk text:
  - `oos4` ("다음 주 로또 번호 추천해줘"): embedding top1 = 0.2719 (correctly below gate). The
    generic 2-character stripped token "다음" coincidentally substring-matches inside an unrelated
    CKD-symptom chunk's "**다음**과 같은 경우 즉시 의사를 만나야 합니다" — bonus 0.25 → after-score
    0.3094, **crosses the gate**. The system would attempt to answer a lottery-number request with
    a kidney-disease chunk instead of correctly refusing it.
  - `oos16` ("내일 비 오나요?"): embedding top1 = 0.2695 (below gate). A different candidate (not
    even the embedding top-1) picks up bonus 0.5 from the short generic tokens "내일"/"오나요"
    coincidentally appearing in unrelated body text → after-score 0.319, **crosses the gate**.
  This is the exact accidental-vocabulary-overlap risk the task flagged as worth checking for, and
  it is real at alpha=0.15, not merely theoretical.
- Diversity-cap and dedup only reorder/collapse the retrieved set — they never change any
  candidate's score, so they cannot by construction cause a gate flip (confirmed no effect since
  the gate decision only looks at `retrieve()`'s output, called identically for every method here).

**Conclusion: alpha=0.05 is safe on this eval set; alpha=0.15 is not** — it demonstrably
introduces false-positive "in-scope" classifications on clearly out-of-scope questions, which is
a real behavior risk for a medical-consultation service before it's even a retrieval-quality
question.

## 추천 구조 (Context-strategy comparison)

- **Strategy A (current production)**: retrieve top-5, use top-5 as context. Baseline Recall@5 =
  0.633.
- **Strategy B**: retrieve top-10, use top-10 as context, no reranking. Baseline Recall@10 =
  0.767 — a real +13.4pp gain over A purely from giving the LLM more candidates, with zero new
  API cost (same single embedding call, just don't truncate to 5). The tradeoff is prompt-token
  cost (roughly 2× the context chunks) and letting 5 more, lower-scoring, potentially irrelevant
  chunks reach the LLM — a distraction/noise risk for a medical-answer system that Strategy A
  avoids by design.
- **Strategy C**: retrieve top-10, apply the best lightweight method here (lexical bonus,
  alpha=0.05), use top 3–5 as context. Recall@3 = 0.533 (vs A's 0.433 — real gain, no added
  latency/cost) and Recall@5 = 0.633 (statistically the same as A's top-5, i.e. no loss), for
  **zero extra token cost over Strategy A** (still only 5 chunks in context) and zero extra API
  calls.

**Recommendation: Strategy C** (retrieve-10, lexical-bonus rerank at alpha=0.05, keep top-5 as
context) is the only option that improves ranking quality (Recall@3) without paying Strategy B's
token/noise cost and without the risk alpha=0.15 introduces. It is a genuine, if modest,
improvement over the current Strategy A at effectively no cost — subject to the "남은 한계"
caveat below that it cannot touch the 7 hard-ceiling questions.

## 최종 lightweight reranking 추천

**Primary recommendation: Method D, lexical bonus at alpha=0.05** (`final_score = embedding_score
+ 0.05 * lexical_bonus`), applied to the top-10 embedding pool with the final answer context
truncated to the top 3–5 of the reranked list.

Reasoning, weighed as instructed:
- **Quality**: the only method with a clear, real Recall@3 gain (0.433→0.533, +10pp / 3 more of
  30 questions) and Recall@5 flat-to-neutral (no loss), with exactly **one** regression (ckd2 —
  itself a source-labeling-artifact case, not a clear quality loss) against **one** genuine gain
  (fruit2, evidence promoted from rank 10 to rank 2 — meaningful, since rank 10 is the pool's
  last slot and would never have reached even a top-5 context in Strategy A).
- **Regression discipline**: alpha=0.15 buys more Recall@5 (+6.7pp beyond alpha=0.05) but doubles
  the regression count AND introduces 2 concrete out-of-scope gate flips — a real safety concern
  for a medical-consultation bot, not just a metrics tradeoff. Per the instruction to weight
  regressions heavily even against a higher average, alpha=0.05 is the correct choice over
  alpha=0.15.
- **Complexity/cost/determinism**: pure Python string operations on already-fetched text, zero
  new API calls, zero added latency worth measuring, fully deterministic and reproducible (no
  randomness, no external model).
- **Simplicity tie-break**: dedup and diversity-cap are equally cheap to implement but showed no
  benefit (dedup) or net harm (diversity-cap) — not recommended, not even as harmless additions,
  since diversity-cap actively regresses two questions for zero offsetting gain.

**Optional second choice: keep as-is (do nothing).** Given the gain is modest (+10pp Recall@3 on
a 30-question eval set — 3 questions), the one regression's debatable status, and that this KB
and eval set will keep evolving, "ship nothing yet, revisit lexical bonus once the eval set is
larger" is a legitimate, non-overselling second choice for a team that wants to be conservative
about touching a clinical-context retrieval pipeline. If forced to pick one lightweight change
to make, it is the lexical bonus at alpha=0.05; if the bar is "meaningfully, robustly better,"
this experiment's honest conclusion is that the gain is real but modest, not transformative.

## 남은 한계

- **7 of 30 positive questions (itch1, prot2, snack1, fish1, transplant1, veg1, snack2) have their
  real evidence confirmed absent from the Top-10 embedding pool entirely.** No lightweight
  reranking method tested here — or any reranking method restricted to reordering a fixed Top-10
  — can fix these; the ceiling is in the embedding retrieval step itself, not the ranking step.
  Raising `top_k` at retrieval time (e.g. to 20–30) before any reranking, or a query-rewrite step
  (shown in prior isolated work to move itch1's evidence to rank 1 — explicitly NOT reintroduced
  here per this task's scope), are the two directions that could actually close these gaps. Both
  are separate, future work, not lightweight-reranking work.
- `fish1` is additionally safe by construction regardless: its best obtainable score across the
  full 536-chunk corpus (≈0.29) sits below `RAG_MIN_SCORE=0.30`, so production already refuses to
  answer it rather than risk a wrong answer — the miss is real but not a safety exposure.
- A scope-classifier/gate was explicitly out of scope for this task and was not built or evaluated
  here; the out-of-scope safety findings above are about the existing score-threshold gate's
  interaction with reranking, not a recommendation to add a new gate.
- The `ckd2` "regression" under lexical reranking is flagged honestly as a probable eval-set
  labeling artifact (the promoted chunk directly answers the question but is from a source not
  in that question's `expected_source` list) rather than a true content-quality loss — worth a
  human relooking at `ckd2`'s `expected_source` list the same way `na1`/`na2`/`ckd1`/`bone1`/
  `kfruit1`/`water2`/`fish1` were already widened in the v2 eval set's changelog entries, but
  that is an eval-set maintenance task, not a reranking-method decision, and out of this task's
  scope to change.
