# Evidence-Miss Root-Cause Diagnosis — supplementary notes (2026-08-14)

Companion to `evidence_miss_diagnosis.py` / `evidence_miss_diagnosis_results.json`.
Live top-30 ranks for prot2/transplant1/itch1 are in the JSON. This file records the
wider (top-200) searches, KB-json chunk inspections, and the TOC-noise census that were
done by direct script calls but not captured in the JSON.

## p2 correction (NOT a miss — eval-harness truncation artifact)

`backend/evaluation/topk_experiment_raw.json` stores each chunk's `text` field truncated
to ~300-800 chars. p2's rank-1 chunk in that file is cut off before reaching the "숨은 인"
sentence. A live `retrieve()` call returns the SAME rank-1 chunk (score 0.4064) at its full
1320-char length, which DOES contain:
  "가공 식품 속의 인은 거의 100% 흡수됩니다 ... '숨은 인'이라 불리며 700-800 mg/일 정도"
So p2 is a Recall@1 HIT in production; it only looked like a miss because of the raw-JSON
text truncation, not because of any retrieval or KB problem. Excluded from the final miss
list on this basis.

## snack1 — wide search (top_k=200, then full 539-chunk scan)

Target chunk: KB json index 527 (source=혈액투석환자를_위한_간식, 449 chars), the recipe
card containing 슈가토스트/러스크/사과조림. NOT found within top-200 for the original
question. Full-corpus rank/score computed directly against all 539 chunk embeddings:

  rank = 224 / 539, score = 0.3634   (top1 score = 0.6068, gap = 0.2434)

Rewrite tried: "혈액투석 환자를 위한 간식 만드는 법 (재료와 조리법)"
  rank = 283 / 539, score = 0.3919   (WORSE rank despite higher raw score — top1 for the
  rewrite scored even higher at 0.6572, so the gap widened). This shows the miss is not
  primarily a wording problem — see root-cause section below.

## fish1 — wide search (top_k=200)

Target: the food-pyramid chunk with "고기, 생선, 달걀, 콩류(매일 3-4회 정도 섭취 권장)".
  rank = 65 / 539, score = 0.2208 (below RAG_MIN_SCORE=0.30 even at that rank; top1 score
  0.2875 is also below threshold, so production safely refuses either way).
This is the KB's only concrete fish-related guidance and it is a frequency-based mention of
"생선" as one word inside a 5-food-group list, not a fish-specific answer — see fish1 deep
dive in the main report for the eval-question-validity discussion.

## TOC/heading-noise chunk census

Heuristic: chunks in `data/FOOK_rag_kb.json` (539 chunks total) matching >=4 occurrences of
the pattern `\d{1,2}\.\s*<text>\s\d{1,3}` (numbered section title + trailing page number) —
i.e. table-of-contents-style chunks.

  19 / 539 chunks (3.5%) match this TOC heuristic.
  11 / 539 chunks (2.0%) are under 200 chars (near-empty/boilerplate fragments).
  chunk length: min=38, max=1874, mean=571, median=450 chars.

Three of these TOC chunks (KB indices 375, 376, 377) are sequential fragments of the SAME
table of contents for 혈액투석_영양식생활관리_2권 (chunked purely by character count with
no TOC-aware filtering at KB-build time). Chunk 377 contains the line:
  "40. 몸이 너무 가려워요. (요독성 가려움증)\n139"
with zero explanatory body text — this is the chunk that ranks #4 (score 0.4206) in itch1's
original top-10, ahead of the real explanation (which never appears until rank 23).

Recurrence check — these same TOC chunks (375/376/377, or their chunk-boundary neighbors)
were found competing in the top-10/top-30 of THREE OTHER questions during this
investigation, not just itch1:
  - prot1 (not a Recall@10 miss, but hit@1=False): the TOC-tail+body chunk that opens
    with "40. 몸이 너무 가려워요...41. 제가 먹는 약이..." is prot1's WRONG rank-1 result
    (score 0.5699), ahead of the real prot1 evidence.
  - prot2 (confirmed miss): same TOC-tail chunk appears at rank 7 (score 0.5300) in top-30.
  - snack1 (confirmed miss): a different fragment of the same TOC (chunk 376, "25. 콩팥이
    하는 일이 무엇인가요? ... 26. 콩팥기능을 확인하려면...") appears at rank 25 (score
    0.5153) in top-30.
So heading/TOC noise is a systemic, recurring contaminant, not an itch1-only artifact.

## prot2 chunk-boundary detail

Two separate KB chunks contain the "1.0-1.2 g/kg/day" fact (indices 318 and 322), both from
콩팥병_환자를_위한_안내서. Chunk 322 is representative of the boundary problem: it opens
mid-sentence with general CKD dietary-restriction bullets ("고단백식이 요법은 만성콩팥병
환자에서 피해야 합니다..."), then pivots to the opposite guidance for dialysis patients
("그러나, 투석 중인 환자는 단백질 섭취량을 1.0-1.2 g/kg/day로 늘려야 합니다."), then
immediately pivots again into an unrelated water-intake section ("3. 수분 섭취..."). The
"increase protein on dialysis" sentence — the one fact prot2 needs — is a single sentence
sandwiched between two topically different passages, diluting the chunk's embedding
centroid away from "협 dialysis protein amount."

## transplant1 chunk-boundary detail

Real evidence (콩팥병_환자를_위한_안내서, line ~2812: "이식 초기에 하루에 3 L 이상의 물을
필요로 할 수 있습니다") sits inside a long post-transplant lifestyle checklist chunk that
also covers diet regularity, exercise restrictions, sexual activity timing, smoking/alcohol,
and infection-precaution advice in the same bullet list. The water-intake bullet is one line
among ~10 unrelated bullets, and the top of the ranking is dominated by OTHER 이식-themed
chunks (donor/recipient matching procedures, KONOS approval process) that share heavy
"이식" vocabulary — rank 18/30, score 0.3569 vs top1 0.4175 (gap 0.0607, the smallest gap
of the five confirmed misses).
