# -*- coding: utf-8 -*-
"""Deterministic (keyword/pattern-based, NOT LLM) risk classifier for the "should a
IN_SCOPE + RAG-relevance-gate-failed question fall back to general ChatGPT-style
knowledge?" evaluation (see backend/evaluation/ task notes, 2026-08-14).

EVALUATION-ONLY: this module is not imported by FOOK_rag_chatbot.py and is not wired
into production in any way. It exists purely to test whether a deterministic risk
classifier COULD safely gate a hypothetical future fallback feature.

Scope of input this classifier is designed for: it only ever needs to see questions
that have ALREADY been judged IN_SCOPE by classify_scope() AND have ALREADY failed the
RAG_MIN_SCORE relevance gate (i.e. would currently get NO_EVIDENCE_ANSWER). It is not a
general-purpose scope classifier and is never meant to see OUT_OF_SCOPE questions in
production. For this evaluation script it is run directly over the full eval set so its
accuracy can be measured in isolation from the scope gate.

Design:
  - No hardcoded exact-sentence matching. All signals are substring/phrase patterns,
    matched the same way classify_scope() matches DOMAIN_TERMS/RED_FLAG_TERMS/etc:
    normalize (strip punctuation/whitespace) then `term in question`.
  - Reuses classify_scope()'s own RED_FLAG_TERMS and DIALYSIS_PROCEDURE_TERMS lists
    (imported, not copied) so a question that would independently trip the EXISTING
    scope gate's medical/procedure red flags always also trips MEDICAL_HIGH_RISK here
    — the two lists must never diverge.
  - Priority order (most restrictive signal wins, mirroring classify_scope()'s style):
      1) MEDICAL_HIGH_RISK  (dominates if ANY high-risk signal present)
      2) NUTRITION_DECISION (else, if any quantity/allowance phrasing present)
      3) GENERAL_LOW_RISK   (else — everything remaining)
    Ambiguous/borderline cases must resolve to the more restrictive bucket, never to
    GENERAL_LOW_RISK, by construction of this ordering.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse production's scope-gate red-flag/procedure term lists and normalization style —
# do not duplicate/diverge them (see module docstring). This is a read-only import;
# nothing in FOOK_rag_chatbot.py is modified or monkeypatched.
from FOOK_rag_chatbot import RED_FLAG_TERMS, DIALYSIS_PROCEDURE_TERMS, _scope_norm

MEDICAL_HIGH_RISK = 'MEDICAL_HIGH_RISK'
NUTRITION_DECISION = 'NUTRITION_DECISION'
GENERAL_LOW_RISK = 'GENERAL_LOW_RISK'

# ── Own MEDICAL_HIGH_RISK signals (beyond the reused RED_FLAG_TERMS/DIALYSIS_PROCEDURE_
# TERMS lists above) — medication, treatment, diagnosis, lab-value interpretation,
# dialysis-frequency/schedule, and emergency/symptom-escalation language. Substring
# patterns, not exact sentences.
MEDICAL_HIGH_RISK_TERMS = [
    # lab values / diagnosis
    '검사수치', '수치', '진단', '크레아티닌', '위험한가요', '위험할까요', '위험해',
    # medication / prescription / dosing
    '약', '복용', '처방', '단위',
    # treatment
    '치료',
    # symptoms / emergency escalation
    '증상', '응급', '병원', '통증', '아프', '어지러', '답답', '숨쉬기', '부었', '저혈당',
    '가려운', '가려움', '가려워',
    # dialysis frequency/schedule phrasing not already covered by DIALYSIS_PROCEDURE_TERMS
    # ('주 몇 번'/'몇 시간씩' assume a terser register; real questions often phrase this as
    # "일주일에 몇 번"/"몇 번 받아야 하나요" instead)
    '몇 번 받아야', '일주일에 몇 번',
]

# ── NUTRITION_DECISION signals — asks for a specific allowed amount/quantity/frequency.
# This is exactly the category the app's grounded RAG / food-DB numeric path exists for;
# it must never be waved through to an ungrounded general-AI answer.
# Verb forms mirror find_food()'s own FOOD_QUESTION_KW register in FOOK_rag_chatbot.py
# ('먹어도','먹을','먹으면','드셔도','드시면','드셔','얼마나 먹') — deliberately reused
# rather than reinvented, since that list already encodes the honorific register real
# patients/caregivers use when asking this app "how much can I eat" questions (found via
# a live spot-check against fish1 = "생선은 얼마나 드셔도 되나요?", a real production
# no-evidence case today: without '드셔도' this classifier originally mis-labeled it
# GENERAL_LOW_RISK).
#
# A second live spot-check (against prot2 = "투석을 받고 있는 환자는 단백질을 얼마나
# 섭취해야 하나요?" and transplant1 = "신장이식을 받은 직후에는 물을 얼마나 마셔야
# 하나요?", both real questions from evaluation/rag_eval_questions_v2.json) found that
# enumerating specific "얼마나 <verb>" combinations still misses phrasings where "얼마나"
# and the eat/drink verb aren't adjacent ("단백질을 얼마나 섭취해야" / "물을 얼마나
# 마셔야"). Rather than keep enumerating verb pairs, bare '얼마나' ("how much/how many")
# is included directly — checked to not appear in ANY Group A (GENERAL_LOW_RISK) question
# in fallback_eval_set.json, and safe by construction even if it over-fires on a stray
# non-quantity "얼마나" elsewhere, since a false GENERAL_LOW_RISK->NUTRITION_DECISION
# reclassification only makes the classifier MORE conservative (blocks fallback), which
# is the tolerated direction of error in this design (see module docstring).
NUTRITION_DECISION_TERMS = [
    '몇 g', '몇 mg', '몇 mL', '몇 개', '몇 회', '몇 잔', '몇 쪽', '몇 조각',
    '얼마나',
    '먹어도', '먹을', '먹으면', '마셔도', '드셔도', '드시면', '드셔',
    '괜찮', '허용량', '제한량', '섭취량',
]


def classify_fallback_risk(question):
    """Return one of MEDICAL_HIGH_RISK / NUTRITION_DECISION / GENERAL_LOW_RISK.
    Priority: MEDICAL_HIGH_RISK > NUTRITION_DECISION > GENERAL_LOW_RISK — the most
    restrictive matching signal always wins, so ambiguous/borderline questions never
    fall through to GENERAL_LOW_RISK by accident."""
    q = _scope_norm(question)

    has_reused_redflag = any(t in q for t in RED_FLAG_TERMS)
    has_reused_procedure = any(t in q for t in DIALYSIS_PROCEDURE_TERMS)
    has_own_medical = any(t in q for t in MEDICAL_HIGH_RISK_TERMS)
    if has_reused_redflag or has_reused_procedure or has_own_medical:
        return MEDICAL_HIGH_RISK

    if any(t in q for t in NUTRITION_DECISION_TERMS):
        return NUTRITION_DECISION

    return GENERAL_LOW_RISK
