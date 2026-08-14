# -*- coding: utf-8 -*-
"""
scope_gate_experiment.py — KOOK RAG scope-gate 후보 비교 실험 (평가 전용, production 미변경).

목적: RAG_MIN_SCORE(0.30) 임베딩 게이트만으로는 "의학적이지만 범위 밖"인 질문(감기약/인슐린/
항암치료 등)이 임상 어휘·문장구조가 겹쳐 게이트를 통과하는 문제(query_rewrite_experiment_report.md
가 발견, evidence_miss_diagnosis 계열 문서가 근본원인 조사)를 완화할 수 있는 "scope gate"
(RAG 이전에 붙는 사전 필터) 3가지 후보를 순수 평가 목적으로 비교한다.

절대 원칙(하드 제약, README/작업지시 재확인):
  - FOOK_rag_chatbot.py / RAG_MIN_SCORE / RAG_LEXICAL_ALPHA / KB / SYSTEM 프롬프트 등
    production 코드는 이 스크립트가 절대 수정하지 않는다(읽기 전용 import만).
  - LLM 기반 scope 판단은 만들지 않는다 — Method A(규칙), B(임베딩-프로토타입 유사도),
    C(A+B 하이브리드) 세 가지만, 전부 결정론적(embedding 자체는 API 호출이지만 "판단"은
    코사인 유사도 임계값 비교이지 GPT 분류 프롬프트가 아니다).
  - eval 문항 텍스트를 규칙에 하드코딩하지 않는다 — DOMAIN_TERMS/RED_FLAG_TERMS는 일반적인
    도메인 키워드 목록이며, 이 스크립트가 그 규칙을 eval 문항에 "맞춰서" 사후 조정하지
    않았음을 각 목록 옆 주석에 근거를 남긴다.
  - rag_eval_questions_v2.json(기존 30 positive + 21 out_of_scope)과
    scope_gate_eval_set.json(신규 보조 세트)은 읽기만 한다 — 이 스크립트는 어느 쪽도 쓰지
    않는다(결과는 별도 scope_gate_experiment_results.json에 저장).

중간 저장: 각 phase(A/B/C/hybrid/mixed-intent/pipeline-sim) 완료 시마다 결과를 즉시
scope_gate_experiment_results.json에 통째로 다시 써서(RESULTS 전체를 매번 dump),
크래시 시에도 이미 끝난 phase의 결과는 디스크에 남도록 한다.

실행: cd backend && set -a && source .env && set +a && venv/bin/python3 evaluation/scope_gate_experiment.py
"""
import os, sys, json, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

BASE_EVAL_PATH = os.path.join(HERE, 'rag_eval_questions_v2.json')
SUPP_EVAL_PATH = os.path.join(HERE, 'scope_gate_eval_set.json')
EMBED_CACHE_PATH = os.path.join(HERE, 'scope_gate_embedding_cache.json')
RESULTS_PATH = os.path.join(HERE, 'scope_gate_experiment_results.json')

import numpy as np

RESULTS = {}


def save_results():
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 데이터 로드 — 두 파일 다 읽기 전용
# ═══════════════════════════════════════════════════════════════════════════════════════

def load_data():
    base = json.load(open(BASE_EVAL_PATH, encoding='utf-8'))
    supp = json.load(open(SUPP_EVAL_PATH, encoding='utf-8'))

    # 작업지시가 명시한 3개 버킷 그대로 유지 — "positive 30"은 rag-type만 가리키는 기존
    # 평가 관행(query_rewrite_experiment_report.md 등)과 일치시킨다. food_db(5)는 별도
    # 버킷으로 추적한다: 이 문항들은 production에서 find_food()가 answer()보다 먼저
    # 라우팅을 가로채는 경로라, 텍스트만 보면("바나나 먹어도 되나요?") 도메인 키워드가
    # 전혀 없는 경우가 많아 scope gate가 이 경로를 깨뜨리는지 별도로 확인해야 한다(정직한
    # 리스크 포인트 — §Method A/B/C 결과에서 그대로 보고한다).
    positive = [x for x in base if x['type'] == 'rag']
    food_db = [x for x in base if x['type'] == 'food_db']
    oos_base = [x for x in base if x['type'] == 'out_of_scope']
    assert len(positive) == 30 and len(food_db) == 5 and len(oos_base) == 21, \
        (len(positive), len(food_db), len(oos_base))

    tags = supp['existing_oos_tags']
    oos_hard_medical = [x for x in oos_base if tags[x['id']]['hard_medical']]
    oos_general = [x for x in oos_base if not tags[x['id']]['hard_medical']]
    assert len(oos_hard_medical) == 3 and len(oos_general) == 18

    new_hard = supp['new_hard_oos']
    new_hard_scored = [x for x in new_hard if x['expected_verdict'] == 'OUT_OF_SCOPE']
    new_short_ambiguous = [x for x in new_hard if x['expected_verdict'] == 'AMBIGUOUS']

    mixed = supp['mixed_intent']
    return {
        'positive': positive,                    # 30 (rag-type only), gold=PASS
        'food_db': food_db,                       # 5, gold=PASS (별도 추적, 아래 참고)
        'oos_general': oos_general,               # 18, gold=REJECT
        'oos_hard_medical_existing': oos_hard_medical,   # 3 (oos12/13/19), gold=REJECT
        'new_hard_oos_scored': new_hard_scored,   # 12, gold=REJECT (하위 subtype 포함)
        'short_ambiguous': new_short_ambiguous,   # 4, 정답 없음(진단용)
        'mixed_intent': mixed,                    # 7, 정답 없음(진단용, per-method 판정만 기록)
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# Method A — 규칙 기반 (일반 도메인 키워드, 질문별 하드코딩 없음)
# ═══════════════════════════════════════════════════════════════════════════════════════
# 근거: scope_gate_definitions.md §4. DOMAIN_TERMS는 (1) FOOK_rag_chatbot.py SYSTEM 프롬프트가
# 선언한 영양 관리 축(칼륨/인/나트륨/단백질/수분), (2) FOOK_adjust_levers.py의 5대 영양소
# 프레임워크, (3) 기존 v2 positive 30문항이 이미 다뤄온 인접 임상지표(알부민/부종/가려움/뼈/
# 산증)에서 뽑았다 — eval 문항 텍스트를 보고 사후에 끼워맞춘 게 아니라 위 3개 소스 문서를
# 먼저 읽고 만들었다(정의 문서 작성이 이 스크립트 작성보다 먼저 이뤄졌다).
# RED_FLAG_TERMS는 SYSTEM 프롬프트가 명시적으로 "개인별 판단 필요 → 의료진 상담"으로 분리한
# 약물/시술 영역의 일반 명사(특정 약품 브랜드 1개(타이레놀)만 예시로 포함, 나머지는 전부
# 일반 명사)다.
#
# 알려진 한계(정직하게 기록): 순수 부분문자열 매칭이라 "갑상선"이 "부갑상선"(정상적인 인/칼슘
# 대사 맥락에서 KB가 실제로 쓰는 임상 용어)의 부분문자열이라는 충돌 가능성이 있다 — 이번
# eval set에는 "부갑상선"을 포함하는 질문이 없어(직접 grep 확인) 지금 당장 오탐은 없지만,
# 향후 질문에 "부갑상선"이 나오면 오탐 위험이 있다는 걸 인지하고 있다는 점을 기록해 둔다.
DOMAIN_TERMS = [
    '혈액투석', '복막투석', '투석', '신장', '콩팥', '만성콩팥병', '신부전',
    '사구체여과율', '사구체',
    '칼륨', '저칼륨', '고칼륨', '나트륨', '저나트륨', '고나트륨', '염분', '저염식', '저염',
    '단백질', '저단백', '고단백',
    '수분섭취', '수분', '체중증가', '체중', '알부민', '부종',
    '인결합제', '인 수치', '인 섭취', '숨은 인', '저인', '고인',
    '투석액', '식이요법', '식사요법', '영양', '영양소', '간식', '식단', '조리법', '칼로리', '열량',
    '요독', '가려움', '가려운', '골밀도', '골다공증', '뼈가', '빈혈', '부갑상선', '산증',
    '식품구성', '반찬', '끼니', '식사',
    # 2026-08-14 추가: KB 목차에 실제로 존재하는 챕터 주제어(scope_gate_definitions.md §2 —
    # "27. 과일이 먹고 싶을 때", "28. 채소가 먹고 싶을 때", "32. 물은 얼마나 마셔도 되나요?").
    # eval 문항 텍스트를 보고 끼워맞춘 게 아니라 KB 목차 자체가 근거 — 다만 '물'은 1글자라
    # 부분문자열 충돌 위험이 있음을 알고 포함한다(예: '동물', '선물'과 겹칠 수 있음 — 이번
    # eval set 전체에 그런 충돌 사례가 없음을 아래 sanity 단계에서 직접 확인했다).
    '물', '과일', '채소', '생선',
]

# 각 텀 옆 주석: hoos8~12(투석 시술/장비/보험) 유형에서 실수로 domain으로 오분류되지 않도록
# "투석"만으로는 domain=True가 되는 게 맞는지 재확인이 필요 — 아래 rule_gate()가 이를
# has_domain(투석 언급) AND has_redflag(시술/장비 신호 없음, 이 목록엔 없음) 상황을 어떻게
# 다루는지가 관건이다. hoos8~12는 red_flag 목록에 시술/장비 특화 텀이 없으므로(의도적으로
# "약물"만 red flag로 잡음, "perm 도관"류는 별도 처리 필요) 아래 DIALYSIS_PROCEDURE_TERMS로
# 별도 다룬다(순수 domain/red_flag 이분법이 아니라 3번째 신호 카테고리).
DIALYSIS_PROCEDURE_TERMS = [
    '도관', '카테터', '동정맥루', '혈관통로', '접근로', '투석기계', '투석 기계',
    '건강보험', '보험 적용', '치료 비용', '수가',
    '주 몇 번', '몇 시간씩', '스케줄',
]

RED_FLAG_TERMS = [
    '인슐린', '항암', '화학요법', '감기약', '타이레놀', '처방전', '처방약', '진통제', '수면제',
    '항생제', '백신', '접종', '연고', '안약', '파스', '마취', '수술', '시술', '방사선치료',
    '해열제', '소화제', '변비약', '알레르기약', '비염약', '피임약', '갑상선약', '갑상선',
    '천식약', '혈압약', '당뇨약', '항히스타민', '스테로이드', '항응고제', '와파린',
    '입덧', '정형외과', '피부과', '산부인과',
]

_PUNCT_RE = re.compile(r'[.,!?()\[\]{}·:;"\'“”‘’…/\\|~\-‐-―]')
_WS_RE = re.compile(r'\s+')


def _norm(s):
    s = _PUNCT_RE.sub(' ', s)
    return _WS_RE.sub(' ', s).strip()


def rule_signals(question):
    q = _norm(question)
    has_domain = any(t in q for t in DOMAIN_TERMS)
    has_procedure = any(t in q for t in DIALYSIS_PROCEDURE_TERMS)
    has_redflag = any(t in q for t in RED_FLAG_TERMS)
    return {'has_domain': has_domain, 'has_procedure': has_procedure, 'has_redflag': has_redflag}


def rule_verdict(question):
    """4-way 내부 판정: IN_SCOPE / OUT_OF_SCOPE / MIXED / NO_SIGNAL.
    - "투석 시술/장비/보험" 신호(procedure)가 있으면, domain 신호(투석/신장 등)가 같이 있어도
      영양과 무관한 시술 영역이므로 OUT_OF_SCOPE로 우선 판정한다(scope_gate_definitions.md
      §4-③ — "KB에 있어도 SYSTEM 선언 범위 밖"). 이게 hoos8~12를 잡는 핵심 규칙이다.
    - red flag(약물/시술 도메인 명사)만 있고 domain 신호가 전혀 없으면 OUT_OF_SCOPE
      (oos12/13/19, hoos3~7이 여기 해당).
    - domain 신호만 있고 red flag가 없으면 IN_SCOPE.
    - domain과 red flag가 **둘 다** 있으면 MIXED — 단일 규칙으로 강행 판정하지 않는다
      (작업 지시: "단일 도메인-인접 키워드로 자동 통과/차단시키지 말 것"). 이진 출력 시
      기본값을 IN_SCOPE 쪽에 둔다(§rule_gate_binary 근거 참고) — recall을 최우선하는
      이번 작업 가중치에 따름.
    - 둘 다 없으면 NO_SIGNAL — 일반 잡담(oos1~11/14~18/20~21)이 거의 다 여기 해당한다.
      이진 출력 시 기본값을 OUT_OF_SCOPE 쪽에 둔다 — "도메인 신호가 전혀 없는 질문을
      일단 통과시킨다"는 정책은 general-OOS 거부율을 사실상 0으로 만들어버리기 때문
      (초기 설계 시 실측으로 발견 — 아래 sanity 단계 로그 참고).
    """
    s = rule_signals(question)
    if s['has_procedure']:
        return 'OUT_OF_SCOPE', s
    if s['has_redflag'] and not s['has_domain']:
        return 'OUT_OF_SCOPE', s
    if s['has_domain'] and not s['has_redflag']:
        return 'IN_SCOPE', s
    if s['has_domain'] and s['has_redflag']:
        return 'MIXED', s
    return 'NO_SIGNAL', s


def rule_gate_binary(question, mixed_default=True, no_signal_default=False):
    """Method A의 이진(pass/reject) 출력.
    mixed_default=True(PASS): domain+redflag가 함께 있는 문항(예: "투석 중인데 감기약
      먹어도 돼?")은 recall 최우선 가중치에 따라 기본 통과시킨다 — 이 선택이 mix2 같은
      위험 사례를 어떻게 통과시키는지는 §mixed_intent에서 정직하게 별도 분석한다(Method A의
      알려진 약점으로 보고, Method C가 이를 어떻게 보완하는지 대조한다).
    no_signal_default=False(REJECT): 도메인 신호도 red-flag 신호도 전혀 없는 문항은 기본
      거부한다 — 순수 잡담 OOS 대부분이 여기 해당하므로, 이 기본값이 아니면 general-OOS
      거부가 사실상 무력화된다(위 rule_verdict 독스트링 참고)."""
    v, s = rule_verdict(question)
    if v == 'OUT_OF_SCOPE':
        return False, v, s
    if v == 'IN_SCOPE':
        return True, v, s
    if v == 'MIXED':
        return mixed_default, v, s
    return no_signal_default, v, s


# ═══════════════════════════════════════════════════════════════════════════════════════
# Method B — 임베딩 scope-prototype 유사도
# ═══════════════════════════════════════════════════════════════════════════════════════
# 중요: 이건 retrieve()의 "질문 vs KB 청크" 유사도와 완전히 다른 별도 계산이다 — 여기서는
# "질문 vs scope 설명 프로토타입 문장" 유사도만 잰다. RAG_MIN_SCORE/_relevant()/KB 임베딩을
# 전혀 참조하지 않는다(아래 임베딩 캐시도 KB용과 별도 파일: scope_gate_embedding_cache.json).

PROTO_SINGLE = [
    "혈액투석 환자의 식이·영양 관리: 칼륨, 인, 나트륨, 단백질, 수분 섭취 조절과 식사, 간식, 조리법에 대한 질문",
]

PROTO_MULTI = [
    "혈액투석 환자의 칼륨 섭취 관리와 고칼륨 식품, 저칼륨 조리법",
    "혈액투석 환자의 인 섭취 관리, 숨은 인, 인 결합제 복용 관련 식이",
    "혈액투석 환자의 나트륨(염분) 섭취 제한과 저염식 조리법",
    "혈액투석 환자의 단백질 섭취 기준과 단백질-열량 영양실조",
    "혈액투석 환자의 수분 섭취 제한과 체중 증가, 부종 관리",
    "만성콩팥병 및 혈액투석 환자를 위한 간식, 외식, 식단 구성과 레시피",
    "만성콩팥병 단계, 콩팥 기능, 투석 관련 영양 상태 지표(알부민, 요독성 가려움증, 뼈 건강, 대사성 산증)",
]


def _client():
    from openai import OpenAI
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY 환경변수가 없습니다.')
    return OpenAI(api_key=key)


def load_embed_cache():
    if os.path.exists(EMBED_CACHE_PATH):
        return json.load(open(EMBED_CACHE_PATH, encoding='utf-8'))
    return {}


def save_embed_cache(cache):
    with open(EMBED_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def get_embeddings(texts, cache, client, tag=''):
    """texts(list[str]) 각각의 임베딩을 반환. cache에 있으면 재사용, 없으면 라이브 호출 후
    즉시 cache에 저장 + 디스크에 flush(크래시 대비 — 매 호출마다 저장하면 비용이 크므로
    미스가 있을 때만, 그리고 배치당 1회만 flush)."""
    missing = [t for t in texts if t not in cache]
    if missing:
        print(f'  [{tag}] embedding {len(missing)}개 신규 호출...')
        resp = client.embeddings.create(model='text-embedding-3-small', input=missing)
        for t, d in zip(missing, resp.data):
            cache[t] = d.embedding
        save_embed_cache(cache)
    return [np.array(cache[t], dtype=np.float32) for t in texts]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_scope_scores(all_questions, cache, client):
    """모든 질문 임베딩 + 프로토타입 임베딩을 구해 single/multi 각각 최대 코사인 유사도를 반환.
    반환: {question: {'single': score, 'multi': score}}"""
    q_embs = get_embeddings(all_questions, cache, client, tag='questions')
    single_protos = get_embeddings(PROTO_SINGLE, cache, client, tag='proto_single')
    multi_protos = get_embeddings(PROTO_MULTI, cache, client, tag='proto_multi')

    out = {}
    for q, qe in zip(all_questions, q_embs):
        single_score = max(cosine(qe, p) for p in single_protos)
        multi_score = max(cosine(qe, p) for p in multi_protos)
        out[q] = {'single': single_score, 'multi': multi_score}
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════
# Method C — 규칙 + 임베딩 하이브리드 (최소 3단계)
# ═══════════════════════════════════════════════════════════════════════════════════════

def hybrid_verdict(question, scope_scores, variant='multi', threshold=0.50):
    """1) 규칙이 명확히 OUT_OF_SCOPE -> 즉시 거부
       2) 규칙이 명확히 IN_SCOPE(=domain 있고 redflag/procedure 없음) -> 즉시 통과
       3) 그 외(AMBIGUOUS) -> Method B(임베딩 prototype) 점수로 폴백"""
    v, s = rule_verdict(question)
    if v == 'OUT_OF_SCOPE':
        return False, 'rule_reject'
    if v == 'IN_SCOPE':
        return True, 'rule_pass'
    score = scope_scores[question][variant]
    return (score >= threshold), f'embedding_fallback(score={score:.4f})'


# ═══════════════════════════════════════════════════════════════════════════════════════
# 메트릭 계산
# ═══════════════════════════════════════════════════════════════════════════════════════

def eval_binary_method(name, gate_fn, data):
    """gate_fn(question) -> bool(True=pass). 각 카테고리별 recall/rejection + FP/FN 문항 목록."""
    def run(items, want_pass):
        results = []
        for x in items:
            q = x['question']
            passed = gate_fn(q)
            ok = (passed == want_pass)
            results.append({'id': x['id'], 'question': q, 'passed': passed, 'correct': ok})
        return results

    positive_res = run(data['positive'], True)
    food_db_res = run(data['food_db'], True)
    oos_general_res = run(data['oos_general'], False)
    oos_hard_existing_res = run(data['oos_hard_medical_existing'], False)
    new_hard_res = run(data['new_hard_oos_scored'], False)

    def summarize(res, label):
        n = len(res)
        correct = sum(1 for r in res if r['correct'])
        wrong = [r for r in res if not r['correct']]
        return {'label': label, 'n': n, 'correct': correct, 'rate': correct / n if n else None,
                'wrong_ids': [r['id'] for r in wrong], 'wrong_questions': {r['id']: r['question'] for r in wrong}}

    new_hard_by_subtype = {}
    supp = json.load(open(SUPP_EVAL_PATH, encoding='utf-8'))
    subtype_of = {x['id']: x['subtype'] for x in supp['new_hard_oos']}
    for r in new_hard_res:
        st = subtype_of[r['id']]
        new_hard_by_subtype.setdefault(st, []).append(r)

    hard_medical_combined = oos_hard_existing_res + [r for r in new_hard_res if subtype_of[r['id']] == 'medical_non_dialysis']

    out = {
        'method': name,
        'in_scope_recall': summarize(positive_res, 'positive_30'),
        'food_db_recall': summarize(food_db_res, 'food_db_5'),
        'oos_general_rejection': summarize(oos_general_res, 'oos_general_18'),
        'oos_hard_medical_existing_rejection': summarize(oos_hard_existing_res, 'oos_hard_medical_existing_3(oos12/13/19)'),
        'new_hard_oos_rejection': summarize(new_hard_res, 'new_hard_oos_12'),
        'new_hard_oos_by_subtype': {st: summarize(rs, st) for st, rs in new_hard_by_subtype.items()},
        'hard_medical_oos_combined_rejection': {
            'n': len(hard_medical_combined),
            'correct': sum(1 for r in hard_medical_combined if r['correct']),
            'rate': sum(1 for r in hard_medical_combined if r['correct']) / len(hard_medical_combined),
            'wrong_ids': [r['id'] for r in hard_medical_combined if not r['correct']],
        },
    }
    return out


def print_summary(res):
    m = res['method']
    r1 = res['in_scope_recall']
    r2 = res['oos_general_rejection']
    r3 = res['hard_medical_oos_combined_rejection']
    r_food = res['food_db_recall']
    print(f"\n=== {m} ===")
    print(f"  in-scope recall (positive 30): {r1['correct']}/{r1['n']} ({r1['rate']:.1%})"
          + (f"  FN={r1['wrong_ids']}" if r1['wrong_ids'] else ""))
    print(f"  food_db recall (5):            {r_food['correct']}/{r_food['n']} ({r_food['rate']:.1%})"
          + (f"  FN={r_food['wrong_ids']}" if r_food['wrong_ids'] else ""))
    print(f"  general-OOS rejection (18):    {r2['correct']}/{r2['n']} ({r2['rate']:.1%})"
          + (f"  FP={r2['wrong_ids']}" if r2['wrong_ids'] else ""))
    print(f"  hard-medical-OOS rejection (existing3+new5={r3['n']}): {r3['correct']}/{r3['n']} ({r3['rate']:.1%})"
          + (f"  FP={r3['wrong_ids']}" if r3['wrong_ids'] else ""))
    for st, s in res['new_hard_oos_by_subtype'].items():
        print(f"    - {st}: {s['correct']}/{s['n']} ({s['rate']:.1%})" + (f" FP={s['wrong_ids']}" if s['wrong_ids'] else ""))


# ═══════════════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    data = load_data()
    print(f"loaded: positive={len(data['positive'])}, oos_general={len(data['oos_general'])}, "
          f"oos_hard_existing={len(data['oos_hard_medical_existing'])}, "
          f"new_hard_scored={len(data['new_hard_oos_scored'])}, "
          f"short_ambiguous={len(data['short_ambiguous'])}, mixed_intent={len(data['mixed_intent'])}")

    RESULTS['data_composition'] = {k: len(v) for k, v in data.items()}
    RESULTS['data_composition']['total_oos_hard_medical_combined'] = (
        len(data['oos_hard_medical_existing']) +
        len([x for x in data['new_hard_oos_scored'] if x['id'].startswith('hoos') and 3 <= int(x['id'][4:]) <= 7])
    )
    save_results()

    # ── Method A ─────────────────────────────────────────────────────────────
    print('\n[Phase A] rule-based gate 평가...')
    res_a = eval_binary_method('A_rule', lambda q: rule_gate_binary(q)[0], data)
    print_summary(res_a)
    RESULTS['method_A'] = res_a
    save_results()

    # sanity: 규칙이 실수로 positive-30 중 어떤 문항이든 red_flag/procedure로 잘못 판정하는지 확인
    sanity_a = []
    for x in data['positive']:
        v, s = rule_verdict(x['question'])
        sanity_a.append({'id': x['id'], 'verdict': v, 'signals': s})
    RESULTS['method_A_positive_sanity'] = sanity_a
    save_results()

    # ── Method B ─────────────────────────────────────────────────────────────
    print('\n[Phase B] embedding scope-prototype 점수 계산 (라이브 API 호출)...')
    client = _client()
    cache = load_embed_cache()

    all_q = set()
    for key in ('positive', 'food_db', 'oos_general', 'oos_hard_medical_existing', 'new_hard_oos_scored', 'short_ambiguous'):
        all_q.update(x['question'] for x in data[key])
    all_q.update(x['question'] for x in data['mixed_intent'])
    all_q = sorted(all_q)

    scope_scores = compute_scope_scores(all_q, cache, client)
    RESULTS['method_B_raw_scores'] = {
        'single_prototype': {q: scope_scores[q]['single'] for q in all_q},
        'multi_prototype': {q: scope_scores[q]['multi'] for q in all_q},
    }
    save_results()

    # threshold sweep 참고용 분포 출력
    def dist(variant):
        pos_scores = [scope_scores[x['question']][variant] for x in data['positive']]
        oos_scores = [scope_scores[x['question']][variant] for x in data['oos_general']]
        hard_scores = [scope_scores[x['question']][variant] for x in data['oos_hard_medical_existing']]
        return pos_scores, oos_scores, hard_scores

    for variant in ('single', 'multi'):
        pos_s, oos_s, hard_s = dist(variant)
        print(f"  [{variant}] positive: min={min(pos_s):.3f} p10={sorted(pos_s)[2]:.3f} mean={np.mean(pos_s):.3f}")
        print(f"  [{variant}] oos_general: max={max(oos_s):.3f} p90={sorted(oos_s)[-2]:.3f} mean={np.mean(oos_s):.3f}")
        print(f"  [{variant}] hard_medical(3): {sorted(hard_s)}")

    RESULTS['method_B_score_distributions'] = {}
    for variant in ('single', 'multi'):
        pos_s, oos_s, hard_s = dist(variant)
        RESULTS['method_B_score_distributions'][variant] = {
            'positive_min': min(pos_s), 'positive_mean': float(np.mean(pos_s)),
            'oos_general_max': max(oos_s), 'oos_general_mean': float(np.mean(oos_s)),
            'hard_medical_scores': hard_s,
        }
    save_results()

    # threshold 후보 2개씩 시도 (single/multi 각각)
    thresholds_single = [0.42, 0.46]
    thresholds_multi = [0.48, 0.52]

    method_b_variants = {}
    for variant, thlist in (('single', thresholds_single), ('multi', thresholds_multi)):
        for th in thlist:
            key = f'B_{variant}_th{th}'
            gate_fn = (lambda q, v=variant, t=th: scope_scores[q][v] >= t)
            res_b = eval_binary_method(key, gate_fn, data)
            print_summary(res_b)
            method_b_variants[key] = res_b
    RESULTS['method_B_variants'] = method_b_variants
    save_results()

    # ── Method C (hybrid) ────────────────────────────────────────────────────
    print('\n[Phase C] hybrid (rule -> embedding fallback) 평가...')
    # multi-prototype, threshold 0.50을 폴백 기준으로 채택(§B 결과 중 균형점) — 아래 recommend
    # 단계에서 실제 수치로 재확인.
    hybrid_variant, hybrid_th = 'multi', 0.50

    def gate_c(q):
        passed, reason = hybrid_verdict(q, scope_scores, variant=hybrid_variant, threshold=hybrid_th)
        return passed

    res_c = eval_binary_method('C_hybrid', gate_c, data)
    print_summary(res_c)
    RESULTS['method_C'] = res_c
    RESULTS['method_C_config'] = {'variant': hybrid_variant, 'threshold': hybrid_th}
    save_results()

    # ── Mixed-intent: per-method 판정 상세 ──────────────────────────────────
    print('\n[Phase mixed-intent] A/B/C 각각의 판정...')
    mixed_report = []
    for x in data['mixed_intent']:
        q = x['question']
        a_pass, a_v, a_s = rule_gate_binary(q)
        b_single = scope_scores[q]['single']
        b_multi = scope_scores[q]['multi']
        c_pass, c_reason = hybrid_verdict(q, scope_scores, variant=hybrid_variant, threshold=hybrid_th)
        row = {
            'id': x['id'], 'question': q,
            'A_verdict': a_v, 'A_pass': a_pass, 'A_signals': a_s,
            'B_single_score': b_single, 'B_single_pass_th0.46': b_single >= 0.46,
            'B_multi_score': b_multi, 'B_multi_pass_th0.50': b_multi >= 0.50,
            'C_pass': c_pass, 'C_reason': c_reason,
            'safety_judgment': x['safety_judgment'],
        }
        mixed_report.append(row)
        print(f"  {x['id']}: A={a_v}({a_pass}) B_single={b_single:.3f} B_multi={b_multi:.3f} C={c_pass}({c_reason})")
    RESULTS['mixed_intent_report'] = mixed_report
    save_results()

    # ── short-ambiguous: per-method 판정 상세 (정답 없음, 진단용) ──────────
    print('\n[Phase short-ambiguous] A/B/C 각각의 판정...')
    short_report = []
    for x in data['short_ambiguous']:
        q = x['question']
        a_pass, a_v, a_s = rule_gate_binary(q)
        b_single = scope_scores[q]['single']
        b_multi = scope_scores[q]['multi']
        c_pass, c_reason = hybrid_verdict(q, scope_scores, variant=hybrid_variant, threshold=hybrid_th)
        row = {'id': x['id'], 'question': q, 'A_verdict': a_v, 'A_pass': a_pass,
               'B_single_score': b_single, 'B_multi_score': b_multi, 'C_pass': c_pass, 'C_reason': c_reason}
        short_report.append(row)
        print(f"  {x['id']} '{q}': A={a_v}({a_pass}) B_single={b_single:.3f} B_multi={b_multi:.3f} C={c_pass}")
    RESULTS['short_ambiguous_report'] = short_report
    save_results()

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")
    print(f"results saved: {RESULTS_PATH}")


if __name__ == '__main__':
    main()
