# -*- coding: utf-8 -*-
"""
FOOK_rag_chatbot.py — 환자 질문 응답 챗봇 (RAG).
data/FOOK_rag_kb.json(청크+임베딩)에서 질문과 유사한 조각을 찾아 GPT에게 근거로 주고 답하게 한다.

API 키: 환경변수 OPENAI_API_KEY.
"""
import os, json, re
from collections import namedtuple
import numpy as np

_KB = None          # [{'text','source','embedding'}, ...]
_KB_MAT = None       # (N, dim) numpy 행렬, 코사인 유사도 계산용


def _load_kb():
    global _KB, _KB_MAT
    if _KB is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'FOOK_rag_kb.json')
        _KB = json.load(open(path, encoding='utf-8'))
        mat = np.array([c['embedding'] for c in _KB], dtype=np.float32)
        _KB_MAT = mat / np.linalg.norm(mat, axis=1, keepdims=True)   # 미리 정규화(코사인=내적)
    return _KB, _KB_MAT


def _client():
    from openai import OpenAI
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
    return OpenAI(api_key=key)


def retrieve(query, top_k=5):
    """질문과 코사인 유사도가 가장 높은 청크 top_k개 반환. [{'text','source','score'}, ...]
    점수 그대로(필터 없이) 반환하는 순수 검색 함수 — "관련 없으면 버린다"는 정책 판단은
    호출부(answer()/food_lookup_answer())가 RAG_MIN_SCORE로 한다(아래 _relevant() 참고).
    평가 스크립트(evaluation/evaluate_rag.py)가 Top-k 적중률을 재려면 원 점수가 그대로 필요하므로
    여기서 걸러버리면 안 된다."""
    kb, mat = _load_kb()
    client = _client()
    q_emb = client.embeddings.create(model='text-embedding-3-small', input=[query]).data[0].embedding
    q = np.array(q_emb, dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = mat @ q       # 코사인 유사도(정규화됐으므로 내적=코사인)
    idx = np.argsort(-scores)[:top_k]
    return [{'text': kb[i]['text'], 'source': kb[i]['source'], 'score': float(scores[i])} for i in idx]


# ── 최소 유사도 게이트: "관련 자료 없음"을 판단하는 기준 ─────────────────────────
# 이 값은 "검색 관련성(retrieval relevance)" 임계값이지, 임상적/의학적으로 검증된 안전 기준이
# 아니다 — 어디까지나 아래 v2 평가셋에서 실측한 점수 분포로부터 고른 값이며, "이 값이 확정적으로
# 최적"이라는 주장도 아니다(더 넓은 질문셋·KB가 쌓이면 재조정 대상).
#
# 1차 도출(2026-08-13, RAG_MIN_SCORE=0.15): OPENAI_API_KEY가 없던 환경에서 실제 이탈 질문을
# 라이브로 임베딩할 수 없어, KB 자체 청크쌍 유사도(평균 0.425, p10=0.282)와 완전 무관한 무작위
# 벡터 200개의 최댓값 분포(≤0.122, "노이즈 바닥")로부터 그 사이 어딘가일 것이라 추정만 하고
# 넉넉한 안전 마진을 둔 잠정값이었다.
#
# 2차 도출(2026-08-13, RAG_MIN_SCORE=0.30): OPENAI_API_KEY가 있는 환경에서 evaluation/
# evaluate_rag_v2.py로 실제 이탈 질문(out_of_scope) 21개 + 정답이 있는 질문(rag) 30개를
# text-embedding-3-small로 라이브 임베딩해 top-5 점수 분포를 직접 측정했다(결과:
# evaluation/rag_eval_results_v2.json, 청크 단위 근거 검토: evaluation/rag_evidence_inspection_v2.md).
# 0.30은 이 측정에서 긍정 재현율(정답 문서가 top-5 안에 있는 질문 중 실제로 게이트를 통과하는
# 비율) 약 29/30(96.7%, fish1만 탈락)과 이탈 질문 정답 거부율 약 16/21(76.2%)을 동시에 준다 —
# 완벽하지 않고(감기약·당뇨 인슐린·암 치료 같은 "의학적이지만 범위 밖" 질문 일부는 여전히
# 통과함, 위 v2 평가 문서 참고), 더 높이면(0.40~0.48) 이탈 거부율은 오르지만 긍정 재현율이
# 눈에 띄게 깎인다(같은 평가에서 확인). 즉 이 값은 "이 특정 평가셋의 측정된 점수 분포에서 고른
# 절충점"이지 임상적으로 안전하다고 보증된 수치가 아니다 — 오답 사례가 있어도 항상
# NO_EVIDENCE_ANSWER로 조심스럽게 안내하고 담당 의료진 상담을 권하는 SYSTEM 프롬프트가 최종
# 안전판이다.
# 환경변수 RAG_MIN_SCORE로 재정의 가능(운영 중 데이터가 쌓이면 조정할 수 있도록).
RAG_MIN_SCORE = float(os.environ.get('RAG_MIN_SCORE', '0.30'))

NO_EVIDENCE_ANSWER = (
    "제공된 자료에서는 확인할 수 없습니다. 이 질문은 저희가 갖고 있는 임상 영양 자료 범위를 "
    "벗어난 것 같아요 — 담당 의료진이나 영양사와 상담해 주세요."
)


def _relevant(hits, min_score=None):
    """retrieve()가 반환한 원(raw) 임베딩 코사인 점수 중 RAG_MIN_SCORE 이상만 남긴다 — "그럴듯한
    top_k"가 아니라 "실제로 관련 있는" 것만 LLM 프롬프트에 넣기 위한 정책 필터.
    min_score를 안 주면 RAG_MIN_SCORE.
    ── 안전 불변식(구조적으로 보장) ──────────────────────────────────────────────────
    이 함수는 h['score'](retrieve()가 매긴 원 코사인 유사도)만 비교한다 — 절대로 어휘 재정렬
    (_rerank()/rerank_score)이 적용된 값과 비교하지 않는다. answer()의 파이프라인은
    retrieve() -> _relevant()(게이트, 여기) -> _rerank()(재정렬) 순서로 호출되므로, 이 함수가
    호출되는 시점에는 rerank_score라는 게 아예 존재하지 않는다 — "우연히 지금 값이 같아서"가
    아니라 함수 호출 순서상 구조적으로 게이트가 재정렬 이전 원점수만 볼 수 있다."""
    thresh = RAG_MIN_SCORE if min_score is None else min_score
    return [h for h in hits if h['score'] >= thresh]


# ── 경량 재정렬(reranking): 어휘 중복 보너스 ──────────────────────────────────────
# backend/evaluation/reranking_experiment.py에서 검증된 실험 결과를 그대로 반영한다:
# alpha=0.05는 Evidence Recall@3을 43.3%->53.3%로 끌어올리면서 out-of-scope 게이트 오통과
# 사례가 0건이었다. alpha=0.15는 더 큰 개선을 보였지만 "다음 주 로또 번호 추천해줘"
# 등 out-of-scope 질문 2건이 게이트를 잘못 통과시키는 것이 확인되어 채택하지 않는다
# (reranking_experiment_results.json의 oos_safety_check 참고). dedup/source-diversity cap은
# 실효가 없거나(중복 없음) 순역행(net-negative)이라 함께 채택하지 않는다.
#
# 중요: 이 재정렬은 반드시 _relevant() 게이트를 이미 통과한 후보에 대해서만, 그 이후 단계에서만
# 적용한다 — 게이트 판정 자체에는 이 보너스가 절대 섞이지 않는다(위 _relevant() 독스트링의
# 안전 불변식 참고).
RAG_LEXICAL_ALPHA = float(os.environ.get('RAG_LEXICAL_ALPHA', '0.05'))

RAG_CANDIDATE_POOL = 10   # answer()가 retrieve()에서 뽑아오는 원 후보 풀 크기(재정렬 여유를 위해 5->10)
RAG_CONTEXT_MAX = 5       # 게이트 통과 + 재정렬 이후, 실제 LLM 프롬프트에 넣는 최종 청크 수 상한

# 한국어 조사/어미 때문에 형태소 분석 없이는 토큰화가 불안정하므로, 질문을 공백 기준으로 쪼갠 뒤
# 흔한 조사/어미 접미사를 벗겨내는 방식으로 "내용어"만 추출한다(형태소 분석기 등 새 의존성 없음).
# backend/evaluation/reranking_fetch_pool.py / reranking_experiment.py 실험에서 쓴 것과 완전히
# 동일한 정규화 규칙·접미사 목록·부분문자열 매칭 로직을 그대로 재사용한다(재발명하지 않음).
_LEX_DASH_RE = re.compile(r'[‐‑‒–—−~－]')
_LEX_WS_RE = re.compile(r'\s+')
_LEX_PUNCT_RE = re.compile(r'[.,!?()\[\]{}·:;"\'“”‘’…/\\|~\-‐-―]')

_LEX_STOP_SUFFIXES = sorted([
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로', '와', '과', '도', '만',
    '나요', '까요', '습니까', '합니까', '되나요', '인가요', '하나요', '해요', '이나', '한가요',
    '해야', '해도', '해서', '해서요', '드셔도', '드릴까요',
], key=len, reverse=True)


def _lex_norm_light(s):
    """공백 정리 + 구두점 제거 + 소문자화 — 어휘 중복(부분문자열 매칭) 비교용."""
    s = _LEX_PUNCT_RE.sub(' ', s)
    s = _LEX_WS_RE.sub(' ', s).strip().lower()
    return s


def _question_content_tokens(question):
    """질문에서 조사/어미를 벗겨낸 2자 이상 내용어 토큰 목록(등장 순서 유지, 중복 제거)."""
    q = _LEX_PUNCT_RE.sub(' ', question).strip()
    words = _LEX_WS_RE.split(q)
    toks = []
    for w in words:
        w2 = w
        for suf in _LEX_STOP_SUFFIXES:
            if w2.endswith(suf) and len(w2) - len(suf) >= 2:
                w2 = w2[: -len(suf)]
                break
        if len(w2) >= 2:
            toks.append(w2)
    seen = set(); out = []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def _lexical_bonus(question, chunk_text):
    """질문의 내용어 토큰 중 청크 텍스트에 (부분문자열로) 등장하는 비율(0.0~1.0).
    독립적으로 테스트 가능하도록 게이트 로직(_relevant())과 완전히 분리된 별도 함수 —
    이 함수의 반환값은 오직 _rerank()에서 게이트 통과 후 정렬 순서를 바꾸는 데만 쓰인다."""
    tokens = _question_content_tokens(question)
    if not tokens:
        return 0.0
    nt = _lex_norm_light(chunk_text)
    hits = sum(1 for t in tokens if t in nt)
    return hits / len(tokens)


def _rerank(question, hits):
    """게이트(_relevant())를 이미 통과한 hits만 입력으로 받는다 — 게이트 판정에는 관여하지 않고,
    그 이후 "어느 순서로 LLM 프롬프트에 넣을까"만 바꾼다.
    각 후보에 rerank_score = score(원 코사인 유사도, 불변) + RAG_LEXICAL_ALPHA * 어휘보너스를
    추가해 그 값으로 재정렬한다. 원본 'score' 키는 절대 덮어쓰지 않는다 — retrieve()가 매긴
    원 코사인 유사도라는 의미를 그대로 유지해야 evaluation/ 스크립트들이 직접 retrieve()를
    불러 쓸 때(원점수 기대)와 로그·디버그 경로에서 'score'를 봐도 재정렬값과 혼동되지 않는다."""
    scored = []
    for h in hits:
        bonus = _lexical_bonus(question, h['text'])
        rerank_score = h['score'] + RAG_LEXICAL_ALPHA * bonus
        scored.append({**h, 'rerank_score': rerank_score})
    # Python list.sort()는 안정 정렬이므로 rerank_score가 같은 경우 원래(원점수 기준) 순서가
    # 그대로 유지된다 — 별도 tiebreaker 코드 없이도 실험 스크립트와 동일한 동점 처리가 된다.
    scored.sort(key=lambda h: -h['rerank_score'])
    return scored


SYSTEM = (
    "너는 혈액투석 환자를 위한 영양 정보 안내 챗봇이다. "
    "반드시 아래 [참고 자료]에 있는 내용만 근거로 답하고, 참고 자료에 없는 내용은 "
    "'제공된 자료에서 확인할 수 없다'고 말해라 — 지어내지 마라. "
    "특정 환자의 검사 수치나 약물처럼 개인별로 다른 판단이 필요한 질문에는 "
    "일반적인 원칙만 설명하고, 반드시 담당 의료진·영양사와 상담하도록 안내해라. "
    "친절하고 간결한 한국어로, 3~5문장 이내로 답하라."
)


# ── 특정 재료 질문("OO 먹어도 되나요?") — RAG 대신 영양DB 직접 조회 ──────────────
# 등급 기준은 apply_potassium_filter.py(임상영양사 피드백 반영)와 동일하게 맞춘다:
# 1회분(식품교환단위) 칼륨 기준 저<100mg, 중100~200mg, 고>200mg.
SERVING_G = {
    '육류 및 그 제품': 40, '어패류 및 그 제품': 50, '난류': 55, '두류': 80,
    '조리가공식품류': 50, '채소류': 70, '버섯류': 70, '해조류': 70, '절임류': 70,
    '곡류 및 그 제품': 70, '감자류 및 전분류': 100, '빵 및 과자류': 70,
    '과일류': 120, '우유 및 그 제품': 200, '음료류': 200,
    '유지류': 5, '견과류 및 종실류': 10, '조미료류': 10, '당류': 10,
}
DEFAULT_SERVING_G = 50


def _grade(k_per_serving):
    if k_per_serving < 100:
        return '저칼륨'
    if k_per_serving <= 200:
        return '중칼륨'
    return '고칼륨'


_ING_INDEX = None   # {짧은이름: [전체 재료명, ...]}

# 재료명 세그먼트 중 상태/조리법을 나타내는 흔한 단어 — 이걸 그대로 인덱싱하면 거의 모든 질문에
# 오매칭되므로 제외 (예: "생것"을 인덱싱하면 "생선"이라는 단어가 들어간 모든 질문이 오탐됨).
_GENERIC_SEGMENTS = {
    '생것', '삶은것', '데친것', '말린것', '냉동', '통조림', '조리한것', '건조',
    '찐것', '구운것', '볶은것', '튀긴것', '가루', '분말', '재배', '수입산', '국내산',
    '젓갈', '국물', '조미',
}

# 재료명 세그먼트 중 너무 넓은 식품 범주명 — 상태어는 아니지만 특정 재료를 가리키지 않아
# 위와 같은 이유로 제외해야 함(2026-08-13, 실측 발견): "과일"이 "빵, 페이스트리, 과일"의
# 비-첫 세그먼트로 들어있어 "통조림 과일은 먹어도 괜찮은가요?" 같은 일반 질문이 그 빵 항목으로
# 오매칭되고 있었다(RAG로 가야 할 질문이 food DB route로 잘못 빠짐). 첫 세그먼트(기본식품명)라면
# 이런 범주명이어도 그대로 인덱싱한다 — 문제는 "비-첫 세그먼트로서" 인덱싱될 때만 발생한다.
_GENERIC_CATEGORY_SEGMENTS = {'과일', '채소', '생선'}


def _food_index():
    global _ING_INDEX
    if _ING_INDEX is None:
        import FOOK_adjust_levers as F
        if F.NUT is None:
            F.NUT = F.load_all()
        ing_nut = F.NUT[0]
        idx = {}
        for name in ing_nut:
            segs = [s.strip() for s in name.split(',')]
            for i, short in enumerate(segs):
                # 첫 세그먼트(기본식품명)는 항상 인덱싱. 그 외는 상태어·범주명이 아니고
                # 충분히 구체적일 때만.
                if i > 0 and (short in _GENERIC_SEGMENTS or short in _GENERIC_CATEGORY_SEGMENTS
                              or len(short) < 2):
                    continue
                if len(short) >= 2:
                    idx.setdefault(short, []).append(name)
        _ING_INDEX = (idx, ing_nut)
    return _ING_INDEX


# 식이 질문 트리거 — 이게 있어야만 재료 조회로 판단한다. 재료명이 우연히 다른 뜻으로 문장에
# 등장하는 오탐 방지(실측 발견: "신장 기능이 걱정돼요"→돼지 신장으로, "사과 드립니다"→과일 사과로
# 잘못 매칭됨. 단어를 하나씩 막는 대신 "먹는 것에 대한 질문인가" 자체를 조건으로 건다).
FOOD_QUESTION_KW = ('먹어도', '먹을', '먹으면', '섭취', '드셔도', '드시면', '드셔', '얼마나 먹',
                     '먹기', '식사', '간식으로')


# 질문에 상태를 나타내는 말이 있으면 그 상태의 DB 항목을 우선한다 — 실측 발견: "말린 바나나"라고
# 물어도 무조건 '생것'을 골라버려서, 냉동건조 바나나(K1526mg/100g, 생것의 4배 이상)를 물었는데
# 생것 수치(K355mg)로 "괜찮다"고 답할 뻔한 사례가 나옴. 재료 상태에 따라 칼륨이 몇 배씩 차이 나므로
# 질문의 상태 표현을 최우선으로 매칭한다.
STATE_KW = {
    '말린': ('말린것', '건조', '동결건조', '반건조'), '건조': ('말린것', '건조', '동결건조'),
    '냉동': ('냉동',), '통조림': ('통조림',), '캔': ('통조림',),
    '삶은': ('삶은것', '삶아서'), '데친': ('데친것',),
    '구운': ('구운것',), '볶은': ('볶은것',), '튀긴': ('튀긴것',),
}


def find_food(question):
    """질문 텍스트에서 영양DB에 있는 재료명을 찾는다. 가장 긴(구체적인) 매칭 우선.
    같은 짧은이름에 후보가 여럿이면: 질문에 재료 상태(말린/냉동/통조림 등)가 명시돼 있으면 그 상태의
    항목을 우선하고, 없으면 '생것'을 우선(가공품보다 원재료 질문일 확률이 높음).
    식이 질문 트리거 단어가 없으면 재료명이 우연히 섞여있어도 매칭하지 않는다."""
    if not any(kw in question for kw in FOOD_QUESTION_KW):
        return None
    idx, ing_nut = _food_index()
    candidates = [s for s in idx if s in question]
    if not candidates:
        return None
    short = max(candidates, key=len)       # "무" 보다 "무말랭이"처럼 더 구체적인 것 우선
    names = idx[short]

    for q_kw, db_kws in STATE_KW.items():
        if q_kw in question:
            for n in names:
                if any(dk in n for dk in db_kws):
                    return n
            break   # 질문에 상태어가 있는데 DB에 해당 상태 항목이 없으면 생것 폴백으로 넘어감

    for n in names:
        if '생것' in n:
            return n
    return names[0]


CONSUMED_KEYS = ('E', 'protein', 'K', 'P', 'Na', 'Na_season')   # /generate 응답의 intake와 동일 키


def _remaining_budget(weight, consumed, meals_left=None):
    """남은 예산 계산. consumed는 /generate의 intake와 같은 형식.
    meals_left(오늘 남은 끼니 수, 이번 것 포함)를 주면 '하루 전체 남은 예산'이 아니라
    meal_bounds()가 쓰는 것과 동일한 '다음 한 끼 몫(남은예산÷남은끼니, ±20% 밴드)'으로 비교한다
    — 끼니가 더 남아있으면 지금 다 써버리면 안 되므로. 안 주면(모르면) 하루 전체 남은 예산으로 폴백
    (마지막 끼니라고 가정하는 게 더 보수적이진 않지만, 정보가 없을 때의 기본 동작).
    나트륨(Na/Na_season)은 여기서 다루지 않는다 — meal_bounds()의 Namax는 끼니 고정값(=SALT_MG,
    첨가염/조미료 전용)이라 consumed와 무관하게 항상 동일하다(FOOK_adjust_levers.py meal_bounds()
    주석 "조미료는 맛 보존 하한이 있어 예산화 의미 없음" 및
    tests/test_adjust_levers.py::test_meal_bounds_namax_is_fixed_regardless_of_consumed 참고).
    즉 'Namax - 오늘 이미 먹은 Na_season' 같은 차감은 '고정된 다음 끼니 목표'에서 '이전 끼니에 이미
    쓴 값'을 빼는 범주 오류라 여기서 계산하지 않는다. 재료 1개의 나트륨 판단은 food_lookup_answer()가
    (Na_season이 아니라) 재료 자체의 총 나트륨 값을 근거로 별도로 처리한다."""
    import FOOK_adjust_levers as F
    c = {k: float(consumed.get(k) or 0) for k in CONSUMED_KEYS}
    if meals_left:
        mb = F.meal_bounds(weight, consumed, meals_left=meals_left)
        return {
            'K_left': mb['Kmax'], 'P_left': mb['Pmax'],
            'K_consumed': c['K'], 'P_consumed': c['P'],
            'scope': f'다음 한 끼 몫 (오늘 {meals_left}끼 남음, 남은예산÷남은끼니 기준)',
        }
    d = F.day_targets(weight)
    return {
        'K_left': d['Kmax'] - c['K'],
        'P_left': d['Pmax'] - c['P'],
        'K_consumed': c['K'], 'P_consumed': c['P'],
        'scope': '오늘 하루 전체 남은 예산',
    }


def food_lookup_answer(question, food_name, weight=None, consumed=None, meals_left=None,
                        top_k=3, model='gpt-4o-mini'):
    """영양DB 실측치로 특정 재료 질문에 답한다. 숫자·최종 판정은 코드가 확정하고 LLM은 문장만
    (아래 [최종 판정]은 절대 못 뒤집음). 단, 지식베이스(RAG)도 같이 검색해서 "말린 과일은 생과일보다
    칼륨이 2배" 같이 숫자 조회만으론 못 잡는 조리법·주의사항을 참고자료로 덧붙인다 — 수치 판정을
    RAG가 뒤집지는 못하지만, 놓치기 쉬운 세부 지침을 보완해준다.
    weight+consumed가 있으면(오늘 이미 먹은 양을 앱이 알고 있으면) '지금 이거 더 먹어도 남은
    예산 안에 들어오는가'까지 계산해서 답한다 — 이게 이 챗봇이 일반 LLM과 다른 핵심 지점.
    meals_left(오늘 남은 끼니 수)까지 있으면 하루 전체 잔여가 아니라 '다음 한 끼 몫'으로 비교한다
    (끼니가 남아있는데 지금 그 몫을 다 써버리면 다음 끼니가 너무 빡빡해지므로).
    반환: (답변, RAG출처 리스트)"""
    import FOOK_adjust_levers as F
    ing_nut = F.NUT[0]
    d = ing_nut[food_name]
    group = d.get('group')
    serving = SERVING_G.get(group, DEFAULT_SERVING_G)
    k100 = d.get('K')
    if k100 is None:
        return None   # 칼륨 데이터 없음 -> RAG로 폴백
    k_serving = k100 * serving / 100
    grade = _grade(k_serving)
    p100, na100 = d.get('P'), d.get('Na')
    p_serving = (p100 * serving / 100) if p100 is not None else None

    facts = f"재료명: {food_name}\n100g당 칼륨 {k100:.0f}mg"
    facts += f", 인 {p100:.0f}mg" if p100 is not None else ""
    facts += f", 나트륨 {na100:.0f}mg" if na100 is not None else ""
    facts += (f"\n1회 섭취 기준량({serving}g)당 칼륨 약 {k_serving:.0f}mg -> {grade} 식품 "
              f"(저칼륨 <100mg / 중칼륨 100~200mg / 고칼륨 >200mg, 1회 섭취 기준량 기준)")

    # ── 나트륨: K/P처럼 "남은 예산"과 비교하는 하드 판정을 하지 않는다(의도적) ──────────────
    # 이유(FOOK_adjust_levers.py의 실제 나트륨 설계와 대조해 확인):
    #  1) season_na()/Na_season은 "첨가염(조미료·젓갈)만" 합산한다(is_salty_seasoning 참고) — 일반
    #     원재료(채소·과일·육류 등, group != '조미료류')는 재료 자체로는 Na_season 기여가 항상 0이다.
    #     즉 "재료 하나를 더 먹어도 되는가"를 Na_season 예산과 비교하는 것은 범주 오류다(그 예산은
    #     "이번 끼니에 조미료를 얼마나 넣을까"를 위한 것이지, 재료 자체의 나트륨을 위한 게 아니다).
    #  2) meal_bounds()의 Namax(첨가염 상한)는 끼니 고정값(SALT_MG)이라 consumed와 무관하다
    #     (다음 끼니 목표라 "이미 먹은 양"을 뺄 대상이 아님 — 위 test_meal_bounds_namax_is_fixed_
    #     regardless_of_consumed 참고). K/P처럼 "남은 예산 - 이 재료 몫"을 계산해 표시하면 의미
    #     없는 숫자(거짓 정밀도)를 만들어낸다.
    #  3) 이 서비스의 실제 나트륨 철학은 app_core_FOOK.py::_total_na_warning()이 보여주듯
    #     "자연 나트륨은 못 줄이니 하드 실패로 막지 않고, 그래도 총량이 많으면 경고만 준다"이다
    #     (미역국 등은 Na_season 판정을 통과해도 총나트륨이 높으면 끼니 단위 경고를 준다).
    #     재료 1개 조회는 그보다도 더 국소적인 질문이므로, K/P처럼 이분법(먹어도 됨/안 됨)으로
    #     가두지 않고 같은 "경고" 철학을 따른다 — 단, "지금 더 드셔도 됩니다"라고 잘못 안심시키지
    #     않도록 재료 자체 나트륨이 매우 높으면(끼니 총나트륨 경고 기준 이상) 그 사실을 최종 결론에도
    #     반영한다(아래 severe_na).
    # 임계값은 새로 만들지 않고 기존 상수를 그대로 재사용한다:
    #   NA_TOTAL_MEAL(655, "끼니 총 나트륨 목표 — 첨가염+음식자체 전부")과
    #   NA_TOTAL_WARN(786, "그래도 넘으면 경고" 기준, _total_na_warning()과 동일 기준).
    na_serving = (na100 * serving / 100) if na100 is not None else None
    severe_na = na_serving is not None and na_serving >= F.NA_TOTAL_WARN
    if na_serving is not None:
        if severe_na:
            facts += (
                f"\n\n[나트륨 참고 — 이 재료 자체]\n"
                f"1회 섭취 기준량({serving}g)의 나트륨만 약 {na_serving:.0f}mg으로, 끼니 전체 나트륨"
                f" 경고 기준({F.NA_TOTAL_WARN:.0f}mg, 소금 2g 상당)에 이미 도달/근접합니다. 자연 나트륨은"
                f" 조미료처럼 줄일 수 없으니, 이 재료는 양을 줄이거나 이번 끼니에는 피하는 게 좋습니다."
            )
        elif na_serving >= F.NA_TOTAL_MEAL:
            facts += (
                f"\n\n[나트륨 참고 — 이 재료 자체]\n"
                f"1회 섭취 기준량({serving}g)의 나트륨이 약 {na_serving:.0f}mg으로, 한 끼 나트륨 목표"
                f"({F.NA_TOTAL_MEAL:.0f}mg) 수준입니다. 이 재료를 먹는 끼니는 다른 반찬을 싱겁게 하는 등"
                f" 나트륨을 함께 조절하는 게 좋습니다."
            )

    personalized = weight and consumed
    verdict = None
    if personalized:
        rb = _remaining_budget(weight, consumed, meals_left=meals_left)
        fits_k = rb['K_left'] >= k_serving
        fits_p = True
        facts += (
            f"\n\n[오늘 이 환자의 실제 섭취 현황 — 기준: {rb['scope']}]\n"
            f"이미 오늘 칼륨 {rb['K_consumed']:.0f}mg 섭취, 비교 대상 남은 칼륨 예산 {rb['K_left']:.0f}mg\n"
            f"{food_name} 1회 섭취 기준량({serving}g)을 지금 더 먹으면 칼륨 {k_serving:.0f}mg 추가 "
            f"(= {rb['K_left']:.0f} - {k_serving:.0f} = {rb['K_left']-k_serving:+.0f}mg 남음)"
        )
        if p_serving is not None:
            fits_p = rb['P_left'] >= p_serving
            facts += (f"\n인은 비교 대상 남은 예산 {rb['P_left']:.0f}mg 중 이 재료로 {p_serving:.0f}mg 추가 "
                      f"(= {rb['P_left']:.0f} - {p_serving:.0f} = {rb['P_left']-p_serving:+.0f}mg 남음)")
        # 판정은 코드가 내리고 LLM은 이 문장을 절대 뒤집지 않는다 — "숫자는 코드, 문장은 LLM" 원칙.
        # (실측 발견: 판정을 LLM에게 맡겼더니 원시 숫자를 스스로 다시 비교하다가 부호를 반대로 말하는
        #  사례 확인 — 426mg 필요·1200mg 남음인데도 "초과한다"고 답한 적 있음. 그래서 최종 결론
        #  문자열 자체를 코드에서 확정해 프롬프트에 박아넣고, LLM은 이유 설명만 담당하게 한다.)
        # 나트륨은 fits_k/fits_p처럼 이분법 게이트에 넣지 않지만(위 설명), severe_na(재료 자체 나트륨이
        # 끼니 경고 기준 이상)면 K/P가 통과여도 "지금 더 드셔도 됩니다"라고 잘못 안심시키지는 않는다.
        if fits_k and fits_p and not severe_na:
            verdict = '지금 더 드셔도 됩니다'
        elif fits_k and fits_p and severe_na:
            verdict = '지금은 피하시는 게 좋습니다 (칼륨·인은 괜찮지만 이 재료 자체의 나트륨이 높음)'
        else:
            verdict = '지금은 피하시는 게 좋습니다'
        facts += f"\n\n[최종 판정 — 코드가 이미 계산 완료, 이 결론을 그대로 따를 것]\n{verdict}"

    # RAG: 이 재료·관련 조리법에 대한 지식베이스 발췌를 참고자료로 덧붙인다(수치 판정을 뒤집는 용도
    # 아님 — "말린 과일은 칼륨이 2배" 같은 조회만으론 안 잡히는 세부 지침 보완용).
    # RAG_MIN_SCORE 미만인 후보는 애초에 프롬프트에 넣지 않는다 — 여긴 보조 정보라 "관련자료 없음"으로
    # 답변을 거부하지는 않지만(주 답변은 영양DB 수치), 무관한 조각이 섞여 들어가 LLM이 엉뚱한 주의사항을
    # 덧붙이는 걸 막는다.
    # (2026-08-14 결정) answer()의 주 RAG 경로와 달리 여기는 RAG_CANDIDATE_POOL 확대 + _rerank()
    # 재정렬을 적용하지 않는다 — 근거: 1) 여긴 "답변 가능 여부"를 좌우하는 게이트가 아니라 이미
    # 확정된 영양DB 수치 답변에 곁들이는 보조 정보일 뿐이라 재정렬로 얻을 안전상 이득이 없고,
    # 2) top_k=3의 작은 풀에서는 재정렬이 뒤집을 여지가 거의 없으며, 3) 이 보조 경로까지 넓히면
    # (게이트가 없는 곳에) 검증되지 않은 새로운 표면적을 만드는 셈이라 이번 작업의 범위(RAG_MIN_SCORE
    # 불변, alpha=0.05만 채택) 밖이다. 그대로 top_k=top_k(기본 3)의 단순 조회를 유지한다.
    hits = _relevant(retrieve(question, top_k=top_k))
    rag_block = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    rag_sources = sorted({h['source'] for h in hits})

    na_instruction = (
        " [나트륨 참고]가 있으면 그 문장 그대로의 취지로 짧게 언급해라(과장하거나 구체적인 "
        "'남은 나트륨 예산' 수치를 새로 만들어내지 마라 — 그런 수치는 계산되지 않았다)."
        if na_serving is not None else ""
    )
    instruction = (
        "위 [영양DB 실측 자료]의 숫자를 우선 근거로 이 재료가 저/중/고칼륨 중 어디에 속하는지와 "
        "혈액투석 환자가 어떻게 섭취하면 좋을지 안내해라. [관련 임상 자료 발췌]는 조리법·주의사항 "
        "같은 보완 정보로만 참고하고, 그 안에 이 상황과 무관한 내용이 있으면 무시해라. "
        "숫자를 임의로 바꾸거나 새로 지어내지 마라." + na_instruction
    )
    if personalized:
        instruction = (
            f"[최종 판정]은 이미 코드가 계산을 마친 확정된 결론이다 — 절대 네가 숫자를 다시 비교하거나 "
            f"판정을 뒤집지 마라. 답변은 반드시 '{verdict}'라는 결론으로 끝나야 한다. "
            "그 위의 [오늘 이 환자의 실제 섭취 현황] 숫자를 근거로 왜 그런 결론인지 설명하고, "
            "[관련 임상 자료 발췌]에 말린 과일·조리법 등 이 재료 섭취에 실제로 도움되는 주의사항이 "
            "있으면 짧게 덧붙여라(무관하면 무시). 숫자를 임의로 바꾸거나 새로 지어내지 마라." + na_instruction
        )
    prompt = (f"[영양DB 실측 자료]\n{facts}\n\n[관련 임상 자료 발췌]\n{rag_block}\n\n"
              f"[환자 질문]\n{question}\n\n{instruction}")
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    if verdict:
        # 이중 안전장치: LLM 문장이 결론을 잘못 말해도(실측 발견 사례 있음) 환자가 최종적으로 보는
        # 결론은 항상 코드가 계산한 값이어야 한다 — 프롬프트 지시만 믿지 않고 답변 끝에 확정 결론을 못박는다.
        text += f"\n\n▶ 결론: {verdict}"
    return text, rag_sources


# ── Scope gate: "이 질문이 KOOK(혈액투석 영양) 도메인 안인가" 사전 필터 ──────────────
# 근거: backend/evaluation/scope_gate_experiment.py에서 결정론적 규칙 기반 "Method A"를
# Method B(임베딩 scope-prototype 유사도) / Method C(A+B 하이브리드)와 비교 평가했고,
# scope_gate_experiment_results.json에서 Method A가 positive 30/30, food_db는 별도 경로라
# 텍스트만으로는 0/5(→ find_food()가 먼저 가로채므로 무관, 아래 참고), general-OOS 18/18,
# hard-medical-OOS(기존 3 + 신규 5) 8/8을 만족함을 확인했다. 실제 production 파이프라인
# 순서(find_food → scope gate → retrieve → _relevant → _rerank)를 그대로 재현한
# scope_gate_pipeline_sim_results.json에서도 동일 결과(no-answer accuracy 18/33 → 33/33,
# positive pass-through 29/30 불변 — fish1은 scope와 무관하게 RAG_MIN_SCORE 게이트에서
# 원래도 탈락, retrieval-unchanged 확인 완료)로 재확인됐다. Method B는 in-scope recall이
# 20~33%로 붕괴해 기각, Method C는 핵심 지표는 동률이지만 mixed-intent 2건에서 새로운
# 과잉차단이 생겨 Method A(더 단순하고 동등 효과)가 채택됐다 — scope_gate_definitions.md
# §4~6 참고. 아래 DOMAIN_TERMS/DIALYSIS_PROCEDURE_TERMS/RED_FLAG_TERMS와 판정 로직은
# scope_gate_experiment.py의 것을 그대로 복제한다(eval 문항 텍스트에 맞춰 사후 조정된 규칙이
# 아님 — 그 스크립트의 출처 주석 참고). 평가와 production 로직이 갈라지지 않도록 나중에
# 어느 한쪽만 고치는 일이 없어야 한다.
#
# 구조적으로 중요한 두 가지:
#   1) 이 게이트는 answer()에서 find_food()가 실패했을 때만 호출된다 — food_db 라우팅
#      질문("바나나 먹어도 되나요?" 등)은 scope gate 이전에 find_food()가 이미 가로채므로
#      절대 이 필터를 거치지 않는다(텍스트만 보면 도메인 키워드가 거의 없어 이 필터를 통과
#      못 하는 경우가 많음 — 그래서 순서가 안전을 좌우한다).
#   2) RAG_MIN_SCORE 임베딩 게이트(_relevant())와 이 scope gate는 완전히 독립된 별도
#      질문("이 질문이 KOOK 도메인 안인가" vs "KB에서 충분히 관련된 근거를 찾았는가")이다 —
#      순수 문자열 매칭만 하며 임베딩 점수/어휘 재정렬 보너스를 전혀 읽거나 쓰지 않는다.
IN_SCOPE = 'IN_SCOPE'
OUT_OF_SCOPE = 'OUT_OF_SCOPE'

# scope_gate_experiment.py DOMAIN_TERMS와 동일 — 출처: (1) SYSTEM 프롬프트가 선언한 영양
# 관리 축(칼륨/인/나트륨/단백질/수분), (2) FOOK_adjust_levers.py의 5대 영양소 프레임워크,
# (3) 기존 v2 positive 30문항이 다뤄온 인접 임상지표(알부민/부종/가려움/뼈/산증), (4) KB
# 목차 자체의 챕터 주제어(과일/채소/생선/물). eval 문항 텍스트를 보고 사후에 끼워맞춘 목록이
# 아니다 — scope_gate_definitions.md가 이 스크립트보다 먼저 작성됐다.
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
    # KB 목차에 실제로 존재하는 챕터 주제어(scope_gate_definitions.md §2 — "과일이 먹고
    # 싶을 때", "채소가 먹고 싶을 때", "물은 얼마나 마셔도 되나요?"). '물'은 1글자라
    # 부분문자열 충돌 위험(예: '동물', '선물')이 있음을 알고 포함한다(scope_gate_experiment.py
    # sanity 단계에서 eval set 전체에 충돌 사례가 없음을 직접 확인).
    '물', '과일', '채소', '생선',
]

# scope_gate_experiment.py DIALYSIS_PROCEDURE_TERMS와 동일 — "투석"만으로는 domain=True가
# 되지만, 영양과 무관한 시술/장비/행정 신호가 함께 있으면 KB에 관련 청크가 있어도(도관/
# 혈관통로/투석 기계/비용 챕터가 실제로 존재) SYSTEM 프롬프트가 선언한 영양 정보 범위
# 밖이라고 우선 판정한다(scope_gate_definitions.md §4-③).
DIALYSIS_PROCEDURE_TERMS = [
    '도관', '카테터', '동정맥루', '혈관통로', '접근로', '투석기계', '투석 기계',
    '건강보험', '보험 적용', '치료 비용', '수가',
    '주 몇 번', '몇 시간씩', '스케줄',
]

# scope_gate_experiment.py RED_FLAG_TERMS와 동일 — SYSTEM 프롬프트가 명시적으로 "개인별
# 판단 필요 → 의료진 상담"으로 분리한 약물/시술 영역의 일반 명사(브랜드명은 '타이레놀' 1개만
# 예시로 포함, 나머지는 전부 일반 명사).
RED_FLAG_TERMS = [
    '인슐린', '항암', '화학요법', '감기약', '타이레놀', '처방전', '처방약', '진통제', '수면제',
    '항생제', '백신', '접종', '연고', '안약', '파스', '마취', '수술', '시술', '방사선치료',
    '해열제', '소화제', '변비약', '알레르기약', '비염약', '피임약', '갑상선약', '갑상선',
    '천식약', '혈압약', '당뇨약', '항히스타민', '스테로이드', '항응고제', '와파린',
    '입덧', '정형외과', '피부과', '산부인과',
]

# scope_gate_experiment.py의 _norm()/rule_signals()와 동일 정규화 규칙(구두점을 공백으로
# 치환 후 공백 정리) — 별도 이름(_SCOPE_*)을 쓰는 이유는 위쪽 어휘 재정렬용 _LEX_PUNCT_RE/
# _LEX_WS_RE와 문자 클래스가 다른 별개 용도이기 때문(서로 재사용하면 두 필터가 실수로
# 얽힐 수 있어 의도적으로 분리).
_SCOPE_PUNCT_RE = re.compile(r'[.,!?()\[\]{}·:;"\'“”‘’…/\\|~\-‐-―]')
_SCOPE_WS_RE = re.compile(r'\s+')


def _scope_norm(s):
    s = _SCOPE_PUNCT_RE.sub(' ', s)
    return _SCOPE_WS_RE.sub(' ', s).strip()


ScopeResult = namedtuple('ScopeResult', ['verdict', 'reason'])


def classify_scope(question):
    """이 질문이 KOOK(혈액투석 영양) 도메인 IN_SCOPE인지 OUT_OF_SCOPE인지 판정한다.
    scope_gate_experiment.py의 rule_verdict()/rule_gate_binary()(Method A, mixed_default=True/
    no_signal_default=False로 검증된 그대로)를 production용으로 그대로 옮긴 것 — 새 규칙을
    추가하지 않았다. 4-way 내부 신호(has_domain/has_procedure/has_redflag)를 아래처럼
    이진 IN_SCOPE/OUT_OF_SCOPE로 접는다:
      - has_procedure (투석 시술/장비/보험 신호) -> OUT_OF_SCOPE
        ('dialysis_procedure_not_nutrition') — domain 신호가 같이 있어도 우선 적용.
      - has_redflag and not has_domain -> OUT_OF_SCOPE ('other_medical_treatment')
        (감기약/인슐린/항암 등 약물·타 치료 영역, domain 신호 전혀 없음).
      - has_domain and not has_redflag -> IN_SCOPE ('dialysis_nutrition').
      - has_domain and has_redflag (MIXED) -> IN_SCOPE ('mixed_intent_default_pass')
        (recall 최우선 가중치에 따른 기본값 — "투석 중인데 감기약 먹어도 돼?" 같은 위험
        사례는 알려진 한계로 남겨두고 이번 작업에서 새 규칙으로 "고치지" 않는다, 작업 지시
        참고 — Method C가 이를 다루는 대안으로 검토됐으나 기각됨).
      - 신호 없음(NO_SIGNAL) -> OUT_OF_SCOPE ('unrelated') — 일반 잡담 기본 거부."""
    q = _scope_norm(question)
    has_domain = any(t in q for t in DOMAIN_TERMS)
    has_procedure = any(t in q for t in DIALYSIS_PROCEDURE_TERMS)
    has_redflag = any(t in q for t in RED_FLAG_TERMS)

    if has_procedure:
        return ScopeResult(OUT_OF_SCOPE, 'dialysis_procedure_not_nutrition')
    if has_redflag and not has_domain:
        return ScopeResult(OUT_OF_SCOPE, 'other_medical_treatment')
    if has_domain and not has_redflag:
        return ScopeResult(IN_SCOPE, 'dialysis_nutrition')
    if has_domain and has_redflag:
        return ScopeResult(IN_SCOPE, 'mixed_intent_default_pass')
    return ScopeResult(OUT_OF_SCOPE, 'unrelated')


# NO_EVIDENCE_ANSWER("KB에서 근거를 못 찾음")와 같은 톤을 쓰되 별도 상수로 분리한다 — 이유가
# 다르면("이 질문 자체가 우리 도메인 밖" vs "도메인 안인데 근거 부족") 디버깅/로그에서 구분이
# 되어야 한다.
OUT_OF_SCOPE_ANSWER = (
    "이 질문은 저희가 제공하는 혈액투석 환자 영양 정보 범위를 벗어난 것 같아요 — "
    "담당 의료진이나 영양사와 상담해 주세요."
)


def answer(question, weight=None, consumed=None, meals_left=None, top_k=5, model='gpt-4o-mini'):
    """질문 -> (답변, 참고한 출처 리스트). 특정 재료 질문이면 영양DB 직접 조회, 아니면 RAG.
    weight+consumed를 주면(오늘 이미 먹은 양을 알면) 남은 예산까지 감안해서 답하고,
    meals_left(오늘 남은 끼니 수)까지 있으면 하루 전체가 아니라 '다음 한 끼 몫'으로 비교한다.

    RAG 검색 파이프라인(2026-08-14, 경량 재정렬 도입) — 순서가 중요하다:
      1) retrieve(question, top_k=RAG_CANDIDATE_POOL)로 원 임베딩 후보 풀(10개)을 가져온다.
      2) _relevant()로 게이트 필터링 — 오직 원 코사인 점수(h['score'])만 RAG_MIN_SCORE와 비교한다.
         이 단계에서 탈락한 후보는 3)의 재정렬 대상에 아예 들어가지 않는다.
      3) 게이트를 통과한 후보에 한해서만 _rerank()로 어휘 중복 보너스를 얹어 재정렬한다
         (RAG_LEXICAL_ALPHA=0.05, backend/evaluation/reranking_experiment.py 실험에서 안전성 확인됨).
      4) 재정렬된 순서에서 최대 RAG_CONTEXT_MAX(5)개만 최종 컨텍스트로 쓴다.
    top_k 매개변수는 하위 호환을 위해 남겨뒀지만 더 이상 후보 풀 크기를 결정하지 않는다(그
    역할은 RAG_CANDIDATE_POOL/RAG_CONTEXT_MAX 상수가 맡는다) — 이 함수를 top_k로 호출하는
    기존 코드/테스트가 없음을 확인했다.

    Scope gate(2026-08-14) — find_food()가 실패한 경우에만 적용: classify_scope()로 이
    질문이 KOOK 도메인(혈액투석 영양) 안인지 먼저 판정하고, OUT_OF_SCOPE면 retrieve()
    (임베딩 API 호출)도 LLM 호출도 하지 않고 바로 고정 안내문을 반환한다 — food_db 라우팅은
    이 게이트보다 앞선 find_food() 체크가 이미 가로채므로 절대 영향받지 않는다."""
    food = find_food(question)
    if food:
        result = food_lookup_answer(question, food, weight=weight, consumed=consumed,
                                     meals_left=meals_left, model=model)
        if result:
            text, rag_sources = result
            return text, sorted({'식약청 국가표준식품성분표 (FOOK 영양DB)', *rag_sources})
    scope = classify_scope(question)
    if scope.verdict == OUT_OF_SCOPE:
        return OUT_OF_SCOPE_ANSWER, []
    hits = retrieve(question, top_k=RAG_CANDIDATE_POOL)
    relevant = _relevant(hits)   # 게이트: 원 코사인 점수만 사용, 어휘 보너스는 아직 계산조차 안 됨
    if not relevant:
        # 후보 전부가 RAG_MIN_SCORE 미만 — 근거 없이 LLM이 지어내게 두지 않고 여기서 바로 거절한다.
        # (여긴 주 답변 경로라 food_lookup_answer()의 보조 RAG와 달리 "관련자료 없음"이 곧 "답변 불가".)
        return NO_EVIDENCE_ANSWER, []
    reranked = _rerank(question, relevant)   # 게이트 통과분만 재정렬 — 게이트 결과 자체는 안 바뀜
    context_hits = reranked[:RAG_CONTEXT_MAX]
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in context_hits)
    prompt = f"[참고 자료]\n{context}\n\n[환자 질문]\n{question}"
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    sources = sorted({h['source'] for h in context_hits})
    return text, sources
