# -*- coding: utf-8 -*-
"""
FOOK_rag_chatbot.py — 환자 질문 응답 챗봇 (RAG).
data/FOOK_rag_kb.json(청크+임베딩)에서 질문과 유사한 조각을 찾아 GPT에게 근거로 주고 답하게 한다.

API 키: 환경변수 OPENAI_API_KEY.
"""
import os, json
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
# 실측 데이터로 도출(2026-08-13, 실 서비스 KB — data/FOOK_rag_kb.json 539개 청크, 실제
# text-embedding-3-small 임베딩 그대로 사용. 이 환경엔 OPENAI_API_KEY가 없어 "오늘 날씨 어때"
# 같은 실제 이탈 질문을 라이브로 임베딩해 확인하지는 못했다 — 대신 실제 KB 임베딩 행렬 자체로
# 두 가지를 측정했다:
#   1) KB 내 모든 청크쌍(539×538/2쌍)의 실제 코사인 유사도: 평균 0.425, 하위10%(p10)조차 0.282,
#      서로 다른 출처 문서끼리(관련성이 가장 낮은 실제 사례)도 평균 0.399 — 즉 이 임베딩 공간은
#      "같은 도메인·같은 언어"라는 이유만으로도 기저 유사도가 상당히 높게 깔린다(OpenAI 임베딩의
#      잘 알려진 양의 편향). 진짜 무관한 질문(날씨·축구 등)도 "한국어"라는 공통점 때문에 이
#      기저선 근처까지는 올라올 가능성이 있다.
#   2) 수학적으로 완전히 무관한 벡터(1536차원 표준정규분포에서 뽑은 무작위 단위벡터 200개)를 같은
#      KB 539개 청크와 비교: 평균 유사도 ≈0(-0.0007), 청크당 최댓값의 평균은 0.058, 전체 관측
#      최댓값은 0.122 — 이게 "의미적으로 전혀 연관 없음"의 실측 하한(노이즈 바닥)이다.
#   실제 이탈 질문(같은 한국어지만 의미상 무관)은 이 둘 사이 어딘가에 위치할 것으로 추정되며,
#   0.15는 노이즈 바닥(≤0.122)보다 확실히 위에 있으면서 실제 도메인 내 최소 관련성(p10=0.282)보다는
#   충분히 아래라 안전 마진이 크다. evaluation/evaluate_rag.py는 OPENAI_API_KEY가 있는 환경에서
#   실제 이탈 질문 점수를 측정해 이 기본값을 검증/재조정하도록 만들어졌다 — 그 결과가 나오면
#   이 기본값을 갱신할 것.
# 환경변수 RAG_MIN_SCORE로 재정의 가능(운영 중 데이터가 쌓이면 조정할 수 있도록).
RAG_MIN_SCORE = float(os.environ.get('RAG_MIN_SCORE', '0.15'))

NO_EVIDENCE_ANSWER = (
    "제공된 자료에서는 확인할 수 없습니다. 이 질문은 저희가 갖고 있는 임상 영양 자료 범위를 "
    "벗어난 것 같아요 — 담당 의료진이나 영양사와 상담해 주세요."
)


def _relevant(hits, min_score=None):
    """retrieve()가 반환한 원 점수 중 RAG_MIN_SCORE 이상만 남긴다 — "그럴듯한 top_k"가 아니라
    "실제로 관련 있는" 것만 LLM 프롬프트에 넣기 위한 정책 필터. min_score를 안 주면 RAG_MIN_SCORE."""
    thresh = RAG_MIN_SCORE if min_score is None else min_score
    return [h for h in hits if h['score'] >= thresh]


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


def answer(question, weight=None, consumed=None, meals_left=None, top_k=5, model='gpt-4o-mini'):
    """질문 -> (답변, 참고한 출처 리스트). 특정 재료 질문이면 영양DB 직접 조회, 아니면 RAG.
    weight+consumed를 주면(오늘 이미 먹은 양을 알면) 남은 예산까지 감안해서 답하고,
    meals_left(오늘 남은 끼니 수)까지 있으면 하루 전체가 아니라 '다음 한 끼 몫'으로 비교한다."""
    food = find_food(question)
    if food:
        result = food_lookup_answer(question, food, weight=weight, consumed=consumed,
                                     meals_left=meals_left, model=model)
        if result:
            text, rag_sources = result
            return text, sorted({'식약청 국가표준식품성분표 (FOOK 영양DB)', *rag_sources})
    hits = retrieve(question, top_k=top_k)
    relevant = _relevant(hits)
    if not relevant:
        # 후보 전부가 RAG_MIN_SCORE 미만 — 근거 없이 LLM이 지어내게 두지 않고 여기서 바로 거절한다.
        # (여긴 주 답변 경로라 food_lookup_answer()의 보조 RAG와 달리 "관련자료 없음"이 곧 "답변 불가".)
        return NO_EVIDENCE_ANSWER, []
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in relevant)
    prompt = f"[참고 자료]\n{context}\n\n[환자 질문]\n{question}"
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    sources = sorted({h['source'] for h in relevant})
    return text, sources
