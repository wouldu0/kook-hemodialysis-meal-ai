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
    """질문과 코사인 유사도가 가장 높은 청크 top_k개 반환. [{'text','source','score'}, ...]"""
    kb, mat = _load_kb()
    client = _client()
    q_emb = client.embeddings.create(model='text-embedding-3-small', input=[query]).data[0].embedding
    q = np.array(q_emb, dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = mat @ q       # 코사인 유사도(정규화됐으므로 내적=코사인)
    idx = np.argsort(-scores)[:top_k]
    return [{'text': kb[i]['text'], 'source': kb[i]['source'], 'score': float(scores[i])} for i in idx]


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
}


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
                # 첫 세그먼트(기본식품명)는 항상 인덱싱. 그 외는 상태어가 아니고 충분히 구체적일 때만.
                if i > 0 and (short in _GENERIC_SEGMENTS or len(short) < 2):
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
    (마지막 끼니라고 가정하는 게 더 보수적이진 않지만, 정보가 없을 때의 기본 동작)."""
    import FOOK_adjust_levers as F
    c = {k: float(consumed.get(k) or 0) for k in CONSUMED_KEYS}
    if meals_left:
        mb = F.meal_bounds(weight, consumed, meals_left=meals_left)
        return {
            'K_left': mb['Kmax'], 'P_left': mb['Pmax'],
            'Na_left': mb['Namax'] - c.get('Na_season', 0),
            'K_consumed': c['K'], 'P_consumed': c['P'],
            'scope': f'다음 한 끼 몫 (오늘 {meals_left}끼 남음, 남은예산÷남은끼니 기준)',
        }
    d = F.day_targets(weight)
    return {
        'K_left': d['Kmax'] - c['K'],
        'P_left': d['Pmax'] - c['P'],
        'Na_left': d['Namax'] - c.get('Na_season', c['Na']),
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
        verdict = '지금 더 드셔도 됩니다' if (fits_k and fits_p) else '지금은 피하시는 게 좋습니다'
        facts += f"\n\n[최종 판정 — 코드가 이미 계산 완료, 이 결론을 그대로 따를 것]\n{verdict}"

    # RAG: 이 재료·관련 조리법에 대한 지식베이스 발췌를 참고자료로 덧붙인다(수치 판정을 뒤집는 용도
    # 아님 — "말린 과일은 칼륨이 2배" 같은 조회만으론 안 잡히는 세부 지침 보완용).
    hits = retrieve(question, top_k=top_k)
    rag_block = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    rag_sources = sorted({h['source'] for h in hits})

    instruction = (
        "위 [영양DB 실측 자료]의 숫자를 우선 근거로 이 재료가 저/중/고칼륨 중 어디에 속하는지와 "
        "혈액투석 환자가 어떻게 섭취하면 좋을지 안내해라. [관련 임상 자료 발췌]는 조리법·주의사항 "
        "같은 보완 정보로만 참고하고, 그 안에 이 상황과 무관한 내용이 있으면 무시해라. "
        "숫자를 임의로 바꾸거나 새로 지어내지 마라."
    )
    if personalized:
        instruction = (
            f"[최종 판정]은 이미 코드가 계산을 마친 확정된 결론이다 — 절대 네가 숫자를 다시 비교하거나 "
            f"판정을 뒤집지 마라. 답변은 반드시 '{verdict}'라는 결론으로 끝나야 한다. "
            "그 위의 [오늘 이 환자의 실제 섭취 현황] 숫자를 근거로 왜 그런 결론인지 설명하고, "
            "[관련 임상 자료 발췌]에 말린 과일·조리법 등 이 재료 섭취에 실제로 도움되는 주의사항이 "
            "있으면 짧게 덧붙여라(무관하면 무시). 숫자를 임의로 바꾸거나 새로 지어내지 마라."
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
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    prompt = f"[참고 자료]\n{context}\n\n[환자 질문]\n{question}"
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    sources = sorted({h['source'] for h in hits})
    return text, sources
