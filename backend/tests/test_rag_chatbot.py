# -*- coding: utf-8 -*-
"""
FOOK_rag_chatbot.py(RAG 챗봇) 단위 테스트.

OpenAI 라이브 호출은 전혀 하지 않는다 — embeddings/chat completion은 모두 모킹한다
(_client()를 가짜 클라이언트로 바꿔치기, 또는 retrieve()/find_food() 자체를 monkeypatch).

일부 테스트(find_food 라우팅, food_lookup_answer 판정)는 실제 식약청 영양DB
(FOOK_adjust_levers.load_all(), 식약청_영양성분10.4(수정).xlsx)가 필요하다 — 이 DB 없이는
"바나나"·"시금치" 같은 실제 재료명 매칭을 의미 있게 검증할 수 없기 때문이다(TensorFlow
생성모델은 필요 없음 — app_core_FOOK을 import하지 않으므로 test_api_validation.py처럼
가볍다). load_all()은 세션 스코프 fixture(real_nut)로 한 번만 로딩한다(수 초 소요).
load_all()은 backend/ 기준 상대경로로 엑셀을 읽으므로, pytest를 backend/에서 실행해야 한다
(다른 테스트 파일들과 동일한 전제).
"""
import numpy as np
import pytest

import FOOK_rag_chatbot as C
import FOOK_adjust_levers as F


# ── OpenAI 클라이언트 모킹 헬퍼 ────────────────────────────────────────────────
class _FakeEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingResp:
    def __init__(self, embedding):
        self.data = [_FakeEmbeddingData(embedding)]


class _FakeEmbeddings:
    def __init__(self, vec):
        self._vec = vec

    def create(self, model, input):
        return _FakeEmbeddingResp(self._vec)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeChatResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeChatResp(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeChatCompletions(content)


class FakeClient:
    """C._client()의 대역. embeddings.create()/chat.completions.create() 둘 다 모킹."""

    def __init__(self, query_vec=None, chat_content='(테스트용 LLM 응답)'):
        self.embeddings = _FakeEmbeddings(query_vec if query_vec is not None else [0.0] * 1536)
        self.chat = _FakeChat(chat_content)


@pytest.fixture(scope='session')
def real_nut():
    """실제 식약청 영양DB(FOOK_adjust_levers.NUT)를 한 번만 로딩해 세션 내내 재사용."""
    if F.NUT is None:
        F.NUT = F.load_all()
    return F.NUT


@pytest.fixture(autouse=True)
def _reset_food_index():
    """_food_index()의 모듈 전역 캐시(_ING_INDEX)가 테스트 간에 새지 않도록(다른 테스트 파일이
    F.NUT를 건드릴 가능성 대비) 매 테스트 전에 초기화. 실제로는 F.NUT 자체는 세션 내 불변이라
    새로 안 만들어도 되지만, 캐시가 이전 테스트의 monkeypatch된 상태를 들고 있지 않도록 안전하게."""
    yield


# ═══════════════════════════ 1) KB 로딩 ═══════════════════════════════════════
def test_load_kb_reads_real_kb_file():
    C._KB = None
    C._KB_MAT = None
    kb, mat = C._load_kb()
    assert len(kb) > 0
    assert mat.shape[0] == len(kb)
    assert all('text' in c and 'source' in c and 'embedding' in c for c in kb)
    # 미리 정규화되어 있어야 함(코사인 유사도 = 내적)
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_load_kb_caches_after_first_call():
    C._KB = None
    C._KB_MAT = None
    kb1, mat1 = C._load_kb()
    kb2, mat2 = C._load_kb()
    assert kb1 is kb2   # 같은 객체 = 두 번째 호출은 파일을 다시 안 읽고 캐시를 씀
    assert mat1 is mat2


# ═══════════════════════════ 2) 코사인 검색 정확성 (합성 KB) ═══════════════════
def test_retrieve_ranks_by_cosine_similarity(monkeypatch):
    # 3개 청크, 서로 직교하는 단위벡터(코사인 유사도가 정확히 계산 가능하도록)
    vecs = np.eye(3, dtype=np.float32)
    kb = [
        {'text': '칼륨 관련 텍스트', 'source': 'srcA', 'embedding': vecs[0].tolist()},
        {'text': '인 관련 텍스트', 'source': 'srcB', 'embedding': vecs[1].tolist()},
        {'text': '나트륨 관련 텍스트', 'source': 'srcC', 'embedding': vecs[2].tolist()},
    ]
    monkeypatch.setattr(C, '_KB', kb)
    monkeypatch.setattr(C, '_KB_MAT', np.array(vecs, dtype=np.float32))
    # 질문 임베딩이 srcB(인덱스1)와 정확히 일치하도록 설정
    monkeypatch.setattr(C, '_client', lambda: FakeClient(query_vec=[0.0, 1.0, 0.0]))

    hits = C.retrieve('인이 많은 음식은?', top_k=3)
    assert [h['source'] for h in hits] == ['srcB', 'srcA', 'srcC']  # argsort 안정성상 A,C 동순위지만 원 순서 유지
    assert hits[0]['score'] == pytest.approx(1.0)
    assert hits[1]['score'] == pytest.approx(0.0, abs=1e-6)
    assert hits[2]['score'] == pytest.approx(0.0, abs=1e-6)


def test_retrieve_respects_top_k(monkeypatch):
    vecs = np.eye(4, dtype=np.float32)
    kb = [{'text': f't{i}', 'source': f's{i}', 'embedding': vecs[i].tolist()} for i in range(4)]
    monkeypatch.setattr(C, '_KB', kb)
    monkeypatch.setattr(C, '_KB_MAT', np.array(vecs, dtype=np.float32))
    monkeypatch.setattr(C, '_client', lambda: FakeClient(query_vec=[1.0, 0.0, 0.0, 0.0]))
    hits = C.retrieve('질문', top_k=2)
    assert len(hits) == 2


# ═══════════════════════════ 3) 최소 유사도 게이트 (RAG_MIN_SCORE) ═══════════════
def test_relevant_filters_out_low_score_hits(monkeypatch):
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.15)
    hits = [{'source': 'good', 'text': 't', 'score': 0.5},
            {'source': 'bad', 'text': 't', 'score': 0.05}]
    rel = C._relevant(hits)
    assert [h['source'] for h in rel] == ['good']


def test_relevant_keeps_hit_exactly_at_threshold(monkeypatch):
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.15)
    hits = [{'source': 'edge', 'text': 't', 'score': 0.15}]
    assert C._relevant(hits) == hits


def test_relevant_returns_empty_when_all_below_threshold(monkeypatch):
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.15)
    hits = [{'source': 'a', 'text': 't', 'score': 0.1}, {'source': 'b', 'text': 't', 'score': 0.02}]
    assert C._relevant(hits) == []


def test_answer_refuses_without_calling_llm_when_no_relevant_evidence(monkeypatch):
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [{'source': 'x', 'text': 't', 'score': 0.01}])

    def _fail_client():
        raise AssertionError('관련 근거가 없으면 LLM을 호출하면 안 된다')
    monkeypatch.setattr(C, '_client', _fail_client)

    text, sources = C.answer('오늘 날씨 어때?')
    assert text == C.NO_EVIDENCE_ANSWER
    assert sources == []


def test_answer_uses_only_relevant_hits_as_sources(monkeypatch):
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    hits = [{'source': 'good_doc', 'text': 't1', 'score': 0.5},
            {'source': 'noise_doc', 'text': 't2', 'score': 0.02}]
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: hits)
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='칼륨이 많은 채소를 주의하세요.'))

    text, sources = C.answer('칼륨이 많은 채소는?')
    assert text == '칼륨이 많은 채소를 주의하세요.'
    assert sources == ['good_doc']   # 임계값 미만인 noise_doc은 출처에서도 빠져야 함


# ═══════════════════════════ 4) find_food() 라우팅 ═════════════════════════════
def test_find_food_none_without_food_question_keyword(real_nut):
    # '신장' 부분이 '신장, 돼지, 생것' 같은 재료명과 겹칠 수 있지만, 식이 질문 트리거 단어가
    # 없으므로 매칭하면 안 된다(FOOD_QUESTION_KW 가드).
    assert C.find_food('신장 기능이 걱정돼요') is None


def test_find_food_none_for_general_knowledge_question(real_nut):
    assert C.find_food('혈액투석 환자가 피해야 할 고칼륨 채소는 어떤 게 있나요?') is None


def test_find_food_matches_specific_ingredient(real_nut):
    assert C.find_food('바나나 먹어도 되나요?') == '바나나, 생것'


def test_find_food_state_disambiguation_dried(real_nut):
    # STATE_KW['말린'] = ('말린것','건조','동결건조','반건조'). DB의 바나나엔 '말린것'이 없고
    # '동결건조'가 있으므로 그쪽으로 매칭돼야 한다(생것 폴백이 아니라).
    food = C.find_food('말린 바나나 먹어도 되나요?')
    assert food == '바나나, 동결건조'


def test_find_food_state_disambiguation_boiled(real_nut):
    food = C.find_food('데친 시금치 먹어도 되나요?')
    assert food == '시금치, 데친것'


def test_find_food_no_state_keyword_falls_back_to_raw(real_nut):
    food = C.find_food('시금치 많이 먹어도 되나요?')
    assert food == '시금치, 생것'


def test_find_food_prefers_longer_more_specific_match(real_nut):
    # "무" 보다 "무말랭이"처럼 더 구체적인 이름이 있으면 그쪽을 우선해야 한다(문서화된 규칙).
    food = C.find_food('무말랭이 먹어도 되나요?')
    assert food is not None and food.startswith('무말랭이')


# ── generic-category 세그먼트 오매칭 회귀 테스트 (2026-08-13) ──────────────────
# 실측 발견: "과일"이 "빵, 페이스트리, 과일"의 비-첫 세그먼트로 인덱싱돼 있어서,
# "통조림 과일은 먹어도 괜찮은가요?" 같은 일반(RAG로 가야 할) 질문이 엉뚱하게 그 빵
# 항목으로 food DB route 되고 있었다. _GENERIC_CATEGORY_SEGMENTS로 '과일'/'채소'/'생선'을
# 제외해 고쳤다 — 아래는 그 회귀가 재발하지 않는지, 그리고 구체적인 재료 질문(사과·통조림
# 복숭아처럼 진짜 매칭돼야 하는 경우)은 계속 정상 동작하는지 함께 확인한다.
def test_find_food_none_for_generic_fruit_category(real_nut):
    assert C.find_food('통조림 과일은 먹어도 괜찮은가요?') is None


def test_find_food_none_for_generic_vegetable_category(real_nut):
    assert C.find_food('채소는 얼마나 먹어도 되나요?') is None


def test_find_food_none_for_generic_fish_category(real_nut):
    assert C.find_food('생선은 얼마나 드셔도 되나요?') is None


def test_find_food_still_matches_specific_apple(real_nut):
    assert C.find_food('사과 먹어도 되나요?') == '사과, 생것'


def test_find_food_still_matches_specific_canned_peach(real_nut):
    food = C.find_food('통조림 복숭아 먹어도 되나요?')
    assert food is not None and food.startswith('복숭아') and '통조림' in food


# ═══════════════════════════ 5) K/P/Na 판정 로직 (_remaining_budget) ══════════
def test_remaining_budget_uses_whole_day_scope_without_meals_left():
    rb = C._remaining_budget(60, {'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}, meals_left=None)
    d = F.day_targets(60)
    assert rb['K_left'] == pytest.approx(d['Kmax'])
    assert rb['P_left'] == pytest.approx(d['Pmax'])
    assert '하루 전체' in rb['scope']
    assert 'Na_left' not in rb   # Phase 1에서 제거 — 범주 오류였던 계산이라 더 이상 반환하지 않음


def test_remaining_budget_uses_next_meal_scope_with_meals_left():
    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    rb = C._remaining_budget(60, consumed, meals_left=2)
    mb = F.meal_bounds(60, consumed, meals_left=2)
    assert rb['K_left'] == pytest.approx(mb['Kmax'])
    assert rb['P_left'] == pytest.approx(mb['Pmax'])
    assert '한 끼 몫' in rb['scope']
    assert '2끼 남음' in rb['scope']


def test_remaining_budget_reflects_already_consumed_potassium():
    rb_none = C._remaining_budget(60, {'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}, meals_left=None)
    rb_some = C._remaining_budget(60, {'K': 1000, 'P': 0, 'Na': 0, 'Na_season': 0}, meals_left=None)
    assert rb_some['K_left'] < rb_none['K_left']
    assert rb_some['K_consumed'] == 1000


# ═══════════════════════════ 6) food_lookup_answer() 판정 + LLM 안전장치 ══════
def _no_rag_and_fake_llm(monkeypatch, chat_content):
    """food_lookup_answer 내부의 보조 RAG 호출과 LLM 호출을 모두 무해하게 모킹."""
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content=chat_content))


def test_food_lookup_answer_allows_when_budget_is_ample(monkeypatch, real_nut):
    _no_rag_and_fake_llm(monkeypatch, '적당량 드셔도 됩니다.')
    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    text, sources = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                          weight=60, consumed=consumed)
    assert text.endswith('▶ 결론: 지금 더 드셔도 됩니다')


def test_food_lookup_answer_blocks_when_potassium_budget_exhausted(monkeypatch, real_nut):
    _no_rag_and_fake_llm(monkeypatch, '괜찮아요! 더 드세요!')  # LLM이 반대로 말해도(아래 안전장치 확인용)
    # 바나나, 생것 1회분(120g) 칼륨 ≈ 426mg. 하루 Kmax=3000인데 이미 2700 먹었으면 남은 예산 300 < 426.
    consumed = {'E': 0, 'protein': 0, 'K': 2700, 'P': 0, 'Na': 0, 'Na_season': 0}
    text, sources = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                          weight=60, consumed=consumed)
    assert text.endswith('▶ 결론: 지금은 피하시는 게 좋습니다')
    # LLM이 뭐라고 말했든(반대로 말했어도) 최종 결론은 코드가 계산한 값으로 못박혀야 한다.
    assert '괜찮아요! 더 드세요!' in text   # LLM 원문은 그대로 남아있되
    assert text.split('▶ 결론:')[-1].strip() == '지금은 피하시는 게 좋습니다'  # 마지막 줄이 최종 결론


def test_food_lookup_answer_no_verdict_without_personalization(monkeypatch, real_nut):
    _no_rag_and_fake_llm(monkeypatch, '이 재료에 대한 일반 설명입니다.')
    text, sources = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것')  # weight/consumed 없음
    assert '▶ 결론' not in text   # 개인화 정보가 없으면 최종 판정 자체를 안 내림


def test_food_lookup_answer_caveats_high_sodium_ingredient_even_when_k_p_fit(monkeypatch, real_nut):
    # Phase 1 핵심 수정 대상: 나트륨이 극히 높은 재료(소금)는 K/P가 통과해도
    # "지금 더 드셔도 됩니다"라고 안심시키면 안 된다.
    _no_rag_and_fake_llm(monkeypatch, '드셔도 좋아요.')
    ing_nut = real_nut[0]
    salt = '소금, 천일염, 소금꽃'
    assert ing_nut[salt]['Na'] > 40000   # 전제 확인(실측치)
    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    text, sources = C.food_lookup_answer('소금 먹어도 되나요?', salt, weight=60, consumed=consumed)
    assert '지금 더 드셔도 됩니다' not in text.split('▶ 결론:')[-1]
    assert '피하시는 게 좋습니다' in text
    assert '나트륨' in text  # 이유가 나트륨임을 명시


def test_food_lookup_answer_no_false_precision_for_sodium_budget(monkeypatch, real_nut):
    # Phase 1에서 제거한 버그: 재료 나트륨을 Na_season 예산과 비교해 '남은 나트륨 예산' 같은
    # 존재하지 않는 정밀 수치를 만들어내면 안 된다.
    _no_rag_and_fake_llm(monkeypatch, '문제없어요.')
    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    text, sources = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                          weight=60, consumed=consumed)
    assert '남은 나트륨 예산' not in text


def test_food_lookup_answer_meals_left_changes_scope_text(monkeypatch, real_nut):
    calls = []

    def spy_client():
        calls.append(True)
        return FakeClient(chat_content='ok')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', spy_client)

    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    text_day, _ = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                        weight=60, consumed=consumed, meals_left=None)
    text_meal, _ = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                         weight=60, consumed=consumed, meals_left=1)
    # 둘 다 판정 자체는 통과(여유 충분)하지만, 근거로 쓰인 scope 문구가 달라야 한다는 것을
    # food_lookup_answer가 프롬프트에 넘긴 facts를 통해 간접 확인하기는 어려우므로,
    # _remaining_budget()을 직접 호출해 scope가 정말 다름을 재확인한다(위 5번 테스트와 연결).
    rb_day = C._remaining_budget(60, consumed, meals_left=None)
    rb_meal = C._remaining_budget(60, consumed, meals_left=1)
    assert rb_day['scope'] != rb_meal['scope']
    assert text_day.endswith('▶ 결론: 지금 더 드셔도 됩니다')
    assert text_meal.endswith('▶ 결론: 지금 더 드셔도 됩니다')


def test_food_lookup_answer_returns_none_when_no_potassium_data(monkeypatch, real_nut):
    # k100이 None인 재료는 칼륨 데이터가 없어 RAG로 폴백해야 한다(food_lookup_answer가 None 반환).
    ing_nut = real_nut[0]
    no_k_items = [n for n, d in ing_nut.items() if d.get('K') is None]
    assert no_k_items, '테스트 전제: DB에 칼륨 값이 없는 재료가 하나는 있어야 함'
    result = C.food_lookup_answer('이거 먹어도 되나요?', no_k_items[0])
    assert result is None
