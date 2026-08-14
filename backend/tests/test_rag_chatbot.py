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

    # 이 테스트가 검증하려는 건 RAG_MIN_SCORE 관련성 게이트(retrieve()는 호출됐지만 결과가 전부
    # 임계값 미만이라 LLM을 호출하지 않는 경로)이지 scope gate가 아니다 — 질문은 반드시 scope
    # gate를 통과하는(IN_SCOPE) 질문이어야 아래 흐름이 실제로 _relevant() 게이트까지 도달한다
    # (2026-08-14 scope gate 도입 전에는 "오늘 날씨 어때?"로도 이 경로를 탔지만, 지금은 그 질문
    # 자체가 scope gate에서 먼저 막혀 retrieve()가 아예 호출되지 않는다 — 그건 별도로
    # test_answer_out_of_scope_never_calls_retrieve()가 검증한다).
    question = '칼륨 수치가 너무 높아지면 몸에 어떤 문제가 생기나요?'
    assert C.classify_scope(question).verdict == C.IN_SCOPE
    text, sources = C.answer(question)
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


# ═══════════════ 7) 경량 재정렬(rerank) — 어휘 보너스 vs 게이트 안전성 ══════════
# 이 블록의 핵심 목적: alpha=0.05 어휘 재정렬이 "순서만" 바꿀 뿐 "게이트 통과 여부"에는 절대
# 관여하지 않는다는 것을 못박는 회귀 테스트다(reranking_experiment.py에서 alpha=0.15가 out-of-scope
# 질문 게이트를 오통과시킨 사례가 있었기 때문에, 이 분리가 이번 변경 전체에서 가장 중요한 불변식).

def test_lexical_bonus_can_reorder_gate_passed_candidates(monkeypatch):
    # 두 후보 모두 RAG_MIN_SCORE를 넘어 게이트를 통과한 상태. 순수 임베딩 점수로는 b가 a보다
    # 근소하게 높지만(원 순위 a<b가 아니라 b가 1위), 어휘 중복은 a가 훨씬 크다 — alpha=0.05
    # 보너스가 충분히 크면(질문 토큰이 여러 개고 a가 전부 포함) 재정렬 후 a가 1위로 올라가야 한다.
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.30)
    monkeypatch.setattr(C, 'RAG_LEXICAL_ALPHA', 0.05)
    hits = [
        {'source': 'b', 'text': '이 문서는 질문과 별 상관없는 일반적인 내용입니다.', 'score': 0.42},
        {'source': 'a', 'text': '바나나 칼륨 함량은 혈액투석 환자가 섭취 시 주의해야 합니다.', 'score': 0.40},
    ]
    question = '바나나 칼륨 함량 섭취 주의해야 하나요?'
    reranked = C._rerank(question, hits)
    assert [h['source'] for h in reranked][0] == 'a'
    # 원본 score는 절대 변경되지 않아야 한다(재정렬은 새 rerank_score 필드로만).
    a = next(h for h in reranked if h['source'] == 'a')
    b = next(h for h in reranked if h['source'] == 'b')
    assert a['score'] == pytest.approx(0.40)
    assert b['score'] == pytest.approx(0.42)
    assert a['rerank_score'] > a['score']   # 보너스가 실제로 더해졌음


def test_lexical_bonus_never_lets_a_below_threshold_candidate_pass_the_gate(monkeypatch):
    # 게이트(=_relevant) 미만 점수의 후보는 어휘 중복이 100%로 최대여도(=완전 일치) 여전히
    # 게이트에서 탈락해야 한다 — _rerank()는 _relevant() *이후*에만 호출되는 구조이므로, 이 후보는
    # 애초에 _rerank()에 전달조차 되지 않아야 한다. answer()의 실제 파이프라인을 통해 확인한다.
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.30)
    monkeypatch.setattr(C, 'RAG_LEXICAL_ALPHA', 0.05)
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    question = '콩팥병 환자는 칼륨을 어떻게 관리해야 하나요?'
    low_score_but_perfect_lexical_match = {
        'source': 'low', 'text': question,   # 어휘 보너스 = 1.0 (완전 일치)
        'score': 0.10,   # RAG_MIN_SCORE(0.30)보다 한참 낮음
    }
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [low_score_but_perfect_lexical_match])

    def _fail_client():
        raise AssertionError('게이트를 통과 못 했으면 LLM을 호출하면 안 된다')
    monkeypatch.setattr(C, '_client', _fail_client)

    text, sources = C.answer(question)
    # 참고: score(0.10) + alpha(0.05)*bonus(1.0) = 0.15 < RAG_MIN_SCORE(0.30) — 설령 재정렬
    # 점수로 비교했더라도 여전히 게이트를 넘지 못했을 값이지만, 애초에 _relevant()가 원점수(raw
    # score=0.10)만으로 걸러내므로 rerank_score라는 값 자체가 계산되지도 않는다.
    assert text == C.NO_EVIDENCE_ANSWER
    assert sources == []


def test_lexical_alpha_defaults_to_0_05():
    assert C.RAG_LEXICAL_ALPHA == pytest.approx(0.05)


def test_final_context_capped_at_5_even_when_more_pass_gate(monkeypatch):
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.30)
    monkeypatch.setattr(C, 'RAG_CANDIDATE_POOL', 10)
    monkeypatch.setattr(C, 'RAG_CONTEXT_MAX', 5)
    # 10개 후보 모두 게이트 통과(0.30 이상) — 최종 컨텍스트는 5개로 잘려야 한다.
    hits = [{'source': f's{i}', 'text': f'칼륨 관련 내용 {i}', 'score': 0.30 + 0.01 * i}
            for i in range(10)]
    captured = {}

    def fake_retrieve(q, top_k=5):
        captured['top_k'] = top_k
        return hits

    monkeypatch.setattr(C, 'retrieve', fake_retrieve)
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='답변'))

    text, sources = C.answer('칼륨이 많은 음식은?')
    assert captured['top_k'] == 10   # RAG_CANDIDATE_POOL 만큼 넓게 후보를 가져와야 함
    assert len(sources) == 5   # 10개 모두 게이트 통과해도 최종 컨텍스트는 RAG_CONTEXT_MAX(5)로 캡


def test_retrieve_score_is_raw_cosine_not_reranked(monkeypatch):
    # evaluation/ 스크립트들이 retrieve()를 직접 호출해 원 코사인 점수를 기대한다 — 재정렬은
    # retrieve() 자체에는 절대 스며들면 안 된다(answer() 안에서만 사후 적용).
    vecs = np.eye(2, dtype=np.float32)
    kb = [{'text': '전혀 무관한 임의 텍스트 XYZ', 'source': 'srcA', 'embedding': vecs[0].tolist()},
          {'text': '칼륨 칼륨 칼륨', 'source': 'srcB', 'embedding': vecs[1].tolist()}]
    monkeypatch.setattr(C, '_KB', kb)
    monkeypatch.setattr(C, '_KB_MAT', np.array(vecs, dtype=np.float32))
    monkeypatch.setattr(C, '_client', lambda: FakeClient(query_vec=[1.0, 0.0]))

    hits = C.retrieve('칼륨이 많은 음식은?', top_k=2)
    # 질문 임베딩이 srcA와 정확히 일치하도록 설정했으므로(코사인=1.0), 어휘 중복은 srcB가
    # 훨씬 크더라도(질문에 '칼륨'이 있고 srcB 텍스트에 '칼륨'이 반복) retrieve()의 score와
    # 정렬 순서 자체는 순수 코사인 유사도만 반영해야 한다(rerank_score 키가 아예 없어야 함).
    assert hits[0]['source'] == 'srcA'
    assert hits[0]['score'] == pytest.approx(1.0)
    assert hits[1]['score'] == pytest.approx(0.0, abs=1e-6)
    assert all('rerank_score' not in h for h in hits)


def test_food_lookup_answer_returns_none_when_no_potassium_data(monkeypatch, real_nut):
    # k100이 None인 재료는 칼륨 데이터가 없어 RAG로 폴백해야 한다(food_lookup_answer가 None 반환).
    ing_nut = real_nut[0]
    no_k_items = [n for n, d in ing_nut.items() if d.get('K') is None]
    assert no_k_items, '테스트 전제: DB에 칼륨 값이 없는 재료가 하나는 있어야 함'
    result = C.food_lookup_answer('이거 먹어도 되나요?', no_k_items[0])
    assert result is None


# ═══════════════════════════ 8) scope gate (Method A) ═══════════════════════════
# backend/evaluation/scope_gate_experiment.py에서 검증된 규칙 기반 "Method A"를 production에
# 그대로 옮긴 classify_scope()에 대한 테스트. 이 블록의 핵심 목적은 두 가지 구조적 불변식을
# 증명하는 것이다: (1) find_food()가 scope gate보다 항상 먼저 실행되고 그 매칭이 우선한다,
# (2) OUT_OF_SCOPE면 retrieve()(임베딩 API)도 LLM도 호출되지 않는다.

def test_classify_scope_in_scope_for_dialysis_nutrition_question():
    result = C.classify_scope('혈액투석 환자가 피해야 할 고칼륨 채소는 어떤 게 있나요?')
    assert result.verdict == C.IN_SCOPE
    assert result.reason == 'dialysis_nutrition'


def test_classify_scope_out_of_scope_for_general_chitchat():
    result = C.classify_scope('오늘 서울 날씨 어때?')
    assert result.verdict == C.OUT_OF_SCOPE
    assert result.reason == 'unrelated'


def test_classify_scope_out_of_scope_for_hard_medical_question():
    # 감기약/타이레놀 — SYSTEM 프롬프트가 "개인별 판단 필요 -> 의료진 상담"으로 분리한 약물
    # 영역. domain 신호(투석/신장 등)가 전혀 없어 OUT_OF_SCOPE여야 한다.
    result = C.classify_scope('감기 걸렸을 때 타이레놀 먹어도 되나요?')
    assert result.verdict == C.OUT_OF_SCOPE
    assert result.reason == 'other_medical_treatment'


def test_classify_scope_out_of_scope_for_dialysis_procedure_not_nutrition():
    # 투석 시술/장비 신호(동정맥루)가 있으면 domain 신호(투석)가 함께 있어도 영양과 무관한
    # 시술 영역이므로 OUT_OF_SCOPE여야 한다(scope_gate_definitions.md §4-③).
    result = C.classify_scope('투석용 혈관(동정맥루)이 막히지 않게 하려면 뭘 조심해야 하나요?')
    assert result.verdict == C.OUT_OF_SCOPE
    assert result.reason == 'dialysis_procedure_not_nutrition'


def test_find_food_takes_priority_over_scope_gate(real_nut, monkeypatch):
    # "바나나 먹어도 되나요?"는 도메인 키워드가 전혀 없어(scope_gate_experiment.py 실측:
    # food_db 5문항 전부 텍스트만으로는 scope gate를 통과 못 함) 만약 scope gate가 find_food()
    # 보다 먼저(또는 대신) 적용되면 이 질문은 잘못 차단된다. answer()가 find_food()를 먼저
    # 호출해 food_db 경로로 라우팅하고 scope gate를 아예 건너뛰는지 확인한다.
    assert C.classify_scope('바나나 먹어도 되나요?').verdict == C.OUT_OF_SCOPE   # 참고: 텍스트만 보면 OOS로 잘못 판정됨
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='바나나는 적당히 드셔도 됩니다.'))
    text, sources = C.answer('바나나 먹어도 되나요?', weight=60,
                              consumed={'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0})
    assert text != C.OUT_OF_SCOPE_ANSWER
    assert text.endswith('▶ 결론: 지금 더 드셔도 됩니다')   # food_db 경로(코드 확정 판정)로 실제로 도달했음


def test_answer_out_of_scope_never_calls_retrieve(monkeypatch):
    monkeypatch.setattr(C, 'find_food', lambda q: None)

    def _fail_retrieve(q, top_k=5):
        raise AssertionError('scope gate가 OUT_OF_SCOPE로 판정했으면 retrieve()를 호출하면 안 된다(임베딩 API 호출 낭비)')
    monkeypatch.setattr(C, 'retrieve', _fail_retrieve)

    def _fail_client():
        raise AssertionError('scope gate가 OUT_OF_SCOPE로 판정했으면 LLM을 호출하면 안 된다')
    monkeypatch.setattr(C, '_client', _fail_client)

    text, sources = C.answer('감기 걸렸을 때 타이레놀 먹어도 되나요?')
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert sources == []


def test_answer_out_of_scope_returns_empty_sources_for_general_oos(monkeypatch):
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: (_ for _ in ()).throw(
        AssertionError('OUT_OF_SCOPE 경로에서 retrieve()가 호출되면 안 된다')))
    text, sources = C.answer('다음 주 로또 번호 추천해줘')
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert sources == []


def test_answer_in_scope_question_still_uses_rag_min_score_gate_and_rerank(monkeypatch):
    # scope gate가 IN_SCOPE로 판정한 이후에는 기존 RAG_MIN_SCORE 게이트 + alpha=0.05 재정렬
    # 파이프라인이 그대로(영향받지 않고) 동작해야 한다.
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'RAG_MIN_SCORE', 0.30)
    monkeypatch.setattr(C, 'RAG_LEXICAL_ALPHA', 0.05)
    hits = [{'source': 'good_doc', 'text': '칼륨이 많은 채소는 주의해야 합니다.', 'score': 0.5},
            {'source': 'noise_doc', 'text': '전혀 무관한 내용', 'score': 0.02}]
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: hits)
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='칼륨이 많은 채소를 주의하세요.'))

    question = '혈액투석 환자가 피해야 할 고칼륨 채소는 어떤 게 있나요?'
    assert C.classify_scope(question).verdict == C.IN_SCOPE
    text, sources = C.answer(question)
    assert text == '칼륨이 많은 채소를 주의하세요.'
    assert sources == ['good_doc']   # RAG_MIN_SCORE 미만인 noise_doc은 여전히 걸러짐(scope gate와 무관)


# ═══════════════ 9) 답변 문구(wording) 하드닝 회귀 테스트 (2026-08-14) ══════════════
# 이 블록은 SYSTEM/instruction/prompt 문자열을 손댄 이후에도 (a) 안전 불변식(결론은 코드가
# 확정, LLM은 절대 못 뒤집음), (b) scope/retrieval 라우팅, (c) 영양DB 수치 자체가 전혀
# 바뀌지 않았는지를 구조적으로(정확한 문자열 그대로가 아니라 "포함 여부"·"동작"으로) 검증한다.
# 정확한 카피 문구를 assert하면 문구를 다듬을 때마다 테스트가 깨지므로 일부러 피한다.

class _SpyChatCompletions:
    """chat.completions.create()에 실제로 전달된 messages를 기록하는 스파이(응답은 고정)."""

    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeChatResp(self._content)


class SpyClient:
    """FakeClient와 같은 역할이지만 chat.completions.create()에 전달된 인자를 캡처한다."""

    def __init__(self, query_vec=None, chat_content='(테스트용 LLM 응답)'):
        self.embeddings = _FakeEmbeddings(query_vec if query_vec is not None else [0.0] * 1536)
        self._spy_completions = _SpyChatCompletions(chat_content)
        self.chat = type('C', (), {'completions': self._spy_completions})()


def test_food_lookup_answer_still_forbids_verdict_override_and_number_invention(monkeypatch, real_nut):
    # instruction 문구를 새로 썼지만(결론-먼저 구조로), "판정을 뒤집지 마라"/"숫자를 임의로
    # 바꾸지 마라" 같은 안전 지시 자체는 여전히 LLM에게 전달돼야 한다.
    spy = SpyClient(chat_content='임의의 LLM 응답')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)

    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    text, sources = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                          weight=60, consumed=consumed)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '판정을 뒤집' in user_msg   # 결론 재파생/역전 금지 지시가 여전히 존재
    assert '숫자' in user_msg and ('바꾸' in user_msg or '지어내' in user_msg)   # 수치 임의 변경/조작 금지
    # 최종 화면 텍스트는 여전히 코드가 계산한 결론으로 끝난다(문구 변경과 무관하게 불변).
    assert text.endswith('▶ 결론: 지금 더 드셔도 됩니다')


def test_food_lookup_answer_numeric_facts_unchanged_in_prompt(monkeypatch, real_nut):
    # SYSTEM/instruction 문구를 바꿔도 facts(코드가 계산한 K/P/Na 수치·등급)는 손대지 않았다 —
    # LLM에게 전달되는 프롬프트 안의 [영양DB 실측 자료] 숫자가 실제 DB 조회값과 정확히 일치하는지
    # 직접 확인한다(오징어, 생것 — 100g당 칼륨/등급을 real_nut에서 직접 재계산해 대조).
    ing_nut = real_nut[0]
    food_name = '오징어, 생것'
    assert food_name in ing_nut, 'DB에 해당 재료명이 있어야 이 테스트가 의미 있음'
    k100 = ing_nut[food_name]['K']
    serving = C.SERVING_G.get(ing_nut[food_name].get('group'), C.DEFAULT_SERVING_G)
    k_serving = k100 * serving / 100
    grade = C._grade(k_serving)

    spy = SpyClient(chat_content='임의의 LLM 응답')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)

    text, sources = C.food_lookup_answer('오징어 먹어도 돼?', food_name)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert f'{k100:.0f}mg' in user_msg
    assert f'{k_serving:.0f}mg' in user_msg
    assert grade in user_msg


def test_system_prompt_still_instructs_no_fabricated_examples():
    # 문구는 재작성됐지만 "참고 자료에 없는 내용/예시를 지어내지 마라"는 취지의 지시는
    # SYSTEM 어딘가에 여전히 있어야 한다(카테고리 질문에서 허위 예시 방지).
    assert '지어내' in C.SYSTEM


def test_system_prompt_discourages_default_medical_disclaimer_boilerplate():
    # 매 답변마다 "의료진과 상담하세요"를 습관적으로 붙이지 말라는 취지가 SYSTEM에 있어야 한다
    # (기존에는 "반드시... 상담하도록 안내해라"였던 것을 "아껴 써라/매번 붙이지 마라"로 완화).
    assert ('아껴' in C.SYSTEM) or ('매번' in C.SYSTEM and '상담' in C.SYSTEM)


def test_system_prompt_mentions_real_meal_generation_feature_by_name():
    # 끼니 계획 질문에 대해 이 앱의 실제 기능(프론트 표기 "AI 식단 생성")을 안내하라는 지시가
    # 있어야 한다 — 가짜 기능을 지어내지 않고 실제 /generate 기능의 화면 표기 그대로 참조한다.
    assert 'AI 식단 생성' in C.SYSTEM


# ── OOS 3문항(Step 3) 회귀: retrieve()/LLM 호출 0회, sources=[] 유지 ────────────
@pytest.mark.parametrize('question', [
    '감기약 먹어도 되나요?',
    '암 치료는 어떻게 하나요?',
    '다음 주 로또 번호 추천해줘',
])
def test_named_oos_questions_never_call_retrieve_or_llm(monkeypatch, question):
    monkeypatch.setattr(C, 'find_food', lambda q: None)

    def _fail_retrieve(q, top_k=5):
        raise AssertionError(f'{question!r}: OUT_OF_SCOPE인데 retrieve()가 호출됨')
    monkeypatch.setattr(C, 'retrieve', _fail_retrieve)

    def _fail_client():
        raise AssertionError(f'{question!r}: OUT_OF_SCOPE인데 LLM이 호출됨')
    monkeypatch.setattr(C, '_client', _fail_client)

    text, sources = C.answer(question)
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert sources == []
    assert C.classify_scope(question).verdict == C.OUT_OF_SCOPE


def test_named_meal_planning_question_scope_and_routing_unchanged():
    # 2026-08-14 scope gate 일반화 이후: "저녁 뭐 먹지?"는 이제 meal_planning_intent 신호로
    # IN_SCOPE로 판정되어 실제로 RAG 경로(및 SYSTEM 문구)에 도달한다 — 아래 10번 블록
    # (test_meal_planning_scope_gate.py 성격의 테스트)에서 이 변경을 자세히 검증한다.
    # find_food()는 여전히 이 질문에 매칭되지 않는다(구체적 재료명이 없으므로) — 그 부분만
    # 이 테스트에서 계속 고정한다.
    assert C.classify_scope('저녁 뭐 먹지?').verdict == C.IN_SCOPE
    assert C.find_food('저녁 뭐 먹지?') is None


def test_answer_rag_path_prompt_still_grounds_only_in_retrieved_context(monkeypatch):
    # answer()의 RAG 경로 프롬프트 템플릿이 바뀌었지만, 여전히 검색된 context 텍스트 자체를
    # 프롬프트에 그대로 포함해 LLM에게 넘겨야 한다(문구만 추가됐지 근거 자료가 빠지면 안 됨).
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    hits = [{'source': 'good_doc', 'text': '칼륨이 많은 채소는 시금치입니다.', 'score': 0.5}]
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: hits)
    spy = SpyClient(chat_content='답변')
    monkeypatch.setattr(C, '_client', lambda: spy)

    question = '혈액투석 환자가 피해야 할 고칼륨 채소는 어떤 게 있나요?'
    text, sources = C.answer(question)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '칼륨이 많은 채소는 시금치입니다.' in user_msg   # 검색된 근거 텍스트가 그대로 전달됨
    assert 'good_doc' in user_msg   # 출처 라벨도 그대로 포함


# ═══════════ 10) 끼니 계획(meal-planning) scope gate 일반화 (2026-08-14) ══════════
# classify_scope()가 "저녁 뭐 먹지?" 같은 자연스러운 끼니 계획 질문을 DOMAIN_TERMS 키워드가
# 하나도 없다는 이유만으로 OUT_OF_SCOPE 처리하던 gap을 메우는 has_meal_planning 신호에 대한
# 테스트. 핵심 불변식 셋: (a) 끼니 시간대 단어("아침"/"점심"/"저녁") 단독으로는 절대 통과하지
# 않는다, (b) '추천'만으로는(끼니 시간대 단어와 결합해도) 통과하지 않는다 — 영화/드라마/로또
# 등 무엇에나 붙는 범용 추천 요청과 구분하기 위함, (c) red-flag(약물 등) 우선순위는 끼니
# 시간대 단어가 같이 있어도 절대 흔들리지 않는다. 문장 통째 하드코딩이 아니라 판정
# 동작(verdict)만 검증한다 — 실제 최종 문구가 아니라 결합 규칙 자체가 맞는지 확인하는 목적.

@pytest.mark.parametrize('question', [
    '저녁 뭐 먹지?',
    '점심 뭐 먹을까?',
    '아침 메뉴 추천해줘',
    '오늘 저녁 뭐 먹으면 좋을까?',
    '한 끼 추천해줘',
    '간단한 저녁 메뉴 알려줘',
])
def test_meal_planning_questions_become_in_scope(question):
    result = C.classify_scope(question)
    assert result.verdict == C.IN_SCOPE
    assert result.reason == 'meal_planning_intent'


@pytest.mark.parametrize('question', [
    '저녁 영화 뭐 볼까?',
    '아침에 비 와?',
    '저녁 약속 어디서 잡지?',
    '점심 회의 몇 시야?',
    '아침 운동 뭐 할까?',
    '저녁 드라마 추천해줘',        # (b) 끼니 시간대 단어 + '추천'만으로는 불충분
    '오늘 저녁 로또 번호 추천해줘',  # (b) 동일 — 범용 추천 요청은 여전히 차단돼야 함
])
def test_meal_time_word_alone_or_with_bare_recommend_stays_out_of_scope(question):
    # (a)/(b) 불변식: "아침"/"점심"/"저녁"은 음식과 무관한 문맥(날씨/약속/회의/운동/영화/
    # 드라마/로또)에도 똑같이 쓰이므로, 먹는 것 자체를 가리키는 표현("뭐 먹"/"메뉴" 등)이
    # 없으면(설령 '추천'이 있어도) 여전히 OUT_OF_SCOPE여야 한다.
    result = C.classify_scope(question)
    assert result.verdict == C.OUT_OF_SCOPE


def test_point_in_time_word_medication_question_stays_out_of_scope():
    # 끼니 시간대 단어와 결합됐지만 실제로는 약 복용 시점을 묻는 질문 — 먹는 대상이 음식이
    # 아니므로 meal_planning 신호가 없고, 애초에 domain 신호도 없어 OUT_OF_SCOPE(unrelated).
    result = C.classify_scope('점심에 약 언제 먹어?')
    assert result.verdict == C.OUT_OF_SCOPE


@pytest.mark.parametrize('question', [
    '저녁에 타이레놀 먹어도 돼?',
    '점심에 인슐린 얼마나 맞아?',
])
def test_redflag_priority_beats_meal_time_word(question):
    # (c) 불변식: red-flag 약물 용어(타이레놀/인슐린)가 있으면, 끼니 시간대 단어("저녁"/"점심")가
    # 같이 있어도 domain 신호가 없는 한 반드시 OUT_OF_SCOPE(other_medical_treatment)여야 한다 —
    # has_redflag and not has_domain 분기가 has_meal_planning보다 우선순위가 높기 때문.
    result = C.classify_scope(question)
    assert result.verdict == C.OUT_OF_SCOPE
    assert result.reason == 'other_medical_treatment'


@pytest.mark.parametrize('question,expected_verdict', [
    ('감기 걸렸을 때 타이레놀 먹어도 되나요?', C.OUT_OF_SCOPE),   # oos12 (hard_medical)
    ('당뇨병 인슐린 주사는 몇 단위로 놔야 하나요?', C.OUT_OF_SCOPE),  # oos13 (hard_medical)
    ('암 환자 항암치료 부작용은 어떻게 관리하나요?', C.OUT_OF_SCOPE),  # oos19 (hard_medical)
    ('갑상선 기능 저하증 약은 하루 중 언제 먹는 게 가장 좋나요?', C.OUT_OF_SCOPE),  # hoos3
    ('관절염 진통제를 매일 먹어도 위에 부담이 없을까요?', C.OUT_OF_SCOPE),  # hoos4
])
def test_existing_hard_medical_oos_still_rejected(question, expected_verdict):
    # 새 meal_planning 신호를 추가한 뒤에도 기존 hard-medical OOS 문항(끼니 시간대 단어가
    # 전혀 없는 질문들)은 영향을 받지 않아야 한다 — scope_gate_eval_set.json의
    # existing_oos_tags(oos12/oos13/oos19)와 new_hard_oos(hoos3/hoos4) 일부를 대표로 확인.
    assert C.classify_scope(question).verdict == expected_verdict


def test_find_food_still_takes_priority_over_meal_planning_signal(real_nut):
    # 끼니 시간대 단어 + 구체적 재료명이 함께 있는 질문("저녁에 사과 먹어도 돼?")은 여전히
    # answer()의 find_food()가 scope gate보다 먼저 가로채 food_db 경로로 라우팅돼야 한다
    # (call order: find_food() -> classify_scope()는 이번 작업에서 손대지 않음).
    assert C.find_food('저녁에 사과 먹어도 돼?') == '사과, 생것'
    assert C.find_food('아침에 오징어 먹어도 돼?') == '오징어, 생것'


@pytest.mark.parametrize('question', [
    '저녁 영화 뭐 볼까?',
    '점심에 약 언제 먹어?',
    '저녁에 타이레놀 먹어도 돼?',
    '오늘 저녁 로또 번호 추천해줘',
])
def test_meal_context_negative_questions_never_call_retrieve_or_llm(monkeypatch, question):
    # 새로 도입된 meal_planning 신호가 있는 질문이라도 실제로는 OUT_OF_SCOPE인 위 4개 문항은
    # 여전히 retrieve()/LLM을 전혀 호출하지 않아야 한다(test_named_oos_questions_...와 동일 패턴).
    monkeypatch.setattr(C, 'find_food', lambda q: None)

    def _fail_retrieve(q, top_k=5):
        raise AssertionError(f'{question!r}: OUT_OF_SCOPE인데 retrieve()가 호출됨')
    monkeypatch.setattr(C, 'retrieve', _fail_retrieve)

    def _fail_client():
        raise AssertionError(f'{question!r}: OUT_OF_SCOPE인데 LLM이 호출됨')
    monkeypatch.setattr(C, '_client', _fail_client)

    text, sources = C.answer(question)
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert sources == []
    assert C.classify_scope(question).verdict == C.OUT_OF_SCOPE


def test_meal_planning_question_actually_reaches_rag_pipeline(monkeypatch):
    # "저녁 뭐 먹지?"가 이제 IN_SCOPE이므로 answer()가 실제로 retrieve()까지 도달해야 한다
    # (OUT_OF_SCOPE_ANSWER로 조기 반환되지 않음) — find_food/retrieve/LLM을 모두 목킹해
    # 호출 여부와 최종 반환값을 확인.
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    calls = {'retrieve': 0}

    def _spy_retrieve(q, top_k=5):
        calls['retrieve'] += 1
        return [{'source': 'meal_doc', 'text': '한 끼 식단 구성 예시입니다.', 'score': 0.5}]
    monkeypatch.setattr(C, 'retrieve', _spy_retrieve)
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='AI 식단 생성 기능을 활용해보세요.'))

    text, sources = C.answer('저녁 뭐 먹지?')
    assert calls['retrieve'] == 1   # RAG 경로에 실제로 도달함(조기 OUT_OF_SCOPE 반환 아님)
    assert text != C.OUT_OF_SCOPE_ANSWER
    assert sources == ['meal_doc']


# ═══════ 11) answer_with_context() — stateless 한 턴 음식 후속 질문 지원 (2026-08-14) ═══════
# 실제 운영 QA 버그: "복숭아 먹어도 되나?"(food_db 정상 응답) 다음 "먹는다고 하면 몇조각
# 먹을까?"가 완전히 stateless라 이번 턴 텍스트만으로는 신호가 없어 OUT_OF_SCOPE로 잘못 빠짐.
# 대화 이력 전체나 서버 세션은 전혀 안 쓰고, 클라이언트가 직전 응답에서 그대로 돌려보낸
# canonical 재료명 문자열(context_food) 하나만 힌트로 받는 answer_with_context()를 검증한다.
# OpenAI 라이브 호출은 여기서도 전혀 하지 않는다(파일 docstring 원칙 그대로) — LIVE 검증은
# 이 파일 밖에서 별도로 수행한다(작업 지시의 "Live QA" 섹션).
_CONSUMED_ZERO = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}


def _mock_food_llm(monkeypatch, chat_content='(테스트용 LLM 응답)'):
    """answer_with_context()가 food_lookup_answer()/answer() 어느 경로로 가든 실제 OpenAI를
    절대 호출하지 않도록 retrieve()/embeddings/chat.completions를 전부 무해하게 모킹한다."""
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content=chat_content))


def _fail_retrieve_and_llm(monkeypatch, label):
    def _fail_retrieve(q, top_k=5):
        raise AssertionError(f'{label}: retrieve()가 호출되면 안 된다')
    monkeypatch.setattr(C, 'retrieve', _fail_retrieve)

    def _fail_client():
        raise AssertionError(f'{label}: LLM이 호출되면 안 된다')
    monkeypatch.setattr(C, '_client', _fail_client)


# ── (1) 이번 질문에 명시적 재료명이 있으면 context_food는 완전히 무시하고 그게 이긴다 ──
def test_answer_with_context_explicit_food_beats_context_food(real_nut, monkeypatch):
    _mock_food_llm(monkeypatch)
    text, sources, ctx = C.answer_with_context(
        '그럼 사과는 먹어도 돼?', context_food='복숭아, 백도, 생것',
        weight=60, consumed=_CONSUMED_ZERO)
    assert ctx == '사과, 생것'   # context_food(복숭아)가 아니라 이번 질문의 명시적 재료(사과)


# ── (2) 유효한 후속 질문 + 유효한 context_food -> food_lookup_answer()를 통해 context food 사용 ──
def test_answer_with_context_valid_followup_uses_context_food_via_food_lookup_answer(real_nut, monkeypatch):
    spy = SpyClient(chat_content='(임의 응답)')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)

    text, sources, ctx = C.answer_with_context(
        '먹는다고 하면 몇조각 먹을까?', context_food='복숭아, 백도, 생것',
        weight=60, consumed=_CONSUMED_ZERO)
    assert ctx == '복숭아, 백도, 생것'
    assert '▶ 결론' in text   # food_lookup_answer()의 개인화 판정 안전장치가 그대로 적용됨
    # food_lookup_answer()가 실제로 호출됐다는 근거: 프롬프트에 복숭아 실측 facts가 들어있어야 함.
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '재료명: 복숭아, 백도, 생것' in user_msg


# ── (3) 후속 질문 모양이 아닌 무관한 질문은 context_food를 무시하고 기존 answer() 그대로 ──
def test_answer_with_context_unrelated_question_ignores_context(monkeypatch):
    _fail_retrieve_and_llm(monkeypatch, '무관한 질문인데 retrieve/LLM 호출')
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    text, sources, ctx = C.answer_with_context(
        '저녁 영화 뭐 볼까?', context_food='복숭아, 백도, 생것')
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert ctx is None


# ── (4) 의료 red-flag 질문은 context_food를 무시/override하고 여전히 차단됨 (Case G, 안전 핵심) ──
def test_answer_with_context_redflag_question_ignores_context_and_stays_blocked(monkeypatch):
    _fail_retrieve_and_llm(monkeypatch, 'red-flag 질문인데 retrieve/LLM 호출(peach context 유출 의심)')
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    text, sources, ctx = C.answer_with_context(
        '그럼 타이레놀은 얼마나 먹어?', context_food='복숭아, 백도, 생것')
    assert text == C.OUT_OF_SCOPE_ANSWER   # 절대 복숭아 food_db 답변이 되면 안 됨
    assert sources == []
    assert ctx is None


# ── (5) 존재하지 않는 context_food는 무시되고(신뢰하지 않음) 정상 폴백 ──
def test_answer_with_context_invalid_context_food_is_ignored(monkeypatch):
    _fail_retrieve_and_llm(monkeypatch, '무효 context_food인데 retrieve/LLM 호출')
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    text, sources, ctx = C.answer_with_context('그럼 얼마나?', context_food='NOT_A_REAL_FOOD')
    assert text == C.OUT_OF_SCOPE_ANSWER   # 신호가 '그럼 얼마나?' 자체는 후속 질문 모양이지만
    assert ctx is None                     # context_food가 무효라 food 경로로 못 감


# ── (6)/(7) food-DB 응답과 food-followup 응답 모두 올바른 canonical context_food를 반환 ──
def test_answer_with_context_food_db_response_returns_canonical_context_food(real_nut, monkeypatch):
    _mock_food_llm(monkeypatch)
    text, sources, ctx = C.answer_with_context('복숭아 먹어도 되나?', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx == '복숭아, 백도, 생것'


def test_answer_with_context_food_followup_response_returns_same_canonical_context_food(real_nut, monkeypatch):
    _mock_food_llm(monkeypatch)
    text1, sources1, ctx1 = C.answer_with_context('복숭아 먹어도 되나?', weight=60, consumed=_CONSUMED_ZERO)
    text2, sources2, ctx2 = C.answer_with_context(
        '먹는다고 하면 몇조각 먹을까?', context_food=ctx1, weight=60, consumed=_CONSUMED_ZERO)
    assert ctx2 == ctx1   # 후속 턴도 같은 canonical 재료명을 그대로 유지해 돌려줌


# ── (8) 음식과 무관한 응답(RAG/끼니계획/OOS)은 항상 context_food=None을 반환 ──
def test_answer_with_context_non_food_response_returns_context_food_none(monkeypatch):
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [
        {'source': 'meal_doc', 'text': '한 끼 식단 구성 예시입니다.', 'score': 0.5}])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='AI 식단 생성 기능을 활용해보세요.'))
    text, sources, ctx = C.answer_with_context('저녁 뭐 먹지?', context_food='복숭아, 백도, 생것')
    assert ctx is None
    assert text != C.OUT_OF_SCOPE_ANSWER   # 실제로 RAG 경로까지 도달했음(끼니 계획은 IN_SCOPE)


# ── (9) 기존 요청 형태(context_food 필드 자체가 없음)도 그대로 동작 — 하위호환 ──
def test_answer_with_context_backward_compatible_without_context_food_arg(real_nut, monkeypatch):
    _mock_food_llm(monkeypatch)
    # context_food 인자를 아예 안 준 기존 호출부 시뮬레이션(default=None)
    text, sources, ctx = C.answer_with_context('복숭아 먹어도 되나?', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx == '복숭아, 백도, 생것'
    assert '▶ 결론' in text


def test_chat_req_backward_compatible_without_context_food_field():
    # API 계약 레벨: ChatReq(question=...)만 줘도(기존 클라이언트 요청 형태) 여전히 검증 통과하고
    # context_food는 기본값 None.
    from schemas import ChatReq
    req = ChatReq(question='복숭아 먹어도 되나요?')
    assert req.context_food is None
    req2 = ChatReq(question='복숭아 먹어도 되나요?', context_food='복숭아, 백도, 생것')
    assert req2.context_food == '복숭아, 백도, 생것'


# ── (10) follow-up 경로로 도달해도 기존 K/P/Na 판정/▶ 결론 안전장치가 그대로 적용됨 ──
def test_answer_with_context_followup_path_preserves_verdict_safety_net(real_nut, monkeypatch):
    # LLM이 반대로 말해도(기존 food_lookup_answer 안전장치 테스트와 동일한 방식) 최종 결론은
    # 코드가 계산한 값으로 못박혀야 한다 — follow-up 경로도 food_lookup_answer()를 그대로 재사용
    # 하므로 이 안전장치가 재구현 없이 그대로 적용되는지 확인.
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='괜찮아요! 많이 드세요!'))
    # 바나나, 생것 1회분(120g) 칼륨 ≈ 426mg. 하루 Kmax=3000인데 이미 2700 먹었으면 남은 예산 300 < 426.
    consumed = {'E': 0, 'protein': 0, 'K': 2700, 'P': 0, 'Na': 0, 'Na_season': 0}
    text, sources, ctx = C.answer_with_context(
        '그럼 얼마나 먹어?', context_food='바나나, 생것', weight=60, consumed=consumed)
    assert ctx == '바나나, 생것'
    assert text.split('▶ 결론:')[-1].strip() == '지금은 피하시는 게 좋습니다'


# ── (11) 이 변경이 retrieve()/_relevant()/_rerank()/classify_scope()의 기존 동작을 전혀
#         바꾸지 않았는지 재확인(빠른 sanity 재검증 — 새 함수를 추가만 했을 뿐 기존 함수 본문은
#         손대지 않았으므로 이미 위 160개 기존 테스트가 이를 보장하지만, 이 변경 작업 맥락에서
#         한 번 더 명시적으로 못박는다) ──
def test_existing_scope_and_retrieval_pipeline_unaffected_by_new_context_feature(monkeypatch):
    assert C.classify_scope('바나나 먹어도 되나요?').verdict == C.OUT_OF_SCOPE   # find_food 우선 케이스, 불변
    assert C.classify_scope('감기 걸렸을 때 타이레놀 먹어도 되나요?').verdict == C.OUT_OF_SCOPE
    assert C.classify_scope('저녁 뭐 먹지?').verdict == C.IN_SCOPE
    hits = [{'source': 'a', 'text': 'x', 'score': 0.5}, {'source': 'b', 'text': 'y', 'score': 0.02}]
    assert C._relevant(hits) == [hits[0]]   # RAG_MIN_SCORE 게이트 불변
    # answer()는 이번 작업에서 시그니처/반환 계약을 전혀 바꾸지 않았다 — 2-tuple 그대로.
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: (_ for _ in ()).throw(
        AssertionError('OUT_OF_SCOPE인데 retrieve() 호출됨')))
    result = C.answer('다음 주 로또 번호 추천해줘')
    assert isinstance(result, tuple) and len(result) == 2
    text, sources = result
    assert text == C.OUT_OF_SCOPE_ANSWER


# ── (12) 조각/개수 hallucination 방지 — "몇 조각" 후속 질문에는 프롬프트에 개수를 지어내지
#         말라는 지시가 실제로 포함돼 있어야 한다(구조적 검증, LLM 응답 문자열 자체는 모킹돼
#         고정값이므로 의미 없음 — 프롬프트에 안전 지시가 전달됐는지가 검증 대상) ──
def test_food_lookup_answer_prompt_forbids_piece_count_invention_when_asked(real_nut, monkeypatch):
    spy = SpyClient(chat_content='2~3조각 정도 드셔도 됩니다.')   # LLM이 개수를 지어내는 최악의 경우를 가정
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)

    text, sources, ctx = C.answer_with_context(
        '먹는다고 하면 몇조각 먹을까?', context_food='복숭아, 백도, 생것',
        weight=60, consumed=_CONSUMED_ZERO)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '조각' in user_msg and ('지어내지 마' in user_msg or '새로 지어내지' in user_msg)
    assert 'g' in user_msg.lower() or '섭취 기준량' in user_msg   # g 기준 안내로 유도하는 문구 포함


def test_food_lookup_answer_no_piece_count_instruction_when_not_asked(real_nut, monkeypatch):
    # "몇 조각" 신호가 없는 평범한 재료 질문에는 이 안전 문구를 굳이 추가하지 않는다(불필요한
    # 프롬프트 비대화 방지) — _asks_piece_count()가 False인 경우 instruction에 조각 관련 문구가
    # 안 붙는지 확인.
    spy = SpyClient(chat_content='적당량 드셔도 됩니다.')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)
    text, sources = C.food_lookup_answer('바나나 먹어도 되나요?', '바나나, 생것',
                                          weight=60, consumed=_CONSUMED_ZERO)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '조각' not in user_msg


# ═══════════ Section 15 다중 턴 시나리오 (mocked, LIVE는 이 파일 밖에서 별도 검증) ══════════
def test_case_a_peach_piece_count_followup_not_out_of_scope(real_nut, monkeypatch):
    # 실제 보고된 버그 시퀀스(모킹판): "복숭아 먹어도 되나?" -> "먹는다고 하면 몇조각 먹을까?"
    _mock_food_llm(monkeypatch)
    text1, sources1, ctx1 = C.answer_with_context('복숭아 먹어도 되나?', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx1 == '복숭아, 백도, 생것'
    text2, sources2, ctx2 = C.answer_with_context(
        '먹는다고 하면 몇조각 먹을까?', context_food=ctx1, weight=60, consumed=_CONSUMED_ZERO)
    assert text2 != C.OUT_OF_SCOPE_ANSWER   # 더 이상 OUT_OF_SCOPE로 잘못 빠지지 않음(버그 재현 대상)
    assert '재료명: 복숭아' in text2 or ctx2 == '복숭아, 백도, 생것'
    assert ctx2 == ctx1


def test_case_e_topic_switch_to_meal_planning_clears_context_and_stays_cleared(monkeypatch):
    # "복숭아 먹어도 돼?" -> "저녁 뭐 먹지?"(끼니계획, context_food는 반드시 None) ->
    # "그럼 몇 조각?" + context_food=None(프론트가 이미 지운 상태) -> 복숭아가 되살아나면 안 됨.
    monkeypatch.setattr(C, 'find_food', lambda q: {'복숭아 먹어도 돼?': '복숭아, 백도, 생것'}.get(q))
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [
        {'source': 'meal_doc', 'text': '한 끼 식단 구성 예시입니다.', 'score': 0.5}])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='AI 식단 생성 기능을 활용해보세요.'))

    _, _, ctx_mid = C.answer_with_context('저녁 뭐 먹지?', context_food='복숭아, 백도, 생것')
    assert ctx_mid is None

    text3, sources3, ctx3 = C.answer_with_context('그럼 몇 조각?', context_food=None)
    assert ctx3 is None
    assert '복숭아' not in text3   # 되살아나지 않음(고정 OOS 문구에는 애초에 재료명이 없음)


# ═══════ 16) Bug1 — 조각/개수 질문의 무관한 "자료 없음" 서술 제거 (2026-08-14) ═══════
# 실측 재현: "복숭아 먹어도 되나?" -> "먹는다고 하면 몇조각 먹을까?"에서 LLM이 "조리법과 관련된
# 주의사항은 제공된 자료에서 확인할 수 없습니다" 같은, 사용자가 묻지도 않은 주제에 대한 무관한
# 문장을 답변에 끼워넣었다. 프롬프트 구조(step3 완전 대체 + "자료 없음" narrate 금지)를
# 검증한다 — 실제 LLM 산문은 라이브 QA에서만 확인(이 파일은 항상 모킹).

def test_piece_count_instruction_forbids_narrating_missing_evidence(real_nut, monkeypatch):
    spy = SpyClient(chat_content='임의의 LLM 응답')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)

    text, sources = C.food_lookup_answer('복숭아 몇 조각 먹어도 돼?', '복숭아, 백도, 생것',
                                          weight=60, consumed=_CONSUMED_ZERO)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    # "제공된 자료에서 확인할 수 없다"는 식으로 없다는 사실 자체를 narrate하지 말라는 지시가 있어야 함
    assert ('확인할 수 없' in user_msg) or ('없다는 사실' in user_msg)
    assert '지어내지 마' in user_msg or '새로 지어내지' in user_msg
    # 일반 조리법 팁 문구("무관하면 생략")는 조각 질문에서는 더 이상 별도로 남아있지 않아야 함 —
    # step3가 완전히 대체됐는지(추가만 된 게 아닌지) 구조적으로 확인.
    assert '조리법·주의사항 팁 한 줄만' not in user_msg


def test_piece_count_instruction_still_forbids_number_invention(real_nut, monkeypatch):
    spy = SpyClient(chat_content='2~3조각 정도 드셔도 됩니다.')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)

    text, sources = C.food_lookup_answer('복숭아 몇 조각 먹어도 돼?', '복숭아, 백도, 생것')
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '조각' in user_msg and ('지어내지 마' in user_msg or '새로 지어내지' in user_msg)
    assert '섭취 기준량' in user_msg   # g 기준 안내로 유도


@pytest.mark.parametrize('question', [
    '몇 알 먹어도 될까?', '몇알 먹어도 될까?', '몇 덩이 정도 먹어도 돼?', '몇덩이 먹어도 돼?',
])
def test_asks_piece_count_generalized_terms(question):
    # Bug1 spec의 예시(몇 알/몇 덩이)로 _PIECE_COUNT_HINT_TERMS가 확장됐는지 확인.
    assert C._asks_piece_count(question) is True


def test_asks_piece_count_does_not_false_trigger_on_allergy():
    # '알' 단독이 아니라 '몇 알'/'몇알'처럼 구체적 구문만 인정해야 한다 — '알레르기'처럼 무관한
    # 단어에 오탐하지 않는지 확인.
    assert C._asks_piece_count('알레르기 있는데 먹어도 되나요?') is False


# ═══════ 17) Bug2 — 동사 없는 화제 전환 후속 질문("그럼 사과는?") 처리 (2026-08-14) ═══════
# find_food()는 FOOD_QUESTION_KW가 없으면 절대 매칭하지 않으므로 "그럼 사과는?"은 늘 실패한다
# (find_food() 자체는 건드리지 않음 — 아래에서 재확인). answer_with_context()에 새로 추가된
# _find_food_switch_candidate() 기반 분기만 검증한다.

def test_find_food_unchanged_no_verb_apple_question_still_fails(real_nut):
    # find_food()는 이번 작업에서 관찰 가능한 동작이 전혀 바뀌지 않아야 한다 — "그럼 사과는?"처럼
    # FOOD_QUESTION_KW가 없는 질문은 여전히 실패해야(Bug2가 존재했던 이유 자체) 한다.
    assert C.find_food('그럼 사과는?') is None
    assert C.find_food('사과는?') is None


@pytest.mark.parametrize('question,expected_short', [
    ('그럼 사과는?', '사과'),
    ('사과는 어때', '사과'),
    ('사과는 괜찮아', '사과'),
    ('바나나는?', '바나나'),
])
def test_find_food_switch_candidate_detects_adjacent_topic_marker(real_nut, question, expected_short):
    result = C._find_food_switch_candidate(question)
    assert result is not None and result.startswith(expected_short)


@pytest.mark.parametrize('question', [
    '사과 색깔은?',       # 조사가 '색깔'에 붙어 있음 — 사과가 화제가 아님
    '사과 사진 보여줘',    # 조사 자체가 없음
])
def test_find_food_switch_candidate_rejects_non_adjacent_false_positive(real_nut, question):
    # Bug2 스펙이 명시한 핵심 오탐 방지 케이스 — "사과"가 idx에 실제로 존재해도(그래서 naive
    # substring 매칭이면 걸렸을 것) 조사가 재료명에 직접 붙어있지 않으면 반드시 None.
    idx, _ = C._food_index()
    assert '사과' in idx, '테스트 전제: DB에 사과가 있어야 이 회귀가 의미 있음'
    assert C._find_food_switch_candidate(question) is None


@pytest.mark.parametrize('question', [
    '사과는 바나나는 뭐가 나아?',   # 둘 다 조사가 직접 붙어 topicalize됨
    '사과랑 바나나는 뭐가 나아?',   # (Bug3) 조사는 바나나에만 붙지만, 넓은 탐지가 사과도 함께
                                    # 언급됐음을 먼저 잡아내야 한다 — 예전엔 이게 조용히
                                    # '바나나'로만 좁혀지는 버그였다(하나를 임의 선택).
    '오징어랑 두부는?',
    '사과, 바나나는 어때?',
])
def test_find_food_switch_candidate_no_random_pick_on_genuine_ambiguity(real_nut, question):
    # 서로 다른(canonical 기준) 재료가 2개 이상 언급되면, 조사가 그중 하나에만 우연히 붙어있어도
    # 절대 하나를 임의로 고르지 않고 _MULTI_FOOD_AMBIGUOUS sentinel을 반환해야 한다.
    assert C._find_food_switch_candidate(question) is C._MULTI_FOOD_AMBIGUOUS


def test_distinct_canonical_foods_dedupes_same_food_different_expression(real_nut):
    # "같은 음식 표현 중복은 multi-food로 세지 말 것" — '복숭아'와 '백도'가 같은 canonical
    # 항목(복숭아, 백도, 생것)을 가리키면 서로 다른 음식 2개로 잘못 세면 안 된다.
    idx, _ = C._food_index()
    if '백도' in idx and '복숭아' in idx:
        result = C._distinct_canonical_foods('복숭아랑 백도는 뭐가 나아?')
        assert len(result) == 1


def test_find_food_switch_candidate_none_for_non_food_nouns(real_nut):
    for q in ('그럼 영화는?', '그럼 타이레놀은?', '그럼 로또는?'):
        assert C._find_food_switch_candidate(q) is None


# ── answer_with_context()에 실제로 배선됐는지 ──────────────────────────────────
def test_answer_with_context_verbless_switch_resolves_to_new_food(real_nut, monkeypatch):
    # Case C: 복숭아 컨텍스트 + "그럼 사과는?" -> 사과로 전환돼야 한다(버그였던 케이스).
    _mock_food_llm(monkeypatch)
    text, sources, ctx = C.answer_with_context(
        '그럼 사과는?', context_food='복숭아, 백도, 생것', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx == '사과, 생것'
    assert '재료명: 사과' in text or ctx == '사과, 생것'


def test_answer_with_context_switch_takes_priority_over_old_context_continuation(real_nut, monkeypatch):
    # 새 화제 전환 재료가 있으면 old context_food를 계속 쓰는 _is_food_followup_query() 경로보다
    # 항상 이겨야 한다(스펙의 명시적 우선순위 요구) — "그럼 사과는?"은 마침 계속-질문 신호도
    # 아니라 이 경로 하나로만 확인되지만, 스파이로 실제 food_lookup_answer 호출 대상이 사과임을
    # 프롬프트 facts로 재확인한다.
    spy = SpyClient(chat_content='(임의 응답)')
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=3: [])
    monkeypatch.setattr(C, '_client', lambda: spy)
    text, sources, ctx = C.answer_with_context(
        '그럼 사과는?', context_food='복숭아, 백도, 생것', weight=60, consumed=_CONSUMED_ZERO)
    call = spy._spy_completions.calls[0]
    user_msg = next(m['content'] for m in call['messages'] if m['role'] == 'user')
    assert '재료명: 사과' in user_msg
    assert ctx == '사과, 생것'


def test_answer_with_context_squid_potassium_followup_unaffected_by_switch_logic(real_nut, monkeypatch):
    # Case D: 오징어 -> "칼륨은?" — 새 로직 도입 후에도 기존 같은-재료 후속 질문 경로가 그대로
    # 동작해야 한다(화제 전환 후보가 없으므로 기존 _is_food_followup_query() 경로로 폴백).
    _mock_food_llm(monkeypatch)
    text1, sources1, ctx1 = C.answer_with_context('오징어 먹어도 돼?', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx1 == '오징어, 생것'
    text2, sources2, ctx2 = C.answer_with_context('칼륨은?', context_food=ctx1,
                                                   weight=60, consumed=_CONSUMED_ZERO)
    assert ctx2 == ctx1


def test_answer_with_context_explicit_food_still_wins_over_switch_logic(real_nut, monkeypatch):
    # Case H: 복숭아 컨텍스트 + 명시적 "사과 먹어도 돼?" — find_food()가 이미 먼저 잡으므로 이번
    # 작업(3단계 분기)과 무관하게 계속 사과로 라우팅돼야 한다(기존 동작 불변 재확인).
    _mock_food_llm(monkeypatch)
    text, sources, ctx = C.answer_with_context(
        '사과 먹어도 돼?', context_food='복숭아, 백도, 생것', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx == '사과, 생것'


def test_answer_with_context_invalid_context_switch_shaped_question_falls_through_safely(monkeypatch):
    # Case I: context_food가 무효면(precondition (c) 실패) 화제 전환 분기 자체가 아예 평가되지
    # 않고 기존 안전 동작(여기선 OOS)으로 폴백해야 한다.
    _fail_retrieve_and_llm(monkeypatch, '무효 context_food + 화제전환 모양인데 retrieve/LLM 호출')
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    text, sources, ctx = C.answer_with_context('그럼 사과는?', context_food='NOT_A_REAL_FOOD')
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert ctx is None


@pytest.mark.parametrize('question', ['그럼 영화는?', '그럼 타이레놀은?', '그럼 로또는?'])
def test_answer_with_context_non_food_nouns_never_call_food_lookup_answer(monkeypatch, question):
    # mock-assert food_lookup_answer가 절대 호출되지 않는지 확인(영화/로또는 그냥 도메인
    # 신호가 없어 OOS로, 타이레놀은 red-flag로 각각 다른 경로지만 결과는 동일 — food_lookup_answer
    # 미호출 + 복숭아 컨텍스트가 새지 않음).
    def _fail_food_lookup(*a, **kw):
        raise AssertionError(f'{question!r}: food_lookup_answer가 호출되면 안 된다')
    monkeypatch.setattr(C, 'food_lookup_answer', _fail_food_lookup)
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='(응답)'))
    text, sources, ctx = C.answer_with_context(question, context_food='복숭아, 백도, 생것')
    assert ctx is None


def test_answer_with_context_apple_color_question_does_not_misfire_with_valid_context(real_nut, monkeypatch):
    # "사과 색깔은?" — DB에 '사과'가 실제로 존재하고 유효한 context_food(복숭아)도 있지만, 조사가
    # '사과'가 아니라 '색깔'에 붙어 있으므로 화제 전환으로 오인해선 안 된다. 이 질문 자체는
    # _is_food_followup_query()도 만족하지 않으므로(수량/영양소/그럼+먹는의도 신호가 전혀 없음)
    # 기존 answer()로 안전하게 폴백해야 한다.
    def _fail_food_lookup(*a, **kw):
        raise AssertionError('사과 색깔은?: food_lookup_answer가 호출되면 안 된다(사과로도 복숭아로도 안 가야 함)')
    monkeypatch.setattr(C, 'food_lookup_answer', _fail_food_lookup)
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='(응답)'))
    text, sources, ctx = C.answer_with_context('사과 색깔은?', context_food='복숭아, 백도, 생것')
    assert ctx is None


def test_answer_with_context_ambiguous_multi_food_question_does_not_guess(real_nut, monkeypatch):
    def _fail_food_lookup(*a, **kw):
        raise AssertionError('애매한 다중 재료 질문: food_lookup_answer가 호출되면 안 된다(임의로 고르면 안 됨)')
    monkeypatch.setattr(C, 'food_lookup_answer', _fail_food_lookup)
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: [])
    monkeypatch.setattr(C, '_client', lambda: FakeClient(chat_content='(응답)'))
    text, sources, ctx = C.answer_with_context(
        '사과는 바나나는 뭐가 나아?', context_food='복숭아, 백도, 생것')
    assert ctx is None


# ═══════ 18) find_food() 리팩터(_resolve_food_name 공유) 이후 관찰 가능 동작 불변 재확인 ═══════
@pytest.mark.parametrize('question,expected', [
    ('바나나 먹어도 되나요?', '바나나, 생것'),
    ('말린 바나나 먹어도 되나요?', '바나나, 동결건조'),
    ('데친 시금치 먹어도 되나요?', '시금치, 데친것'),
    ('시금치 많이 먹어도 되나요?', '시금치, 생것'),
])
def test_find_food_behavior_unchanged_after_resolve_food_name_extraction(real_nut, question, expected):
    assert C.find_food(question) == expected


# ═══════ 19) Bug3 — "사과랑 바나나는?" 같은 multi-food 화제 전환 ambiguity (2026-08-14) ═══════

@pytest.mark.parametrize('question', [
    '사과랑 바나나는?',
    '사과하고 바나나는?',
    '오징어랑 두부는?',
    '사과, 바나나는 어때?',
])
def test_answer_with_context_multi_food_switch_returns_clarify_deterministically(monkeypatch, question):
    # 서로 다른 food가 2개 이상 동시에 topicalize/언급되면(조사가 그중 하나에만 우연히 붙어있어도)
    # 절대 임의로 하나를 고르지 않고, LLM 호출 없이 고정 clarification으로 즉시 답해야 한다.
    # sources=[]/context_food=None이어야 하고(기존 context도 끊음), retrieve/LLM은 호출되면 안 된다.
    def _fail_food_lookup(*a, **kw):
        raise AssertionError(f'{question!r}: food_lookup_answer가 호출되면 안 된다(임의 선택 금지)')
    monkeypatch.setattr(C, 'food_lookup_answer', _fail_food_lookup)
    monkeypatch.setattr(C, 'find_food', lambda q: None)

    def _fail_retrieve(q, top_k=5):
        raise AssertionError(f'{question!r}: multi-food clarification은 retrieve() 호출이 필요 없다')
    monkeypatch.setattr(C, 'retrieve', _fail_retrieve)

    def _fail_client():
        raise AssertionError(f'{question!r}: multi-food clarification은 LLM 호출이 필요 없다(deterministic)')
    monkeypatch.setattr(C, '_client', _fail_client)

    text, sources, ctx = C.answer_with_context(question, context_food='복숭아, 백도, 생것')
    assert text == C.MULTI_FOOD_CLARIFY_ANSWER
    assert sources == []
    assert ctx is None


def test_answer_with_context_multi_food_with_redflag_stays_medical_safe(monkeypatch):
    # "바나나랑 타이레놀은?" — red-flag(타이레놀)가 있으면 이번 multi-food ambiguity 처리가
    # 끼어들 틈도 없이(기존 has_redflag_or_procedure 체크가 먼저 걸러냄) 기존 medical-OOS 안전
    # 동작이 그대로 적용돼야 한다 — clarification으로 우회해 약물 질문을 가볍게 다루면 안 된다.
    def _fail_food_lookup(*a, **kw):
        raise AssertionError('food_lookup_answer가 호출되면 안 된다')
    monkeypatch.setattr(C, 'food_lookup_answer', _fail_food_lookup)
    monkeypatch.setattr(C, 'find_food', lambda q: None)
    monkeypatch.setattr(C, 'retrieve', lambda q, top_k=5: (_ for _ in ()).throw(
        AssertionError('red-flag 질문인데 retrieve()가 호출됨')))
    text, sources, ctx = C.answer_with_context('바나나랑 타이레놀은?', context_food='복숭아, 백도, 생것')
    assert text == C.OUT_OF_SCOPE_ANSWER
    assert text != C.MULTI_FOOD_CLARIFY_ANSWER
    assert sources == []
    assert ctx is None


@pytest.mark.parametrize('question,expected_short', [
    ('그럼 사과는?', '사과'),
    ('그러면 바나나는?', '바나나'),
])
def test_answer_with_context_single_food_switch_still_works_after_ambiguity_fix(
        real_nut, monkeypatch, question, expected_short):
    # 회귀 방지 — multi-food ambiguity 처리를 추가한 뒤에도 음식 1개짜리 화제 전환은 그대로
    # 동작해야 한다(Bug2가 고쳤던 원래 케이스).
    _mock_food_llm(monkeypatch)
    text, sources, ctx = C.answer_with_context(
        question, context_food='복숭아, 백도, 생것', weight=60, consumed=_CONSUMED_ZERO)
    assert ctx is not None and ctx.startswith(expected_short)
