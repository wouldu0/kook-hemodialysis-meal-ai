# -*- coding: utf-8 -*-
"""
recipe_editor_FOOK.py — 조리과정을 LLM으로 "편집"(생성 아님).
실제 원본 조리과정(data/FOOK_recipe_steps.json)을 기반으로:
  · 레버가 바꾼 재료·양 반영
  · 화학/인공 조미료 문구 삭제
  · 투석 조리법 추가 (칼륨 빼는 데치기·물담그기, 저염)
환각 방지 = 맨땅 생성 X, 실제 스텝을 고치기만.

API 키: 환경변수 OPENAI_API_KEY (채팅에 붙이지 말고 직접 set).
  Windows:  set OPENAI_API_KEY=sk-...
"""
import os, json

_STEPS = None
def _steps():
    global _STEPS
    if _STEPS is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'FOOK_recipe_steps.json')
        _STEPS = json.load(open(path, encoding='utf-8-sig'))
    return _STEPS

CHEM_WORDS = ('화학조미료', '핵산조미료', '복합조미료', '조미료', '미원', '다시다')

# 채소 조리 시 칼륨 빼는 방법 — 레시피 옆에 '접히는(아코디언)' 형태로 프론트가 렌더.
# 채소 종류(단단한/부드러운)에 따라 방법이 달라 두 묶음으로. 서버 /veg_potassium_tips로 내려줌.
VEG_POTASSIUM_TIPS = [
    {
        'category': '감자·고구마·당근·무 같은 단단한 채소',
        'steps': [
            {'title': '얇게 썰기',
             'detail': '껍질을 깎은 다음, 물이 잘 스며들도록 0.3cm 정도로 얇게 썰어주세요.'},
            {'title': '물에 담가두기',
             'detail': '썰어둔 채소를 따뜻한 물에 최소 2시간 이상(하룻밤 정도면 더 좋아요!) 푹 담가두세요.'},
            {'title': '살짝 끓이기',
             'detail': '담가두었던 물은 버리고, 새 물을 채소 양의 5배쯤 받아서 5분 동안 보글보글 끓인 뒤 '
                       '헹궈주세요. (이때 끓여낸 물은 절대로 드시면 안 돼요!)'},
        ],
    },
    {
        'category': '시금치·호박·버섯 같은 부드러운 채소',
        'steps': [
            {'title': '잎 위주로 쓰기',
             'detail': '줄기나 껍질 쪽에 칼륨이 많으니, 껍질은 벗기고 줄기보다는 부드러운 잎 위주로 골라주세요.'},
            {'title': '물에 헹구기',
             'detail': '찬물에 2시간 이상 담가두었다가 깨끗한 물로 여러 번 헹궈냅니다.'},
            {'title': '데치고 꽉 짜기',
             'detail': '끓는 물에 데쳐낸 다음, 채소 속에 남은 물기를 꼭 짜서 버려주세요. '
                       "요리할 때는 꼭 '새 물'을 받아서 써야 합니다."},
        ],
    },
]

SYSTEM = (
    "너는 혈액투석 환자용 식단의 조리법을 다듬는 영양·조리 도우미다. "
    "반드시 '주어진 원본 조리과정을 기반으로 편집'만 하고, 없는 과정을 지어내지 마라. "
    "출력은 한국어, ① ② ③ 번호 형식, 간결하게."
)

BROTH_HINT_KW = ('육수', '국물', '멸치', '다시', '장국', '사골', '곰탕')
SOUP_MENU_KW = ('국', '찌개', '탕', '전골', '장국')


SOUP_CLASSES = {'국', '찌개', '탕', '전골', '장국'}


def _has_broth(menu, ingredients):
    # 실측(2026-07-24): 이름 문자열 매칭만으론 틀림 — 육개장은 실제 F.MENU_CLASS='국'인데 이름에
    # '국/찌개/탕' 글자가 없어 못 잡히고(육수 규칙 자체가 통째로 안 걸림), 닭볶음탕은 반대로 이름에
    # '탕'이 있어 걸리지만 실제 F.MENU_CLASS='볶음'(조림류, 물 많이 안 씀)이라 국물 규칙이 잘못
    # 적용됨. 진짜 분류 데이터(MENU_CLASS)를 우선 쓰고, 없을 때만 이름/재료 매칭으로 폴백.
    import FOOK_adjust_levers as F
    cls = F.MENU_CLASS.get(menu)
    if cls is not None:
        return cls in SOUP_CLASSES
    if any(w in menu for w in SOUP_MENU_KW):
        return True
    return any(any(k in i for k in BROTH_HINT_KW) for i, _ in ingredients)


# 본 육수량 계산 — "5~8배로 판단해서 써라"고 LLM에 맡겼더니 계산을 실제로 안 하고 "국 한 그릇=
# 보통 500~600ml"라는 감으로 찍어버림(2026-07-24 실측: 유부장국 건더기 16g→80~128ml여야 하는데
# 600ml로 나옴). 그래서 코드가 직접 계산해 고정값으로 못박는다 — LLM은 그 값을 그대로 쓰기만 함.
BROTH_WATER_MULTIPLIER = 6   # 건더기 총중량의 배수(5~8배 범위의 중간값)
BROTH_WATER_MIN = 300        # ml, 1인분이라도 국 한 그릇 답으려면 이 정도는 있어야 함(2026-07-24, 사용자 조정)
BROTH_WATER_MAX = 500        # ml, 1인분 기준 이보다 많으면 비현실적
BLANCH_WATER_ML = 200        # ml, 핏물빼기·데치기용(재료가 잠길 정도, 고정값 — 버리는 물이라 정밀 불필요)
NON_SOLID_GROUPS = ('조미료류', '유지류', '당류')   # 건더기 무게 계산에서 제외(양념·기름·당류)


def _broth_water_ml(ingredients):
    """국/탕/찌개의 본 육수량(ml)을 건더기 무게로 계산. LLM 판단에 맡기지 않고 고정값으로 프롬프트에 박는다."""
    nut = _ing_nut()
    solid_g = 0.0
    for name, amt in ingredients:
        if name.split(',')[0].strip() in ('물', '생수'):
            continue
        d = nut.get(name)
        if d and d.get('group') in NON_SOLID_GROUPS:
            continue
        solid_g += amt
    ml = max(BROTH_WATER_MIN, min(BROTH_WATER_MAX, solid_g * BROTH_WATER_MULTIPLIER))
    return int(round(ml / 10) * 10)


# 재료별 전처리 지침 — "위험해 보이면 LLM이 알아서 판단"이 아니라, 코드가 재료 데이터(식품군·칼륨값·
# 이름패턴)로 결정해서 프롬프트에 콕 집어 넘긴다. LLM은 여기 나열된 재료에만, 그것도 TIP이 아니라
# 조리 단계 안의 알맞은 위치에 반영한다(2026-07-24, 사용자 제안 — 접히는 안내만으론 안 읽고 넘어갈
# 수 있어 순응도가 떨어짐). 고칼륨 채소가 아닌 재료(양파·파·마늘 등)는 대상에서 자동으로 빠진다.
HIGH_K_GROUPS = ('채소류', '버섯류', '해조류')
HIGH_K_THRESHOLD = 400  # mg/100g. FOOK_임상검수_2 문서의 '고칼륨' 기준과 동일하게 맞춤.
HIGH_K_MIN_AMT = 10  # g. 마늘·생강처럼 100g당 밀도는 높아도 실제로는 1~수g만 쓰는 향신채는 제외
# (예: 마늘 0.3g은 K밀도 401mg/100g이라도 실제 칼륨 기여가 1.2mg뿐 — 데치라고 하면 무의미).

_ING_NUT = None
def _ing_nut():
    global _ING_NUT
    if _ING_NUT is None:
        import FOOK_adjust_levers as F
        if F.NUT is None:
            F.NUT = F.load_all()
        _ING_NUT = F.NUT[0]
    return _ING_NUT


def _prep_notes(ingredients):
    """재료별 전처리 지침을 [(재료명, 지침문구), ...]로 판정. 재료당 하나만(우선순위 순)."""
    import FOOK_adjust_levers as F
    nut = _ing_nut()
    notes = []
    for name, amt in ingredients:
        d = nut.get(name)
        if not d:
            continue
        group, k = d.get('group'), d.get('K')
        if F.is_processed_name(name, group):
            notes.append((name, '끓는 물에 살짝 데쳐 기름기와 첨가물을 뺀 후 사용한다.'))
        elif '통조림' in name:
            notes.append((name, '체에 밭쳐 국물(충전액)을 따라내고 흐르는 물에 헹궈 사용한다.'))
        elif group == '두류' and any(w in name for w in F.DRIED):
            notes.append((name, '찬물에 충분히 불린 후 처음 삶은 물은 버리고 새 물로 조리한다.'))
        elif any(w in name for w in F.DRIED):
            notes.append((name, '찬물에 충분히 불린 후 물기를 짜서 사용한다.'))
        elif group in HIGH_K_GROUPS and k is not None and k >= HIGH_K_THRESHOLD and amt >= HIGH_K_MIN_AMT:
            notes.append((name, '얇게 썰어 물에 담갔다가(또는 데쳐서) 사용한다.'))
    return notes


def _fmt_amt(a):
    """1g 이상=정수("150g"), 1g 미만=소수점 1자리("0.2g")로 표시.
    전부 :.0f로 반올림하면 후추 0.2g·마늘 0.3g 같은 소량 재료가 "0g"이 돼 재료 자체가
    안 쓰인 것처럼 LLM에 전달되는 문제가 있었음(2026-07-24, 사용자 지적으로 발견).
    단, 0.05g처럼 소수점 1자리로도 반올림하면 0이 되는 극소량(저염오이김치를 아주 작게
    스케일한 소금 0.025g 등 실사례로 발견)은 0.1g으로 바닥을 둔다 — 재료가 있는데 "0g"으로
    표시되면 그 재료가 안 쓰인 것처럼 보이는 건 똑같은 문제라서."""
    if a > 0:
        r = round(a, 1) or 0.1
    else:
        r = 0
    return f"{int(r)}g" if r == int(r) else f"{r}g"


def _prompt(menu, ingredients, original_steps):
    ing_txt = ", ".join(f"{i}({_fmt_amt(a)})" for i, a in ingredients)
    base = original_steps or "(원본 조리과정 없음 — 아래 재료로 표준 조리법을 간단히 제시)"
    prep = _prep_notes(ingredients)
    prep_block = ""
    if prep:
        lines = "\n".join(f"- {n}: {tip}" for n, tip in prep)
        prep_block = (
            "\n[코드 지정 전처리 필수 재료 — TIP이 아니라 조리 단계 안의 알맞은 위치에 반드시 반영]\n"
            f"{lines}\n"
            "이 목록은 코드가 재료 데이터로 판정한 것이니 그대로 따르고, 목록에 없는 재료에는 "
            "이런 전처리를 임의로 추가하지 마라."
        )
    broth_rule = ""
    if _has_broth(menu, ingredients):
        # 임상 지침: 육수를 아예 내지 않는다. 나트륨·칼륨·인이 국물로 빠져나와 그대로 섭취되기 때문.
        water_ml = _broth_water_ml(ingredients)
        broth_rule = (
            "\n[★ 육수 규칙 — 이 메뉴는 국물이 있다]\n"
            "- 육수를 절대 내지 마라. 그냥 '맹물(생수)'로 끓여라.\n"
            "- 멸치·다시마·표고·사골·가다랑어·채소(채수) 등으로 육수 내는 것 모두 금지 "
            "(나트륨·칼륨·인이 국물로 우러나온다).\n"
            "- 시판 코인육수·다시다·장국베이스·요리수 등 가공 육수도 절대 금지.\n"
            "- '끓였다 물을 버리고 다시 끓이기(2차 육수)'도 하지 마라.\n"
            "- 원본 조리과정에 '멸치육수/다시마육수/사골국물' 등이 있으면 전부 '물'로 바꿔 써라.\n"
            "- 완성 국물은 '건더기 위주로 먹고 국물은 100ml 미만(종이컵 반 컵 이하)만'을 팁에 포함할 수 있다.\n"
            "- [코드 계산 완료 — 아래 두 수치는 이미 재료 무게로 계산된 값이다. 재계산·수정 금지, "
            "그리고 반드시 지정된 단계에만 써라. 서로 뒤바뀌면 안 된다]\n"
            f"  · 데치기/핏물제거용 물: {BLANCH_WATER_ML}ml (그 요리에 데치기·핏물빼기 단계가 "
            "있을 때만 사용)\n"
            f"  · 본 육수/국물용 물: {water_ml}ml (국물을 본격적으로 끓이는 단계에 반드시 사용)\n"
            "  물 쓰는 단계가 2개(고기·뼈 데치기 + 본 육수)인 요리는:\n"
            f"    1차 데치기 단계에 반드시 {BLANCH_WATER_ML}ml를 쓰고 그 물은 버려라 "
            f"(예: '냄비에 물 {BLANCH_WATER_ML}ml를 부어 끓인 뒤 재료를 넣어 데치고, 데친 물은 "
            "버린다').\n"
            f"    2차 본 육수 단계에 반드시 {water_ml}ml를 쓰라 (예: '냄비에 데친 재료와 물 "
            f"{water_ml}ml를 넣고 끓인다'). {water_ml}ml가 데치기 단계에 들어가거나, 본 육수 "
            "단계에 물 양이 아예 빠지면 안 된다.\n"
            f"  물 쓰는 단계가 1개(두부·채소 위주라 데치기가 없는 국)인 요리는: 그 한 단계에만 "
            f"{water_ml}ml를 써라.\n"
            "- '적당량의 물', '물을 자작하게 붓는다' 같은 모호한 표현은 절대 쓰지 마라.\n"
            "- 원본에 '체(거름망)에 밭쳐 기름기를 제거한다'처럼 체로 기름을 뺀다는 표현이 있으면 "
            "정정하라 — 체는 거품·찌꺼기 같은 고형물만 거를 뿐 기름은 걸러내지 못한다. "
            "'거품과 불순물(찌꺼기)을 걸러낸다'로 바꿔 쓰고, 기름기 제거가 실제 목적이면 "
            "'식혀서 위에 뜬 기름을 걷어낸다' 또는 '키친타월(기름종이)로 겉기름을 제거한다'처럼 "
            "실제로 기름이 빠지는 방법으로 정확하게 다시 써라."
        )
    return (
        f"[메뉴] {menu}\n\n"
        f"[투석 기준으로 조정된 최종 재료]\n{ing_txt}\n\n"
        f"[원본 조리과정]\n{base}\n"
        f"{broth_rule}\n"
        f"{prep_block}\n\n"
        "[편집 지침]\n"
        "1) 원본 조리과정을 기반으로, 위 '최종 재료'에 맞게 재료명·양을 반영해 다시 써라. "
        "**[최종 재료]에 있는 재료는 예외 없이 전부(주재료든 된장·고추장 같은 양념 베이스든 "
        "식초·후추·참기름·마늘 같은 풍미 양념이든) 조리 단계 어딘가에 반드시 등장해야 한다** — "
        "하나도 빠짐없이. 임의로 생략하지 마라.\n"
        "2) 화학조미료·핵산조미료·복합조미료·미원·다시다 같은 인공 조미료 문구는 모두 삭제하라.\n"
        "3) 조리 단계 자체는 원본 흐름을 유지하고 간은 약하게(저염) 반영하라. "
        "국물이 있으면 위 [육수 규칙]대로 육수 없이 맹물로 끓이도록 반영하라. "
        "[코드 지정 전처리 필수 재료]가 있으면 그 재료를 다루는 단계에 지시된 전처리를 자연스럽게 포함하라.\n"
        "4) ① ② ③ 형식, 6단계 이내로 간결하게.\n"
        "5) TIP은 위 [최종 재료] 표에 이미 담긴 내용(양·재료)이나 조리 단계에 이미 반영한 내용을 "
        "다시 말하는 게 아니라, 그 어디에도 없는 '환자가 먹을 때 지킬 행동 지침'일 때만 써라. "
        "**'국물은 OOml 미만만 먹기'는 위 [★ 육수 규칙]이 실제로 붙어있는 국물 요리에만 해당하는 "
        "예시다 — 그 규칙 블록이 없다면(예: 무침·볶음·구이처럼 국물 자체가 없는 요리) 이 예시를 "
        "그대로 베껴 쓰지 마라. 국물 없는 요리는 국물 관련 TIP을 아예 쓰지 마라.** "
        "'간장/소금을 적게 써서 저염으로 조리하라'는 조언, 그리고 [코드 지정 전처리 필수 재료]에 "
        "이미 넣은 내용(채소 데치기·물에 담그기 등)의 반복은 절대 쓰지 마라 — 전자는 이미 재료표의 정확한 양으로 반영돼 있고, "
        "후자는 이미 조리 단계 안에 반영했으므로 TIP에서 또 말하면 같은 말이 두 번 나오는 것뿐이다. "
        "해당하는 행동 지침이 없으면 맨 마지막 줄의 TIP 자체를 쓰지 마라. "
        "화살표(↓, → 등) 기호는 쓰지 말고 말로 풀어써라. 단계와 중복 금지."
    )

def _missing_ingredients(text, ingredients):
    """생성된 조리법에 재료표의 재료가 다 들어있는지 점검. 완전 일치가 아니라 관대하게 봐서
    (① 쉼표로 나뉜 단어 중 아무거나 하나만 있어도 통과 — "두부, 유부"인데 텍스트가 "유부"만
    써도 정상, ② 첫 2글자까지만 일치해도 통과 — 후추="후춧가루" 등) 자연스러운 동의어·축약형은
    오탐으로 잡지 않는다 — 된장·고추장·물엿처럼 아예 다른 단어가 통째로 빠진 진짜 누락만 잡는다."""
    missing = []
    for name, amt in ingredients:
        tokens = [t.strip() for t in name.split(',') if t.strip()]
        if not tokens:
            continue
        if any(t in text or (len(t) >= 2 and t[:2] in text) for t in tokens):
            continue
        missing.append(tokens[0])
    return missing


def edit_recipe(menu, ingredients, model="gpt-4o-mini", source_menu=None):
    """menu: 표시용 메뉴명, ingredients: [(재료명, 양g)]. 편집된 조리과정 문자열 반환.
    source_menu: 재료 교체로 이름이 바뀐 경우의 원래 메뉴명(조리법 원본은 이 이름으로 찾음).
    예) 오징어→고등어 교체로 '고등어볶음'이 됐어도 원본 조리과정은 '오징어볶음'에 있음.

    생성 직후 재료 누락 여부를 코드로 검증하고(사람이 보는 게 아니라 서버가 즉시 자동으로),
    빠진 재료가 있으면 그것만 콕 집어 딱 한 번 더 요청한다 — LLM이 가끔(특히 비슷한 재료가
    같이 있을 때, 예: 고추장+고춧가루) 재료 하나를 빠뜨리는 확률적 실수를 잡기 위함
    (2026-07-24, 사용자 지적으로 발견). 재시도 결과도 놓치면 그냥 그대로 반환(무한 재시도 안 함)."""
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. `set OPENAI_API_KEY=sk-...` 후 실행하세요.")
    client = OpenAI(api_key=key)
    original = _steps().get(source_menu or menu) or _steps().get(menu)
    prompt = _prompt(menu, ingredients, original)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()

    missing = _missing_ingredients(text, ingredients)
    if missing:
        retry_prompt = (
            f"{prompt}\n\n[검증 결과] 방금 쓴 조리법에 다음 재료가 조리 단계 어디에도 안 보인다: "
            f"{', '.join(missing)}. 나머지 내용은 유지하되, 이 재료들을 알맞은 단계에 반드시 "
            "포함해서 처음부터 다시 써라."
        )
        resp2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": retry_prompt}],
            temperature=0.3,
        )
        text = resp2.choices[0].message.content.strip()
    return text


TTS_VOICES = ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'ash', 'coral', 'sage', 'verse', 'ballad')


def text_to_speech(text, voice='nova', model='tts-1'):
    """조리과정 텍스트를 음성(mp3 bytes)으로 변환. voice는 TTS_VOICES 중 하나."""
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. `set OPENAI_API_KEY=sk-...` 후 실행하세요.")
    if voice not in TTS_VOICES:
        raise ValueError(f"voice는 {TTS_VOICES} 중 하나여야 합니다: {voice}")
    client = OpenAI(api_key=key)
    resp = client.audio.speech.create(model=model, voice=voice, input=text)
    return resp.read()


if __name__ == "__main__":
    # 단독 테스트: 레버로 한 끼 만들고 앵커 메뉴 조리과정 편집
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    menu = "제육고추장불고기"
    b = F.meal_bounds(60)
    # 그 메뉴 하나로 재료 뽑아 조정
    _, _, inst, _ = F.adjust([menu, "백미밥", "배추된장국", "콩나물무침", "배추김치"], b, anchor=menu)
    ings = [(i["ing"], i["amt"]) for i in inst if i["menu"] == menu]
    print(f"[{menu}] 조정된 재료:", ings)
    print("\n--- 원본 조리과정 ---")
    print(_steps().get(menu, "(없음)"))
    print("\n--- LLM 편집 결과 ---")
    print(edit_recipe(menu, ings))
