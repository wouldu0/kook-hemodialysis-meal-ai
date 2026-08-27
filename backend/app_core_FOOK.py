# -*- coding: utf-8 -*-
"""
app_core_FOOK.py — 투석 한끼 앱 코어 (생성 → resample → 레버조정, end-to-end)

입력 모드:
  1) 메뉴 지정   : 그 메뉴를 슬롯에 고정(앵커) + 나머지 자연스럽게 생성
  2) 재료 지정   : 그 재료 쓰는 메뉴 역검색 → 앵커
  3) 랜덤        : 자연스러운 한끼 생성
공통: 생성 → 레버(투석 5영양 조정) → 5영양 통과할 때까지 resample → 완성 한끼 반환.

실행:
  cd backend
  python app_core_FOOK.py
"""
import os, sys, copy, threading
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# 이 파일 위치 기준 상대경로를 쓰므로 어느 PC/폴더에서도 그대로 실행된다.
_HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(_HERE, 'Diet-Generation-As-Sequence-master', 'Diet-Generation-As-Sequence-master', 'Code')
DATA = os.path.join(_HERE, 'data')
# 체크포인트: RL(레버 조정량 보상) 파인튜닝본. imit=0.02가 스윕 최적(lr 5e-5, 300에폭, 앵커인지 레버 기준 학습).
#   모방 대비 n=2000: 앵커보존 .649->.676, 전체보존 .279->.344, 5영양 첫시도통과 52->71%, 다양성 633->611(-3%).
#   imit을 더 낮추면(0) 5영양 82%까지 오르나 다양성 493(-22%)+종료토큰 붕괴 -> 나쁜 거래.
#   되돌리려면 'results_FOOK','checkpoints'(순수 모방)로.
CKPT = os.path.join(CODE, 'results_sweep_FOOK', 'i002')
SPECIAL = {0, 825, 826}
sys.path.insert(0, CODE)

import tensorflow as tf
from util import (nutrition_preprocessor, diet_sequence_preprocessor,
                  food_to_token, diet_to_incidence)
from Model import Sequence_Generator
import FOOK_adjust_levers as F


# ---------------- 로드 (한 번만) ----------------
print('모델·데이터 로딩...')
feature = pd.read_csv(os.path.join(DATA, 'FOOK_nutrition.csv'))
nutrient_data, food_dict = nutrition_preprocessor(feature_data=feature)()
name2idx = {v: k for k, v in food_dict.items()}

diet_raw = pd.read_csv(os.path.join(DATA, 'FOOK_meals_for_model.csv'))
dsp = diet_sequence_preprocessor(sequence_data=diet_raw.copy(), DB_quality='correct2', integrate=False)
diet_np = food_to_token(dsp(nutrient_data), nutrient_data, empty_delete=True, num_empty=3)
incidence = diet_to_incidence(diet_np, food_dict)

kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(food_dict), 'batch_size': diet_np.shape[0], 'imitation_only': True}
gen = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
tf.train.Checkpoint(generator=gen).restore(tf.train.latest_checkpoint(CKPT)).expect_partial()

# 메뉴 → 대표 슬롯(0..4): 학습데이터에서 그 메뉴가 주로 나온 열
slot_of = {}
tmp = defaultdict(Counter)
for _, r in diet_raw.iterrows():
    for c in range(5):
        tmp[r.iloc[c]][c] += 1
for m, cnt in tmp.items():
    slot_of[m] = cnt.most_common(1)[0][0]

# 재료 → 메뉴 역검색 (FOOK_menus_kor.csv: 메뉴별 재료)
mk = pd.read_csv(os.path.join(DATA, 'FOOK_menus_kor.csv'))
mk['menu'] = mk['name'].ffill()
menu_ings = defaultdict(list)
for _, r in mk.iterrows():
    if isinstance(r['ingredient'], str):
        menu_ings[r['menu']].append(r['ingredient'])

# 슬롯 마스크: 슬롯별 허용 Class + 밥/국/김치 중복 방지
UNIQUE_GROUPS = [{'밥', '일품밥', '일품(간식)'}, {'국', '수프(간식)'}, {'김치'}]
_m2c = dict(zip(feature['name'], feature['Class']))
_slot_allowed = defaultdict(set)
for _, _r in diet_raw.iterrows():
    for _j in range(5):
        _c = _m2c.get(_r.iloc[_j])
        if isinstance(_c, str):
            _slot_allowed[_j].add(_c)
_tokclass = {i: _m2c.get(n) for i, n in food_dict.items()}
# 김치 그룹은 Class가 아니라 이름 기반(겉절이가 데이터상 샐러드로도 분류돼 있어서)
_KIMCHI_SUF = ('김치', '겉절이', '깍두기', '물김치', '소박이', '짠지')
def _grp(name, c):
    if name and name.endswith(_KIMCHI_SUF):
        return 2                                   # 김치 그룹
    if c in UNIQUE_GROUPS[0]:
        return 0                                   # 밥
    if c in UNIQUE_GROUPS[1]:
        return 1                                   # 국
    return None
SLOT_OK = {}
for _j in range(5):
    _a = np.zeros(len(food_dict))
    for _t, _c in _tokclass.items():
        if isinstance(_c, str) and _c in _slot_allowed[_j]:
            _a[_t] = 1.0
    SLOT_OK[_j] = _a
TOK_GRP = {}
for _t, _n in food_dict.items():
    _g = _grp(_n, _m2c.get(_n))
    if _g is not None:
        TOK_GRP[_t] = _g
GRP_TOK = {gi: [t for t, g in TOK_GRP.items() if g == gi] for gi in range(len(UNIQUE_GROUPS))}

print('레버(식약청 DB) 로딩...')
F.NUT = F.load_all()

# ── 군 메뉴 추가 (선택/앵커링용, 재학습 없이 post-substitution) ──
import json as _json
gun_nut = pd.read_csv(os.path.join(DATA, 'FOOK_gun_nutrition.csv'))
gun_recipes = _json.load(open(os.path.join(DATA, 'FOOK_gun_recipes.json'), encoding='utf-8'))
gun_names = set(gun_nut['name'])
gun_class = dict(zip(gun_nut['name'], gun_nut['Class']))
CLASS_TO_SLOT = {'밥': 0, '일품밥': 0, '일품(간식)': 0, '면': 0,
                 '국': 1, '수프(간식)': 1,
                 '볶음': 2, '조림': 2, '구이': 2, '찜': 2, '튀김': 2, '전': 2,
                 '샐러드': 3, '반찬': 3, '곡류(간식)': 3, '과일(간식)': 3,
                 '음료(간식)': 3, '우유(간식)': 3, '김치': 4}
gun_slot = {m: CLASS_TO_SLOT.get(c, 3) for m, c in gun_class.items()}
F.NUT[5].update({m: [(i, a) for i, a in rec] for m, rec in gun_recipes.items()})  # 레버 master에 군 레시피
# 열량 레버의 밥계열(RICE)은 FOOK_nutrition.csv(월 824)에서만 만들어짐 → 군 밥/일품밥/면도 등록.
# 안 하면 군 일품밥 앵커에서 열량 레버가 줄일 대상을 못 찾아 초과를 못 고침.
F.MENU_CLASS.update(gun_class)
F.RICE.update(m for m, c in gun_class.items() if c in F.RICE_CLASSES)
for m, rec in gun_recipes.items():                                                # 재료검색 인덱스
    menu_ings[m] = [i for i, a in rec]
model_menu_names = [food_dict[i] for i in range(1, len(food_dict) - 2)]

# 김치가 '주재료'로 든 메뉴(김치찌개·김치볶음밥 등)는 제외한다.
# 이유: 김치 절임염은 '채소류'라 나트륨 판정(첨가염)에서 빠지는데, 이런 메뉴는 김치를 줄일 수도 없어
#       (김치 적은 김치찌개는 성립 안 함) 실제 총 나트륨이 788~1028mg까지 새면서 '통과'로 잡힌다.
#       투석 환자에게 권하기 어려운 고나트륨 메뉴이므로 생성·선택 양쪽에서 뺀다.
# 단, '김치 반찬'(배추김치·깍두기·열무김치 등)은 제외 대상이 아니다 → lever_kimchi가 저염김치로 교체.
def _is_kimchi_side(name):
    return (name.endswith('김치') or '깍두기' in name or '소박이' in name
            or name.endswith('겉절이') or name.endswith('물김치'))


KIMCHI_MAIN = {m for m, igs in menu_ings.items()
               if not _is_kimchi_side(m)
               and any(('김치' in ig or '깍두기' in ig or '겉절이' in ig) for ig in igs)}
# 뼈육수가 본체라 육수를 빼면 남는 게 없는 메뉴(꼬리곰탕=당면만) + 소고기맑은국으로 중복되는
# 사골곰국·사골곰탕(설렁탕 하나로 통합)을 제외. 나머지 뼈육수 메뉴는 육수→맹물 + 이름 변경으로 살린다.
BONE_EXCLUDE = {'꼬리곰탕', '사골곰국', '사골곰탕'}
# 콩가루(2026-07-22, 사용자 요청 "콩가루도 안써야지") — 이 데이터에서 콩가루는 전부 국물 걸쭉이
# 용도(콩가루배춧국·우무냉국·콩가루배추국)라 대체할 수도 없고(제형이 안 맞음, 앞서 대두→두부 매핑
# 제거 사유와 동일) 빼는 것도 애매해 아예 메뉴 자체를 생성 선택지에서 제외.
GARU_EXCLUDE = {m for m, ings in F.NUT[5].items() if any('콩가루' in ing for ing, _ in ings)}
# 소스/양념장류 + 나물/무침류 + 기타 중복 메뉴(2026-07-22, 전수조사) — 한 메뉴 이름이 다른 메뉴
# 이름의 접두어인 모든 쌍을 스캔해 레시피 대조. 같은 요리의 재료·비율만 다른 중복 기록으로 확인된
# 것들만 선정(완전히 다른 요리로 판단된 모닝빵/모닝빵샌드위치 등은 제외 안 함). 짧은/단순한 이름을
# 표준으로 남기고 나머지를 생성 후보에서 제외(데이터 삭제 아님 — 대부분 학습모델 핵심 어휘라 행
# 삭제 시 vocabulary 크기가 깨져 체크포인트 복원이 실패할 위험이 있어 안전하게 선택지만 차단).
DUP_MENU_EXCLUDE = {
    # 소스/양념장류 (1차 발견)
    '물만두양념장', '삼색나물비빔밥양념장', '양배추샐러드키위드레싱', '양배추찜쌈장',
    '양상추샐러드오리엔탈드레싱', '치킨가스소스', '탕수육소스', '함박스테이크소스',
    '해물파전달래장', '훈제오리구이머스터드소스',
    '훈제오리구이머스타드',   # 훈제오리구이 3중복 중 나머지 하나(위와 별개로 추가 발견)
    # 나물/무침류(재료 동일 or 거의 동일, 양념 비율만 다름)
    '깻잎순나물볶음',        # =깻잎순나물(완전 동일 레시피)
    '유채나물무침',          # 재료 동일, 양념 비율만 다름 → 유채나물
    '비름나물무침', '비름나물된장무침',   # → 비름나물
    '숙주나물무침', '오리주물럭볶음', '양념깻잎지무침',
    '고갈비구이', '섭산적구이',
    '돈삼겹수육새우젓', '스크램블에그케첩',
    '우유미숫가루',          # → 미숫가루우유 (둘 다 SNACK_EXCLUDE_KW '미숫'에도 걸려 간식으론 이미 제외)
    # 완전히 다른 재료 구성이지만 '보리차 세트'/'우유 곁들임'처럼 굳이 별도 유지할 실익이 없는 것
    '물만두보리차',          # → 물만두 (보리차를 실제로 같이 제공 안 함)
    '커스타드우유',          # → 커스타드
    '삼색수제비국',          # → 삼색수제비
    '액상요구르트',          # → 요구르트 (재료 동일 '요구르트, 액상', 양만 75g/70g)
    '등심돈까스소스',        # → 돈까스소스 (재료 구성·양 사실상 동일)
    # 참깨·들깨·견과류를 재료에서 제거(2026-07-24)한 뒤 대조해보니, 아래는 사실상 같은 레시피의
    # 중복 기록이었음(육수용 멸치·다시마 차이는 어차피 맹물로 치환돼 무관, 남는 차이가 반올림 수준
    # 이거나 조리 시점 표기 차이(생것/삶은것 등)뿐) → 더 단순한 이름 쪽을 표준으로 남김.
    '고사리들깨나물',        # → 고사리나물 (들깨·참깨 빼면 재료·양 완전 동일)
    '들깨무나물',            # → 무나물 (재료 동일, 양 차이는 반올림 수준)
    '들깨수제비국',          # → 수제비국 (수제비 반죽 30g 동일, 멸치는 육수용이라 무관, 채소만 차이)
    '브로콜리들깨볶음',      # → 브로콜리볶음 (브로콜리 52.5g 완전 동일)
    '죽순채들깨볶음',        # → 죽순채볶음 (생것/삶은것은 조리 시점 표기 차이로 판단)
    '토란대들깨나물',        # → 토란대나물 (위와 동일 이유)
    '잔멸치건포도볶음',      # → 잔멸치호두볶음(호두는 GARNISH_STRIP_KW로 별도 제거).
                             # 원본 raw 데이터 자체가 "건포도"라 쓰고 재료엔 돼지허파·부추가 들어간
                             # 입력 오류였음(조리법 텍스트엔 멸치+건포도만 언급) — 오류난 레시피라
                             # 병합보다 차단이 더 안전하다고 판단.
}
# '성인' 계열(원본 소아용 추정, 작은 분량) — 성인용 큰 분량판을 표준으로 쓰고 이름에서 '성인'을
# 떼기로 함(FOOK_adjust_levers.MENU_DISPLAY_RENAME이 표시명 변경 담당) → 원본은 생성에서 제외
# (안 그러면 같은 표시이름 아래 서로 다른 두 레시피가 나오는 충돌 발생).
ADULT_ORIGINAL_EXCLUDE = {'채소달걀말이', '토마토달걀볶음'}
BLOCK_MENUS = KIMCHI_MAIN | BONE_EXCLUDE | GARU_EXCLUDE | DUP_MENU_EXCLUDE | ADULT_ORIGINAL_EXCLUDE
BLOCK_TOK = {name2idx[m] for m in BLOCK_MENUS if m in name2idx}   # 생성에서 확률 0으로 막을 토큰

# 월+군 전체 선택가능. 잡곡밥류(→백미밥 치환)·김치주재료·뼈육수·콩가루 메뉴는 선택지에서 제외.
ALL_MENU_NAMES = sorted((set(model_menu_names) | gun_names) - F.JAPGOK_RICE - BLOCK_MENUS)
print(f'군 메뉴 {len(gun_names)}개 추가, 김치주재료 {len(KIMCHI_MAIN)}개 + 뼈육수 {len(BONE_EXCLUDE)}개 + '
      f'콩가루 {len(GARU_EXCLUDE)}개 제외 (총 선택가능 {len(ALL_MENU_NAMES)}개)')


# ── 메뉴별 정체성(주) 재료 — 한 끼 안에서 재료 겹침 판정용 (2026-07-22) ──────────
# lever_adjust의 is_identity_ingredient와 같은 개념: (1) 메뉴명에 든 재료(오징어볶음의 오징어)
# 우선, 없으면 (2) 물·조미료를 뺀 건더기 중 양이 가장 많은 재료(어묵볶음의 어묵).
def _menu_identity_ing(menu):
    ings = F.NUT[5].get(menu)
    if not ings:
        return None
    for ing, amt in ings:
        core = ing.split(',')[0].strip()
        if core and core in menu:
            return core
    ing_nut = F.NUT[0]
    solids = [(ing, amt) for ing, amt in ings
              if ing.split(',')[0].strip() not in ('물', '생수')
              and ing_nut.get(ing, {}).get('group') != '조미료류']
    if not solids:
        return None

    def _n(a):
        try:
            return float(a)
        except (TypeError, ValueError):
            return 0.0
    best = max(solids, key=lambda x: _n(x[1]))
    return best[0].split(',')[0].strip()


MENU_IDENTITY_ING = {m: v for m in ALL_MENU_NAMES for v in [_menu_identity_ing(m)] if v}


def _has_ingredient_clash(menus):
    """같은 끼니 안에서 서로 다른 메뉴가 같은 정체성 재료를 쓰는지 (예: 오징어볶음+오징어젓)."""
    seen = set()
    for m in menus:
        ing = MENU_IDENTITY_ING.get(m)
        if ing:
            if ing in seen:
                return True
            seen.add(ing)
    return False


# 자연나트륨군(어패류·해조류) 끼니당 개수 제한 (2026-07-22, 나트륨 예산이월로도 못 잡는 '트리플 겹침' 대응).
# 실측(1095끼): 0개4.4%·1개52.9%·2개36.3%(어묵볶음+미역줄기나물 등 흔한 정상조합)·3개+6.4%(문제구간).
# 1개로 제한하면 42.7%가 거부돼 과함 → 2개까지 허용, 3개부터 거부.
HIGH_NA_GROUPS = {'어패류 및 그 제품', '해조류'}
HIGH_NA_MAX = 2


def _menu_is_high_na_group(menu):
    ings = F.NUT[5].get(menu)
    if not ings:
        return False
    ing_nut = F.NUT[0]
    return any(ing_nut.get(ing, {}).get('group') in HIGH_NA_GROUPS for ing, _ in ings)


HIGH_NA_MENUS = {m for m in ALL_MENU_NAMES if _menu_is_high_na_group(m)}


def _has_seafood_overload(menus):
    """어패류·해조류 메뉴가 한 끼에 HIGH_NA_MAX(2)개를 넘으면 True (트리플 겹침 등 극단조합 거부)."""
    return sum(1 for m in menus if m in HIGH_NA_MENUS) > HIGH_NA_MAX


# 고인비율군(가공식품·유제품·견과류) 끼니당 개수 제한 (2026-07-22, 나트륨과 같은 방식).
# 실측(1095끼): 0개35.2%·1개46.8%(정상다수)·2개16.9%·3개+1.2%(진짜꼬리, 채소/단백질과 달리 절벽 뚜렷).
# 채소류(3개+99.3%)·단백질군(2개+92%)은 정상범위라 제한 안 함 — 이 군만 예외적으로 고농도 겹침이 드묾.
HIGH_P_GROUPS = {'조리가공식품류', '우유 및 그 제품', '견과류 및 종실류'}
HIGH_P_MAX = 2


def _menu_is_high_p_group(menu):
    ings = F.NUT[5].get(menu)
    if not ings:
        return False
    ing_nut = F.NUT[0]
    return any(ing_nut.get(ing, {}).get('group') in HIGH_P_GROUPS for ing, _ in ings)


HIGH_P_MENUS = {m for m in ALL_MENU_NAMES if _menu_is_high_p_group(m)}


def _has_high_p_overload(menus):
    """가공식품·유제품·견과류 메뉴가 한 끼에 HIGH_P_MAX(2)개를 넘으면 True."""
    return sum(1 for m in menus if m in HIGH_P_MENUS) > HIGH_P_MAX


def _is_staple_menu(m):
    """밥·김치는 끼니마다 반복되는 게 정상(한식 식단 구조)이라 하루 중복 체크에서 제외.
    국은 제외하지 않음 — 하루 세 끼 국이 다 같은 건 부자연스러움."""
    return _grp(m, _m2c.get(m) or gun_class.get(m)) in (0, 2)


print('준비 완료.\n')


# ---------------- 앵커링 생성 ----------------
# FastAPI는 동기 엔드포인트를 스레드풀에서 돌리므로, 동시에 들어온 요청 두 개가
# 이 함수를 동시에 호출하면 전역으로 공유하는 gen(encoder/decoder) 모델을 두 스레드가
# 동시에 건드리게 된다. TF/Keras 레이어는 이런 동시 호출에 안전하다고 보장되지 않고,
# 특히 최초 호출 시 지연 빌드(build)되는 GRU 셀의 kernel 가중치가 두 스레드의 빌드가
# 겹치면서 일부만 생성된 채로 쓰이는 레이스가 실제로 재현됐다
# ("'GRUCell' object has no attribute 'kernel'"). 배치 크기가 작고(seed 12~48개,
# 5스텝) 락을 걸어도 지연은 무시할 만해서, 정확성을 위해 전역 락으로 직렬화한다.
_GEN_LOCK = threading.Lock()


def gen_batch(anchor_menu=None, n=12, temp=0.8):
    """seed=랜덤 실제한끼 n개. anchor_menu 있으면 그 슬롯 고정 후 나머지 샘플. (batch>1 필수: squeeze 회피)"""
    idx = np.random.randint(diet_np.shape[0], size=n)
    seeds = diet_np.numpy()[idx].copy()      # (n,7)
    fixed = {}
    if anchor_menu is not None and anchor_menu in name2idx:
        s = slot_of.get(anchor_menu, 2)
        seeds[:, s + 1] = name2idx[anchor_menu]
        fixed[s] = name2idx[anchor_menu]

    with _GEN_LOCK:
        enc_hidden = tf.zeros([n, gen.encoder.units])
        enc_output, enc_hidden = gen.encoder(seeds, enc_hidden)
        dec_hidden = copy.deepcopy(enc_hidden)
        res = np.zeros((n, 7), dtype=int); res[:, 0] = seeds[:, 0]; res[:, -1] = 826
        used = [set(fixed.values()) for _ in range(n)]
        used_grp = [{TOK_GRP[t] for t in fixed.values() if t in TOK_GRP} for _ in range(n)]
        for j in range(5):                       # slot s=j → position j+1
            outputs, dec_hidden, _ = gen.decoder(seeds[:, j], dec_hidden, enc_output)
            probs = np.array(outputs, dtype=float)
            if probs.ndim == 1:
                probs = probs[None, :]
            for b in range(n):
                if j in fixed:
                    res[b, j + 1] = fixed[j]; continue
                p = probs[b].copy()
                for t in SPECIAL: p[t] = 0.0
                for t in BLOCK_TOK: p[t] = 0.0        # 김치주재료 메뉴 생성 금지
                for t in used[b]: p[t] = 0.0
                masked = p * SLOT_OK[j]                       # 슬롯 허용 카테고리
                for gi in used_grp[b]:                        # 밥/국/김치 중복 금지
                    masked[GRP_TOK[gi]] = 0.0
                if masked.sum() > 0:
                    p = masked
                p = np.clip(p, 1e-12, None); p = p ** (1.0 / temp); p /= p.sum()
                tok = int(np.random.choice(len(p), p=p))
                res[b, j + 1] = tok; used[b].add(tok)
                gi = TOK_GRP.get(tok)
                if gi is not None:
                    used_grp[b].add(gi)
    return [[food_dict[int(t)] for t in r if int(t) not in SPECIAL] for r in res]


# 서버가 요청을 받기 전에 한 번 미리 호출해서, encoder/decoder GRU의 지연 빌드(kernel
# 가중치 생성)를 단일 스레드 상태에서 끝내둔다. 이렇게 안 하면 배포 직후 첫 실제 요청이
# (동시 요청이 아니어도) 빌드 시점과 겹쳐서 위와 같은 레이스를 유발할 수 있다.
# n은 반드시 1보다 커야 한다 — n=1이면 Decoder.call()의 tf.squeeze()가 배치 축까지
# 없애버려 다음 스텝의 Attention 레이어가 깨진다(gen_batch 자체의 기존 제약, 위 docstring 참고).
gen_batch(n=2)


# ---------------- 전체 플로우 ----------------
def passes(t, b):
    # 인(P)은 실제 섭취 인(raw P, 하드 기준)으로 판정한다 — 흡수보정 인(Peff, FOOK_adjust_levers.py
    # 참고)은 대체재 선호도 등 최적화용 소프트 신호일 뿐, 여기서(최종 통과/실패)는 쓰지 않는다.
    ok = (b['Elo'] <= t['E'] <= b['Ehi'] and b['Plo'] <= t['protein'] <= b['Phi'] and
          t['K'] < b['Kmax'] and t['P'] < b['Pmax'] and t['Na_season'] <= b['Namax'])
    # custom sodium(총나트륨 하루 상한)을 설정한 사용자만 총나트륨도 추가 hard constraint로 본다
    # (첨가염 조건은 그대로 유지 + AND). custom 없는 사용자는 이 분기를 아예 안 타서 위 결과
    # 그대로다 — b['custom_sodium_active']는 값(1965 등) 비교가 아니라 명시적 flag로 판단한다.
    if b.get('custom_sodium_active'):
        ok = ok and t['Na'] <= b['Na_total_target']
    return ok


def _total_na_warning(after):
    """첨가염 판정(소금1g)은 통과해도, 미역국·해산물·절임처럼 줄여도 못 맞추는 '음식 자체 나트륨'이
    많으면 총 나트륨(소금2g=786mg)을 넘을 수 있다. 이땐 끼니 단위로 안내(하루가 아니라 이 끼니에서 바로)."""
    if (after.get('Na') or 0) > F.NA_TOTAL_WARN:
        return (f'이 끼는 미역국·해산물·절임 등으로 총 나트륨이 높아요 '
                f'(총 {after["Na"]:.0f}mg ≈ 소금 {after["Na"] / 393:.1f}g). '
                f'국물은 100ml 미만(종이컵 반 컵 이하)만 드시거나 반찬을 조금 남기시면 좋아요.')
    return ''

def make_meal(menu=None, ingredient=None, W=60, tries=48, temp=0.8, bounds=None, used_today=None):
    # tries=후보 수(resample). 12는 부족했음: 실패한 앵커들이 인 상한에 118~202mg이나 여유가
    # 있는데도 저인 조합을 못 찾은 것뿐이었음(앵커 탓이 아니었음). 48로 올려 실패 5.0%->1.7%.
    # 비용은 거의 없음(35->41ms): gen_batch가 배치 한 방에 n개를 뽑고, 레버 루프는 첫 통과에 조기종료.
    # bounds=이 끼의 목표. 하루 이어서 만들 때 '남은 예산'을 넘겨받는다(없으면 첫 끼 기준).
    # used_today=오늘 이미 나온 메뉴(하루 중복 방지). 유저가 직접 지정한 앵커 메뉴는 예외(재요청 존중).
    used_today = used_today or set()
    anchor = menu
    note = ''
    if anchor is None and ingredient:
        cands = [m for m, igs in menu_ings.items()
                 if any(ingredient in ig for ig in igs) and (m in name2idx or m in gun_names)]
        if cands:
            anchor = np.random.choice(cands)
            note = f'(재료 "{ingredient}" → 메뉴 "{anchor}" 선택)'
        else:
            note = f'(재료 "{ingredient}" 쓰는 메뉴 없음 → 랜덤)'
    if anchor in F.JAPGOK_RICE:
        # 선택지에서 이미 뺐지만 직접 요청될 수 있으므로, 조용히 바꾸지 말고 이유를 알림
        note += (f' ("{anchor}"은 잡곡의 인·칼륨이 높아 투석 환자에게 권하지 않아요 → 흰쌀밥으로 만듭니다)')
        anchor = F.WHITE_RICE
    elif menu is not None and anchor not in name2idx and anchor not in gun_names:
        # 유저가 직접 지정한 메뉴명이 월/군 어디에도 없는 경우 — 예전엔 그냥 앵커 없이(=랜덤)
        # 조용히 넘어가서 mode=random과 구분이 안 됐다. 재료 미매칭 케이스(위 399줄)와 같은
        # 방식으로, 못 찾았다는 사실을 note에 남긴다(찾기 알고리즘 자체는 안 건드림).
        note += f' ("{anchor}"을 찾을 수 없는 메뉴예요 → 랜덤으로 만듭니다)'
    # 군 메뉴는 모델 사전에 없음 → 일반 생성 후 해당 슬롯에 끼워넣기(post-substitution)
    gun_s = gun_slot.get(anchor) if (anchor and anchor in gun_names) else None
    tok_anchor = anchor if (anchor in name2idx) else None    # 월 메뉴만 토큰 앵커
    b = bounds if bounds is not None else F.meal_bounds(W)
    best = None; best_score = -1
    # 레버들은 대부분 조용히 포기함(lever_phosphorus만 성공여부 반환). 그래서 레버 말을 믿지 않고
    # 최종 영양값을 passes()로 직접 검사한다 → 모든 실패 유형을 잡음. (adjust의 p_ok는 그래서 안 씀)
    for menus in gen_batch(tok_anchor, n=tries, temp=temp):     # 후보 tries개 배치 생성 = resample
        if gun_s is not None:
            menus = list(menus); menus[gun_s] = anchor          # 군 메뉴 끼워넣기
        # 하루 중복 방지: 오늘 이미 나온 메뉴가 이 조합에 있으면 거부.
        # 앵커(유저 재요청)와 밥·김치(끼니마다 반복되는 게 정상인 주식류)는 예외.
        dup_today = any(m in used_today and m != anchor and not _is_staple_menu(m) for m in menus)
        # 끼니 내 재료 겹침 방지: 서로 다른 메뉴가 같은 정체성 재료(주재료)를 쓰면 거부.
        clash = _has_ingredient_clash(menus)
        # 자연나트륨군(어패류·해조류) 3개+ 겹침 방지 — 레버로도 못 잡는 트리플 겹침을 생성단계에서 차단.
        overload = _has_seafood_overload(menus)
        # 고인비율군(가공식품·유제품·견과류) 3개+ 겹침 방지 — 나트륨과 같은 이유·같은 방식.
        p_overload = _has_high_p_overload(menus)
        before, after, inst, _ = F.adjust(menus, b, anchor=anchor)   # 유저메뉴 재료 보존
        # 영양뿐 아니라 '양이 현실적인가'도 봄. 레버가 밥 1g·재료 11배로 만든 후보는 버림
        # (레버 안에서 자르면 레버끼리 싸워 영양이 깨지므로, 후보 선택 단계에서 거른다.)
        unreal = F.unrealistic_reason(inst)
        ok = (passes(after, b) and unreal is None and not dup_today and not clash
              and not overload and not p_overload)
        # before = 레버(양 조절·재료 대체)를 돌리기 전의 원본 레시피 영양값.
        # 화면에서 '재구성 전 / 후'를 나란히 보여주려고 후보에 같이 담아 올린다.
        cand = (menus, inst, after, ok, before)
        if ok:
            return cand, note, b, anchor, _total_na_warning(after)
        score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                     after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
        if unreal is None:
            score += 0.5                    # 동점이면 현실적인 후보를 우선
        if not dup_today:
            score += 0.3                    # 하루 중복 없는 후보 우선
        if not clash:
            score += 0.3                    # 재료 안 겹치는 후보 우선
        if not overload:
            score += 0.3                    # 자연나트륨군 과다 아닌 후보 우선
        if not p_overload:
            score += 0.3                    # 고인비율군 과다 아닌 후보 우선
        if score > best_score:
            best, best_score = cand, score
    warns = []
    best_after = best[2]
    who = f'"{anchor}"' if anchor else '이 조합'
    # 주의: "불가능"이라 단정하지 말 것. 실패 앵커들도 인 상한에 여유가 있는 경우가 대부분이라
    # (밥+국+찬+김치 최소 ~170mg을 빼면 빠듯하지만) 다시 뽑으면 통과하기도 함 → 재생성을 먼저 권함.
    if best_after['P'] >= b['Pmax']:
        warns.append(f'{who}는 인이 높아 이번 조합으로는 인 기준(<{b["Pmax"]:.0f}mg)을 못 맞췄어요 '
                     f'(현재 {best_after["P"]:.0f}mg). 다시 만들어보면 통과할 수도 있고, '
                     f'계속 안 되면 오늘 다른 끼니를 저인으로 맞춰 하루 기준을 지키세요.')
    if best_after['Na_season'] > b['Namax']:        # 최소 조미료(0.3× 하한)에서도 소금 1g 초과
        warns.append(f'기본 레시피 간이 세서 조미료를 최소로 줄여도 첨가염이 소금 1g(≤{b["Namax"]:.0f}mg)을 넘어요 '
                     f'(현재 {best_after["Na_season"]:.0f}mg). 국물은 100ml 미만(종이컵 반 컵 이하)만 드시고 간을 더 싱겁게 드세요.')
    if best_after['K'] >= b['Kmax']:                # 저칼륨 대체재가 동나거나 건조식품이라 교체 불가
        warns.append(f'칼륨이 기준(<{b["Kmax"]:.0f}mg)을 넘어요 (현재 {best_after["K"]:.0f}mg). '
                     f'채소는 잘게 썰어 물에 담갔다 데쳐서 드시고, 국물은 100ml 미만(종이컵 반 컵 이하)만 드세요.')
    if best_after['E'] > b['Ehi']:                  # 밥/기름을 하한(0.4x)까지 줄여도 초과
        warns.append(f'열량이 기준(≤{b["Ehi"]:.0f}kcal)을 넘어요 (현재 {best_after["E"]:.0f}). 밥 양을 더 줄여보세요.')
    elif best_after['E'] < b['Elo']:
        warns.append(f'열량이 기준(≥{b["Elo"]:.0f}kcal)에 모자라요 (현재 {best_after["E"]:.0f}). '
                     f'투석 환자는 열량이 부족하면 근육이 분해되니 밥을 더 드시거나 간식을 더하세요.')
    if best_after['protein'] > b['Phi']:
        warns.append(f'단백질이 기준(≤{b["Phi"]:.0f}g)을 넘어요 (현재 {best_after["protein"]:.0f}g). '
                     f'인도 같이 올라가니 고기·생선 양을 줄이세요.')
    elif best_after['protein'] < b['Plo']:
        warns.append(f'단백질이 기준(≥{b["Plo"]:.0f}g)에 모자라요 (현재 {best_after["protein"]:.0f}g). '
                     f'투석은 단백질 손실이 커서 부족하면 안 돼요 — 고기·생선·달걀을 더 드세요.')
    na_w = _total_na_warning(best_after)
    if na_w:
        warns.append(na_w)
    warn = ' '.join(warns)
    return best, note + f' [{tries}회 완전통과 실패 → 최선 {best_score}/5]' + ('\n  ⚠ ' + warn if warn else ''), b, anchor, warn


def show(cand, note, b):
    menus, inst, after, ok, _before = cand
    # 최종 메뉴(레버가 김치교체/기름추가 반영)
    final_menus = list(dict.fromkeys(i['menu'] for i in inst))
    print(f'  {note}' if note else '', end='')
    print(f"\n  완성 한끼: {' · '.join(final_menus)}")
    print(f"  영양(끼니): 열량 {after['E']:.0f}(목표{b['Elo']:.0f}~{b['Ehi']:.0f}) "
          f"단백 {after['protein']:.0f}({b['Plo']:.0f}~{b['Phi']:.0f}) "
          f"K {after['K']:.0f}(<{b['Kmax']:.0f}) P {after['P']:.0f}(<{b['Pmax']:.0f}) "
          f"Na(조미료) {after['Na_season']:.0f}(<={b['Namax']:.0f}, 총{after['Na']:.0f})")
    print(f"  5영양 통과: {'O 전부충족' if ok else 'X (일부 미달)'}")


def _changes(menus, inst):
    """레버가 원본 대비 뭘 바꿨는지 사람이 읽을 목록 (재료 수정 내역)."""
    orig = F.expand(menus)
    def bym(lst):
        d = defaultdict(dict)
        for i in lst:
            d[i['menu']][i['ing']] = d[i['menu']].get(i['ing'], 0) + i['amt']
        return d
    o, f = bym(orig), bym(inst)
    om = list(dict.fromkeys(i['menu'] for i in orig))
    fm = list(dict.fromkeys(i['menu'] for i in inst))
    d = F.display_menu_name                         # 표시명 정리('돈까스소스'->'돈까스+소스')
    out = []
    added_menus = [m for m in fm if m not in o]
    for m in om:                                    # 메뉴 통째 교체(저염김치·재료교체로 이름 바뀐 것 등)
        if m not in f:
            repl = added_menus.pop(0) if added_menus else None
            out.append(f'{d(m)} → {d(repl)} 교체' if repl else f'{d(m)} 뺌')
    for m in fm:                                    # 메뉴 내 재료 교체/양조정
        if m not in o:
            continue
        removed = [ing for ing in o[m] if ing not in f[m]]
        added = [ing for ing in f[m] if ing not in o[m]]
        for a, bb in zip(removed, added):
            out.append(f'{d(m)}: {a} → {bb}')       # 전체 이름 (미역,마른것 → 미역,생것 처럼 구분됨)
        for ing in f[m]:
            if ing in o[m] and abs(f[m][ing] - o[m][ing]) > max(2, 0.15 * o[m][ing]):
                out.append(f'{d(m)}: {ing} 양 {o[m][ing]:.2f}→{f[m][ing]:.2f}g')
    return out[:10]


def meal_result(menu=None, ingredient=None, weight=60, consumed=None, meals_left=3,
                height=None, sex=None, used_today=None, custom_targets=None):
    """API용: 완성 한끼를 JSON-able dict로.
    consumed = 오늘 이미 먹은 누적 영양(첫 끼면 None), meals_left = 이번 끼 포함 남은 끼니 수.
    둘을 주면 '남은 예산 ÷ 남은 끼니'로 이 끼의 목표를 잡는다(하루를 이어서 만들 때).
    height(cm)+sex를 주면 영양 목표를 '표준체중'(키²×22/21) 기준으로 잡는다(임상 지침).
    height가 없으면 weight를 표준체중으로 간주해 그대로 쓴다.
    used_today = 오늘 이미 나온 메뉴 집합(하루 중복 방지). day_result가 끼니마다 누적해서 넘긴다.
    custom_targets = 의료진 안내값으로 자동 산출값을 부분 override(FOOK_adjust_levers.day_targets
    참고). None이면 기존과 완전히 동일하게 동작한다."""
    if height is not None:
        weight = F.standard_weight(height, sex)
    # consumed가 없어도(첫 끼) 항상 여기서 bounds를 만든다 — make_meal(bounds=None)이 쓰는
    # 내부 기본값(F.meal_bounds(W))과 결과가 같아 동작은 그대로면서, custom_targets를
    # 놓치지 않고 흘려보낼 수 있다(전에는 None을 넘겨 make_meal 내부 기본값에 맡겼었음).
    bounds = F.meal_bounds(weight, consumed, meals_left=meals_left, overrides=custom_targets)
    cand, note, b, anchor, warn = make_meal(menu=menu, ingredient=ingredient, W=weight, bounds=bounds,
                                             used_today=used_today)
    menus, inst, after, ok, before = cand
    # 표시명 정리: '돈까스소스' -> '돈까스+소스' (데이터는 그대로, 보여줄 때만)
    disp = F.display_menu_name
    final_menus = list(dict.fromkeys(disp(i['menu']) for i in inst))
    # 간식은 조리법을 주지 않음(시판 빵·떡은 조리법이 무의미하고, 원본 레시피가 없어 LLM이 지어냄).
    # 단, 조리법이 없어도 '양(g)'은 알려줘야 함(얼마나 먹을지) → {메뉴, 양g}로 반환.
    snack_names = {disp(s[0]) for s in F.SNACK_POOL}
    snack_amt = {}                                   # 간식 메뉴 → 총 양(g)
    for i in inst:
        d0 = disp(i['menu'])
        if d0 in snack_names:
            snack_amt[d0] = snack_amt.get(d0, 0) + i['amt']
    # 재료·간식의 양(g)은 화면에서 소수점 2자리까지 그대로 보여주므로 2자리로 내려준다.
    snacks = [{'menu': m, 'amount_g': round(snack_amt.get(m, 0), 2)}
              for m in final_menus if m in snack_names]
    # 메뉴별 영양 (레시피 상세 화면에서 '이 음식 한 그릇'의 5영양소를 보여주기 위함).
    # 끼니 전체 합계와 달리, 그 메뉴에 속한 재료 인스턴스만 모아 다시 계산한다.
    dish_inst = {}
    for i in inst:
        dish_inst.setdefault(disp(i['menu']), []).append(i)
    dish_nutrition = {}
    for d0, items in dish_inst.items():
        t = F.totals(items)
        dish_nutrition[d0] = {'energy': round(t['E']), 'protein': round(t['protein'], 1),
                              'potassium': round(t['K']), 'phosphorus': round(t['P']),
                              'sodium': round(t['Na_season']), 'sodium_total': round(t['Na'])}
    dish_ings = {}                                   # 메뉴별 최종 재료 (레시피 편집용, 간식 제외)
    recipe_src = {}                                  # 표시명 → 조리법을 찾을 원래 데이터명
    for i in inst:
        d = disp(i['menu'])
        if d in snack_names:
            continue
        dish_ings.setdefault(d, []).append([i['ing'], round(i['amt'], 2)])
        src = i.get('orig_menu') or i['menu']        # 재료교체로 이름이 바뀌었으면 교체 전 이름
        if src != d:
            recipe_src[d] = src                      # 예: 고등어볶음→오징어볶음, 돈까스+소스→돈까스소스
    return {
        'mode': 'menu' if menu else ('ingredient' if ingredient else 'random'),
        'anchor': anchor,
        'meal': final_menus,
        'raw_menus': menus,        # 생성 직후(레버 재료교체 전) 슬롯별 메뉴명 — 하루 중복 체크용

        'nutrition': {'energy': round(after['E']), 'protein': round(after['protein'], 1),
                      'potassium': round(after['K']), 'phosphorus': round(after['P']),
                      'sodium': round(after['Na_season']),        # 조미료(첨가염)만 = 판정 대상
                      'sodium_total': round(after['Na'])},        # 총 나트륨 (자연재료 포함, 참고)
        # 레버가 손대기 전(원본 레시피 그대로)의 영양값. 화면의 '재구성 전 | 후' 비교용.
        'nutrition_before': {'energy': round(before['E']), 'protein': round(before['protein'], 1),
                             'potassium': round(before['K']), 'phosphorus': round(before['P']),
                             'sodium': round(before['Na_season']),
                             'sodium_total': round(before['Na'])},
        'targets': {'energy': [round(b['Elo']), round(b['Ehi'])],
                    'protein': [round(b['Plo'], 1), round(b['Phi'], 1)],
                    'potassium': round(b['Kmax']), 'phosphorus': round(b['Pmax']),
                    'sodium': round(b['Namax']),                    # 조미료(첨가염) 상한 — 기존과 동일
                    'sodium_total_target': round(b['Na_total_target'])},  # 총나트륨 목표(신규, custom 반영)
        'passed': bool(ok),
        'note': note.strip(),      # 안내문(재료→메뉴 선택 결과, 잡곡밥→흰쌀밥 대체 사유 등)
        'warning': warn,
        'changes': _changes(menus, inst),
        'dish_ingredients': dish_ings,
        'dish_nutrition': dish_nutrition,   # 메뉴별 5영양소 (레시피 상세 화면용)
        'snacks': snacks,          # 조리법 없이 [{메뉴, 양g}]로 제시하는 간식 (양은 고지)
        'recipe_source': recipe_src,   # 재료가 바뀌어 이름이 달라진 메뉴의 원본명(조리법 조회용)
        # 이 끼의 실제 섭취량. 다음 끼를 만들 때 consumed에 누적해서 넘기면 남은 예산이 반영된다.
        'intake': {k: round(after[k], 1) for k in ('E', 'protein', 'K', 'P', 'Na', 'Na_season')},
    }


MEAL_LABELS = ('아침', '점심', '저녁')


def day_targets_only(weight=60, height=None, sex=None, custom_targets=None):
    """메뉴 생성 없이 하루 전체 영양 기준만 계산해서 반환한다.
    day_result()의 day_targets와 계산식은 같지만, 후보 생성(make_meal)을 전혀 돌리지 않아
    훨씬 가볍다 — 식사 기록 화면에서 '오늘 얼마나 채웠는지' 진행률을 보여줄 때 쓴다.
    custom_targets: 의료진 안내값 부분 override(F.day_targets 참고). None이면 기존과 동일."""
    if height is not None:
        weight = F.standard_weight(height, sex)
    d = F.day_targets(weight, custom_targets)
    return {'energy': [round(d['Elo']), round(d['Ehi'])],
            'protein': [round(d['Plo'], 1), round(d['Phi'], 1)],
            'potassium': round(d['Kmax']),
            'phosphorus': round(d['Pmax']),
            'sodium': round(d['Namax']),                    # 조미료(첨가염) 상한 — 기존과 동일
            'sodium_total_target': round(d['Na_total_max'])}  # 총나트륨 하루 목표(신규, custom 반영)


def day_result(weight=60, menus=None, ingredients=None, height=None, sex=None, custom_targets=None):
    """하루 세 끼를 '남은 예산' 방식으로 이어서 생성.
    앞 끼에서 먹은 양을 빼고 남은 끼니로 나눠 다음 끼 목표를 잡으므로, 한 끼씩 따로 만들 때와 달리
    하루 총량이 기준 안에 들어온다. menus/ingredients = 끼니별 지정(길이 3, None이면 랜덤).
    height(cm)+sex를 주면 표준체중 기준으로 목표를 잡는다.
    custom_targets: 의료진 안내값 부분 override — 끼니마다 그대로 이어서 넘긴다."""
    if height is not None:
        weight = F.standard_weight(height, sex)
    menus = list(menus) if menus else [None] * 3
    ingredients = list(ingredients) if ingredients else [None] * 3
    consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
    used_today = set()   # 하루 중복 방지: 끼니마다 나온 메뉴를 누적
    out = []
    for mi in range(3):
        r = meal_result(menu=menus[mi], ingredient=ingredients[mi], weight=weight,
                        consumed=consumed, meals_left=3 - mi, used_today=used_today,
                        custom_targets=custom_targets)
        for k in consumed:
            consumed[k] = round(consumed[k] + r['intake'][k], 4)   # 부동소수점 누적오차 방지(방법2)
        used_today.update(r['raw_menus'])
        r['label'] = MEAL_LABELS[mi]
        out.append(r)
    dt = F.day_targets(weight, custom_targets)
    EPS = 1e-6   # 부동소수점 누적오차 허용오차(방법1) — 1799.9999999998 같은 값이 미달로 오판되는 것 방지
    # 하루 통과 판정은 기본적으로 첨가염(Na_season) 기준 그대로다 — custom sodium이 없는 사용자는
    # 이 줄이 예전과 완전히 동일하다. custom sodium을 설정한 사용자만(dt['custom_sodium_active'])
    # 총나트륨(Na <= Na_total_max)도 추가로 만족해야 day_passed=True가 된다(passes()와 같은 원칙).
    day_ok = (dt['Elo'] - EPS <= consumed['E'] <= dt['Ehi'] + EPS
              and dt['Plo'] - EPS <= consumed['protein'] <= dt['Phi'] + EPS
              and consumed['K'] < dt['Kmax'] + EPS and consumed['P'] < dt['Pmax'] + EPS
              and consumed['Na_season'] <= dt['Namax'] + EPS)
    if dt.get('custom_sodium_active'):
        day_ok = day_ok and consumed['Na'] <= dt['Na_total_max'] + EPS
    return {
        'meals': out,
        'day_nutrition': {'energy': round(consumed['E']), 'protein': round(consumed['protein'], 1),
                          'potassium': round(consumed['K']), 'phosphorus': round(consumed['P']),
                          'sodium': round(consumed['Na_season']), 'sodium_total': round(consumed['Na'])},
        'day_targets': {'energy': [round(dt['Elo']), round(dt['Ehi'])],
                        'protein': [round(dt['Plo'], 1), round(dt['Phi'], 1)],
                        'potassium': round(dt['Kmax']), 'phosphorus': round(dt['Pmax']),
                        'sodium': round(dt['Namax']),
                        'sodium_total_target': round(dt['Na_total_max'])},
        'day_passed': bool(day_ok),
    }


if __name__ == '__main__':
    import json
    np.random.seed(0)
    for label, kw in [('메뉴 "제육고추장불고기"(월)', dict(menu='제육고추장불고기')),
                      ('군 메뉴 "옥돔구이"', dict(menu='옥돔구이')),
                      ('재료 "고등어"', dict(ingredient='고등어')),
                      ('랜덤', dict())]:
        print('=' * 60, f'\n[{label}]')
        print(json.dumps(meal_result(**kw), ensure_ascii=False, indent=2))
