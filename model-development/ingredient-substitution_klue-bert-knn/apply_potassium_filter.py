"""
대체 쌍에 칼륨 필터 적용 (임상영양사 피드백 반영, 2026-07-21 개정)
투석 환자용: 1회분(식품교환단위) 기준 칼륨 등급으로 '고칼륨' 재료를 '저/중'으로 교체.

등급 기준(1회분 대표중량당 칼륨):
    저칼륨 < 100mg,  중칼륨 100~200mg,  고칼륨 > 200mg
1회분 대표중량(SERVING_G): 채소 70g / 육류 40g / 생선(어패) 50g 등 식품교환표 기준.
유효 대체 조건:
    (1) 원재료 = 고칼륨   (2) 대체재 = 저 또는 중   (3) 같은 식품군(호환군)
    (4) 원재료 대비 칼륨 20% 이상 감소 (동일 무게=100g당 기준)
"""
import json

with open('foodbert_embeddings/data/substitute_pairs_foodbert_text.json', encoding='utf-8') as f:
    pairs = json.load(f)

with open('nutrition_db.json', encoding='utf-8') as f:
    nutrition = json.load(f)

# 식품군이 달라도 대체 허용할 묶음 (투석 환자 관점에서 실제로 대체 가능한 것)
# ※ '두류'는 임상 식품교환표에 따라 두 가상군으로 분기(effective_group 참조):
#   - 두류(단백질): 두부·순두부·유부·대두·콩가루 등 → 어육류군(단백질 8g/교환). 고기·생선 감량 시 단백질 보존 안전지대.
#   - 두류(전분):   녹두·팥·강낭콩·묵류 → 탄수화물(전분질콩). 백미·전분과 호환.
COMPATIBLE_GROUPS = [
    {'육류 및 그 제품', '어패류 및 그 제품', '난류', '조리가공식품류', '두류(단백질)'},  # 단백질류
    {'채소류', '버섯류', '해조류', '절임류'},                          # 채소류
    {'곡류 및 그 제품', '감자류 및 전분류', '두류(전분)', '빵 및 과자류'},  # 탄수화물류
    {'조미료류'},                                            # 향신료/장류/소금 등 - 기름/당류와는 섞지 않음
    {'유지류'},
    {'당류'},
    {'과일류'},
    {'우유 및 그 제품'},
    {'견과류 및 종실류'},
]

# 전분질 콩(탄수화물 예외군) — 나머지 두류는 단백질군으로 간주.
# 매칭은 재료 key + 식약청 식품명 문자열 양쪽에서 부분일치.
STARCHY_BEAN_WORDS = ('녹두', '팥', '강낭콩', '완두', '동부', '병아리콩', '렌틸', '묵', '빈대떡')

# 1회분 대표중량(g) = 식품교환단위 (임상영양사 제공 표, 2026-07-21).
# 대표값이 범위인 경우 중앙/대표치를 채택. 등급 분류 및 1회분 칼륨 계산에만 사용.
SERVING_G = {
    # 어육류군 (단백질 8g/교환)
    '육류 및 그 제품': 40,       # 고기 1교환
    '어패류 및 그 제품': 50,     # 생선 1교환 (흰살 우선)
    '난류': 55,                  # 계란 1개 (표준 교환단위)
    '두류(단백질)': 80,          # 두부 1교환 (식물성 단백질·중인군). 대두·콩가루도 이 군(고칼륨→두부 치환)
    '두류(전분)': 70,            # 녹두·팥 등 전분질콩 → 곡류 준함
    '조리가공식품류': 50,        # 가공 어육류 준함
    # 채소군 (1교환 70g)
    '채소류': 70,
    '버섯류': 70,
    '해조류': 70,
    '절임류': 70,
    # 곡류·서류 (탄수화물 23g/교환)
    '곡류 및 그 제품': 70,       # 밥 1교환 (잡곡·찹쌀→백미 스케일)
    '감자류 및 전분류': 100,     # 감자 140/고구마 70 → 대표 100 (★고칼륨군, 침출 필요)
    '빵 및 과자류': 70,
    # 과일 (탄수화물 12g/교환)
    '과일류': 120,               # 100~150 대표 (★고칼륨군, 1일 1단위 이하)
    # 유제품 (★고인·고칼륨)
    '우유 및 그 제품': 200,      # 우유 1교환 200ml
    '음료류': 200,
    # 지방·견과 (지방 5g/교환)
    '유지류': 5,                 # 기름 1교환
    '견과류 및 종실류': 10,      # 8~10 대표 (★고인·고칼륨)
    # 소량 사용군 (분류만 — 같은 군끼리만 매칭되어 대체는 거의 트리거 안 됨)
    '조미료류': 10,
    '당류': 10,
    '차류': 5,
    '주류': 100,
    '기타': 50,
}
DEFAULT_SERVING = 50   # 표에 없는 군 폴백

# 대체 제외 원재료 (정체성 재료 = 대체보다 '양 조절'이 임상 정답. 해조류와 같은 논리).
# 녹두: 키워드 '녹두'가 원물+빈대떡가루를 한 토큰으로 뭉쳐 제형 판정이 어긋남 + 녹두전의 정체성 재료.
# 미역: 고칼륨이지만 정서적/식문화 예외(2026-07-22, 미역국=생일음식). 대체 대신 양 조절로 대응.
EXCLUDE_ORIGINAL_KEYS = {'녹두', '미역'}

# 원재료 자체는 살리되 특정 대체 '페어'만 제외 (조리법 불일치, 2026-07-22).
# 브로콜리→오이: 브로콜리는 데쳐서(숙회) 초고추장에 찍어먹는 조리법인데, 오이는 데쳐 먹지 않는
# 채소(생으로/무침만) — '오이숙회'는 존재하지 않는 요리. 브로콜리→양배추(같은 십자화과, 숙회 가능)는 유지.
PAIR_EXCLUDE = {('브로콜리', '오이')}

# 등급 임계치 (1회분당 칼륨 mg)
K_LOW_MAX = 100      # < 100 : 저칼륨
K_MID_MAX = 200      # 100~200 : 중칼륨,  > 200 : 고칼륨
REDUCTION_MIN = 20.0  # 원재료 대비 최소 감소율(%) — '20% 이상 변화'


def get_group(ingredient):
    """효과적 식품군. '두류'는 임상 기준으로 두류(단백질)/두류(전분)으로 분기."""
    if ingredient not in nutrition:
        return None
    g = nutrition[ingredient].get('식품군')
    if g == '두류':
        name = ingredient + ' ' + str(nutrition[ingredient].get('식품명', ''))
        if any(w in name for w in STARCHY_BEAN_WORDS):
            return '두류(전분)'      # 녹두·팥·묵류 → 탄수화물군
        return '두류(단백질)'         # 두부·대두·콩가루 → 어육류군
    return g


def same_category(g1, g2):
    if g1 is None or g2 is None:
        return False
    if g1 == g2:
        return True
    for group in COMPATIBLE_GROUPS:
        if g1 in group and g2 in group:
            return True
    return False


def get_potassium(ingredient):
    """100g당 칼륨(mg)."""
    if ingredient not in nutrition:
        return None
    val = nutrition[ingredient]['칼륨']
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('(') and val.endswith(')'):
            val = val[1:-1]  # 식약청 DB의 '(240)' = 추정치 표기, 실측값으로 취급
    try:
        return float(val)
    except (TypeError, ValueError):
        return None  # 'Tr'(미량), '-'(미측정) 등 실제 값이 없는 경우


def serving_g(group):
    return SERVING_G.get(group, DEFAULT_SERVING)


def potassium_per_serving(k100, group):
    """1회분(식품교환단위) 칼륨(mg)."""
    return k100 * serving_g(group) / 100.0


def grade(k_serving):
    """1회분 칼륨으로 저/중/고 등급."""
    if k_serving < K_LOW_MAX:
        return '저'
    if k_serving <= K_MID_MAX:
        return '중'
    return '고'


# ── 도메인 카테고리 가드 (임베딩 KNN 후단 post-filter, 2026-07-21) ─────────────
# 임베딩은 '조리맥락 유사도'로 후보를 뽑아 요리역할이 안 맞는 쌍(미역→콩나물, 녹두→전분)이 섞임.
# 아래 3가드로 도메인 규칙을 얹어 걸러낸다.
POWDER_STARCH_WORDS = ('가루', '전분', '분말', '녹말', '당면')
# 초고인 가공식품(인산염 첨가) — 자연식품의 대체재로 부적합
PROCESSED_HIGH_P_WORDS = ('맛살', '어묵', '햄', '소시지', '베이컨', '미트볼', '너겟',
                          '완자', '스팸', '훈제', '젓', '진미채', '인스턴트')
# ※ 매칭은 재료 key 기준. 식약청 식품명(full)엔 '녹두 빈대떡,가루'처럼 원물이 '가루'로
#    오염된 경우가 있어 key로만 판정한다.


def is_seaweed(group):
    return group == '해조류'


def is_seasoning(group):
    return group == '조미료류'


def is_powder_or_starch(ingredient, group):
    if any(w in ingredient for w in POWDER_STARCH_WORDS):
        return True
    return group == '감자류 및 전분류'     # 전분·당면·녹말 등


def is_whole_grain_or_legume(ingredient, group):
    """원물 곡류/두류 (이미 가루·전분형이면 제외)."""
    if group in ('곡류 및 그 제품', '두류(전분)', '두류(단백질)'):
        return not any(w in ingredient for w in POWDER_STARCH_WORDS)
    return False


def is_processed_high_p(ingredient, group):
    if group in ('조리가공식품류', '절임류'):
        return True
    return any(w in ingredient for w in PROCESSED_HIGH_P_WORDS)


def passes_domain_guards(original, g_orig, substitute, g_sub, stats):
    """3가지 도메인 가드. 통과 시 True, 차단 시 stats 갱신 후 False."""
    # 1. 해조류 락: 미역·다시마·김은 오직 해조류끼리만
    if is_seaweed(g_orig) and not is_seaweed(g_sub):
        stats['seaweed_lock'] += 1
        return False
    # 2. 제형 락: 원물 곡류/두류 → 가루·전분 차단 (조리 제형 불일치)
    if is_whole_grain_or_legume(original, g_orig) and is_powder_or_starch(substitute, g_sub):
        stats['form_lock'] += 1
        return False
    # 3. 가공/조미료 락: 초고인 가공식품·조미료를 대체재로 쓰지 않음
    if is_processed_high_p(substitute, g_sub) or is_seasoning(g_sub):
        stats['processed_lock'] += 1
        return False
    return True


results = []
stats = {'total': len(pairs), 'both_have_nutrition': 0, 'kept': 0,
         'no_nutrition': 0, 'different_category': 0,
         'orig_not_high': 0, 'sub_still_high': 0, 'below_20pct': 0,
         'seaweed_lock': 0, 'form_lock': 0, 'processed_lock': 0}

for pair in pairs:
    original, substitute = pair[0], pair[1]
    if original in EXCLUDE_ORIGINAL_KEYS:      # 정체성 재료 → 대체 안 함(양 조절)
        continue
    if (original, substitute) in PAIR_EXCLUDE:  # 조리법 불일치 페어만 콕 집어 제외
        continue
    k_orig = get_potassium(original)
    k_sub = get_potassium(substitute)

    if k_orig is None or k_sub is None:
        stats['no_nutrition'] += 1
        continue
    stats['both_have_nutrition'] += 1

    g_orig = get_group(original)
    g_sub = get_group(substitute)
    if not same_category(g_orig, g_sub):
        stats['different_category'] += 1
        continue

    ks_orig = potassium_per_serving(k_orig, g_orig)   # 1회분 칼륨
    ks_sub = potassium_per_serving(k_sub, g_sub)
    grade_orig = grade(ks_orig)
    grade_sub = grade(ks_sub)

    # (1) 원재료는 반드시 고칼륨
    if grade_orig != '고':
        stats['orig_not_high'] += 1
        continue
    # (2) 대체재는 저 또는 중 (= 고칼륨이면 탈락)
    if grade_sub == '고':
        stats['sub_still_high'] += 1
        continue
    # (4) 원재료 대비 20% 이상 감소 (동일 무게=100g당 기준)
    reduction_rate = (k_orig - k_sub) / k_orig * 100 if k_orig > 0 else 0
    if reduction_rate < REDUCTION_MIN:
        stats['below_20pct'] += 1
        continue

    # (5) 도메인 카테고리 가드 (해조류 락 / 제형 락 / 가공·조미료 락)
    if not passes_domain_guards(original, g_orig, substitute, g_sub, stats):
        continue

    stats['kept'] += 1
    results.append({
        'original': original,
        'substitute': substitute,
        'food_group': g_orig,
        'substitute_group': g_sub,
        'original_potassium': k_orig,          # 100g당
        'substitute_potassium': k_sub,         # 100g당
        'serving_g': serving_g(g_orig),
        'substitute_serving_g': serving_g(g_sub),
        'original_k_per_serving': round(ks_orig, 1),
        'substitute_k_per_serving': round(ks_sub, 1),
        'original_grade': grade_orig,
        'substitute_grade': grade_sub,
        'potassium_reduction': round(k_orig - k_sub, 1),          # 100g당 감소량
        'per_serving_reduction': round(ks_orig - ks_sub, 1),      # 1회분 감소량
        'reduction_rate': round(reduction_rate, 1),
    })

# 1회분 감소량 기준 정렬 (임상 체감 효과 순)
results.sort(key=lambda x: x['per_serving_reduction'], reverse=True)

print(f"전체 대체 쌍: {stats['total']}개")
print(f"영양정보 있는 쌍: {stats['both_have_nutrition']}개")
print(f"  ├ 카테고리 불일치 제거: {stats['different_category']}개")
print(f"  ├ 원재료가 고칼륨 아님(제외): {stats['orig_not_high']}개")
print(f"  ├ 대체재도 고칼륨(제외): {stats['sub_still_high']}개")
print(f"  ├ 20% 미만 감소(제외): {stats['below_20pct']}개")
print(f"  ├ 해조류 락(제외): {stats['seaweed_lock']}개")
print(f"  ├ 제형 락 원물→가루(제외): {stats['form_lock']}개")
print(f"  └ 가공·조미료 락(제외): {stats['processed_lock']}개")
print(f"영양정보 없음: {stats['no_nutrition']}개")
print(f"==> 최종 채택(고→저/중, 20%↓): {stats['kept']}개")
print()
print("1회분 칼륨 감소 TOP 20:")
for r in results[:20]:
    print(f"  [{r['food_group']}] {r['original']}({r['original_grade']},{r['original_k_per_serving']}mg/{r['serving_g']}g)"
          f" → {r['substitute']}({r['substitute_grade']},{r['substitute_k_per_serving']}mg/{r['substitute_serving_g']}g)"
          f" [1회분 -{r['per_serving_reduction']}mg, 100g당 -{r['reduction_rate']}%]")

# 재료별로 묶기 (원재료 기준)
by_original = {}
for r in results:
    by_original.setdefault(r['original'], []).append(r)

print(f"\n대체 가능한 고칼륨 원재료 수: {len(by_original)}개")

# 저장
with open('substitute_pairs_low_potassium.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open('substitute_by_ingredient.json', 'w', encoding='utf-8') as f:
    json.dump(by_original, f, ensure_ascii=False, indent=2)

print("\n저장 완료: substitute_pairs_low_potassium.json, substitute_by_ingredient.json")
