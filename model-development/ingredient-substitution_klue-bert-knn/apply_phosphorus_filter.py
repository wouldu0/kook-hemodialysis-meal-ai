"""
인(P) 전용 대체 사전 생성 — 임베딩(1,604쌍)에서 인 로직으로 새로 추출 (2026-07-22).
※ 칼륨 사전 재활용 금지: 칼륨은 '고칼륨 채소→저칼륨', 인은 타깃이 다름(단백질원의 P/단백질 비율).

핵심 지표 = 인/단백질 비율(mg P per g 단백질). 단백질을 지키면서 인만 효율적으로 줄이는 대체.
등급(임상영양사 확정): 저<12 / 중12~16 / 고>16 mg/g.
식물성 인 흡수율 보정: 두류(단백질=대두·두부계열)의 인은 ×0.7 (흡수율 낮음).

유효 대체 조건:
    (1) 원재료·대체재 모두 단백질 식품(단백질 ≥ 3g/100g)
    (2) 같은 단백질 호환군 (역할 보존)
    (3) 원재료 = 고비율(>16),  대체재 = 저·중(≤16)
    (4) 단백질 보존: 대체재 단백질 ≥ 원재료의 75%
    (5) 가공식품(인산염 첨가)·조미료 대체재 금지  ← 인에서 특히 중요
데이터: 인=nutrition_db(인), 단백질=식약청10.4 조인(식품명 기준).
"""
import json, openpyxl

with open('foodbert_embeddings/data/substitute_pairs_foodbert_text.json', encoding='utf-8') as f:
    pairs = json.load(f)
with open('nutrition_db.json', encoding='utf-8') as f:
    nutrition = json.load(f)

# 단백질 조인 (식약청 10.4, 식품명 기준)
_wb = openpyxl.load_workbook('../../식약청_영양성분10.4(수정).xlsx', read_only=True)
_ws = _wb.active
def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
_saf = {}
for _row in _ws.iter_rows(min_row=4, values_only=True):
    _nm = _row[3]
    if _nm and str(_nm).strip():
        _saf[str(_nm).strip()] = _num(_row[7])   # 단백질 g/100g
_wb.close()

# 인 대체는 '단백질/어육류 전용' (2026-07-22 확정): 곡류·서류는 expand() JAPGOK 정규화가
# 이미 백미치환으로 선처리하므로 인 대체사전에서 제외. 원재료·대체재 모두 이 클러스터 안이어야 함.
PROTEIN_CLUSTER = {'육류 및 그 제품', '어패류 및 그 제품', '난류', '조리가공식품류', '두류(단백질)'}

# 식품군 호환 클러스터 (두류 분기 반영)
COMPATIBLE_GROUPS = [
    PROTEIN_CLUSTER,                                          # 단백질류(인 대체 대상)
    {'채소류', '버섯류', '해조류', '절임류'},
    {'곡류 및 그 제품', '감자류 및 전분류', '두류(전분)', '빵 및 과자류'},
    {'조미료류'}, {'유지류'}, {'당류'}, {'과일류'}, {'우유 및 그 제품'}, {'견과류 및 종실류'},
]
STARCHY_BEAN_WORDS = ('녹두', '팥', '강낭콩', '완두', '동부', '병아리콩', '렌틸', '묵', '빈대떡')

PLANT_PROTEIN_GROUP = '두류(단백질)'
PLANT_P_FACTOR = 0.7      # 식물성 유기인 흡수 보정(↓)
PROCESSED_P_FACTOR = 1.5  # 가공육 무기인(인산염 첨가) 흡수 상향 가중(↑). 식물성 0.7의 대칭.
                          # 가공육은 단백질이 높아 P/단백질 비율이 낮게 나오지만, 실제 무기인 흡수율~100%.
PROTEIN_MIN = 3.0         # 단백질 식품 판정(g/100g)
RATIO_LOW_MAX = 12.0      # 저 < 12
RATIO_MID_MAX = 16.0      # 중 12~16, 고 > 16
PROTEIN_KEEP = 0.75       # 대체재 단백질 ≥ 원재료 75%

PROCESSED_HIGH_P_WORDS = ('맛살', '어묵', '햄', '소시지', '베이컨', '미트볼', '너겟',
                          '완자', '스팸', '훈제', '젓', '진미채', '인스턴트')


def get_group(ing):
    if ing not in nutrition:
        return None
    g = nutrition[ing].get('식품군')
    if g == '두류':
        name = ing + ' ' + str(nutrition[ing].get('식품명', ''))
        return '두류(전분)' if any(w in name for w in STARCHY_BEAN_WORDS) else PLANT_PROTEIN_GROUP
    return g


def same_category(g1, g2):
    if g1 is None or g2 is None:
        return False
    if g1 == g2:
        return True
    return any(g1 in grp and g2 in grp for grp in COMPATIBLE_GROUPS)


def get_P(ing):
    """100g당 인(mg)."""
    if ing not in nutrition:
        return None
    val = nutrition[ing].get('인')
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('(') and val.endswith(')'):
            val = val[1:-1]
    return _num(val)


def get_protein(ing):
    if ing not in nutrition:
        return None
    return _saf.get(str(nutrition[ing].get('식품명', '')).strip())


def is_processed_high_p(ing, group):
    if group in ('조리가공식품류', '절임류'):
        return True
    return any(w in ing for w in PROCESSED_HIGH_P_WORDS)


def eff_P(P, group, ing):
    """흡수 보정 인. 가공육 ×1.5(무기인), 식물성 단백질군 ×0.7(유기인), 그 외 원값."""
    if P is None:
        return None
    if is_processed_high_p(ing, group):
        return P * PROCESSED_P_FACTOR
    if group == PLANT_PROTEIN_GROUP:
        return P * PLANT_P_FACTOR
    return P


def ratio_grade(r):
    if r < RATIO_LOW_MAX:
        return '저'
    if r <= RATIO_MID_MAX:
        return '중'
    return '고'


results = []
stats = {'total': len(pairs), 'both_protein_food': 0, 'kept': 0,
         'no_data': 0, 'not_protein_food': 0, 'not_protein_cluster': 0,
         'different_category': 0, 'orig_not_high': 0, 'sub_still_high': 0,
         'protein_lost': 0, 'no_reduction': 0, 'processed_lock': 0}

for pair in pairs:
    o, s = pair[0], pair[1]
    Po, Ps = get_P(o), get_P(s)
    pro_o, pro_s = get_protein(o), get_protein(s)
    if None in (Po, Ps, pro_o, pro_s):
        stats['no_data'] += 1
        continue
    # (1) 둘 다 단백질 식품
    if pro_o < PROTEIN_MIN or pro_s < PROTEIN_MIN:
        stats['not_protein_food'] += 1
        continue
    stats['both_protein_food'] += 1

    go, gs = get_group(o), get_group(s)
    # (2a) 단백질/어육류 전용: 곡류·서류 등은 제외(JAPGOK 정규화가 선처리)
    if go not in PROTEIN_CLUSTER:
        stats['not_protein_cluster'] += 1
        continue
    # (2b) 같은 호환군
    if not same_category(go, gs):
        stats['different_category'] += 1
        continue
    # (5) 가공식품·조미료 대체재 금지
    if is_processed_high_p(s, gs) or gs == '조미료류':
        stats['processed_lock'] += 1
        continue

    effPo, effPs = eff_P(Po, go, o), eff_P(Ps, gs, s)
    ratio_o, ratio_s = effPo / pro_o, effPs / pro_s
    grade_o, grade_s = ratio_grade(ratio_o), ratio_grade(ratio_s)

    # (3) 원재료 고비율 → 대체재 저·중
    if grade_o != '고':
        stats['orig_not_high'] += 1
        continue
    if grade_s == '고':
        stats['sub_still_high'] += 1
        continue
    # (4) 단백질 보존
    if pro_s < pro_o * PROTEIN_KEEP:
        stats['protein_lost'] += 1
        continue
    # 비율이 실제로 낮아져야 함
    if ratio_s >= ratio_o:
        stats['no_reduction'] += 1
        continue

    stats['kept'] += 1
    results.append({
        'original': o, 'substitute': s,
        'food_group': go, 'substitute_group': gs,
        'original_P': Po, 'substitute_P': Ps,
        'original_protein': pro_o, 'substitute_protein': pro_s,
        'original_ratio': round(ratio_o, 1), 'substitute_ratio': round(ratio_s, 1),
        'original_grade': grade_o, 'substitute_grade': grade_s,
        'ratio_reduction': round(ratio_o - ratio_s, 1),
        'protein_kept_pct': round(pro_s / pro_o * 100, 0),
    })

results.sort(key=lambda x: x['ratio_reduction'], reverse=True)

print(f"전체 임베딩 쌍: {stats['total']}")
print(f"단백질 식품 쌍(둘 다 단백질≥3g): {stats['both_protein_food']}")
print(f"  ├ 데이터 없음: {stats['no_data']}   ├ 단백질식품 아님(제외): {stats['not_protein_food']}")
print(f"  ├ 비단백질군(곡류·서류 등 제외): {stats['not_protein_cluster']}")
print(f"  ├ 카테고리 불일치: {stats['different_category']}   ├ 가공·조미료 락: {stats['processed_lock']}")
print(f"  ├ 원재료 고비율 아님: {stats['orig_not_high']}   ├ 대체재도 고비율: {stats['sub_still_high']}")
print(f"  ├ 단백질 손실(<75%): {stats['protein_lost']}   └ 비율감소 없음: {stats['no_reduction']}")
print(f"==> 최종 채택(고→저/중, 단백질보존): {stats['kept']}")
print()
print("인/단백질 비율 감소 TOP 20:")
for r in results[:20]:
    print(f"  [{r['food_group']}] {r['original']}({r['original_grade']},비율{r['original_ratio']}) "
          f"→ {r['substitute']}({r['substitute_grade']},비율{r['substitute_ratio']}) "
          f"[Δ비율 -{r['ratio_reduction']}, 단백질 {r['protein_kept_pct']:.0f}% 유지]")

by_original = {}
for r in results:
    by_original.setdefault(r['original'], []).append(r)
print(f"\n대체 가능한 고인비율 원재료: {len(by_original)}종")

with open('substitute_pairs_low_phosphorus.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
with open('substitute_by_ingredient_P.json', 'w', encoding='utf-8') as f:
    json.dump(by_original, f, ensure_ascii=False, indent=2)

# 검수용 CSV
import csv
with open('FOOK_인대체사전.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['원재료', '원재료등급', '원재료_P/단백질', '대체재', '대체재등급', '대체재_P/단백질',
                '식품군', 'Δ비율', '단백질유지%', '원재료_인', '대체재_인',
                '원재료_단백질', '대체재_단백질'])
    for r in results:
        w.writerow([r['original'], r['original_grade'], r['original_ratio'],
                    r['substitute'], r['substitute_grade'], r['substitute_ratio'],
                    r['food_group'], r['ratio_reduction'], r['protein_kept_pct'],
                    r['original_P'], r['substitute_P'], r['original_protein'], r['substitute_protein']])

print("\n저장: substitute_by_ingredient_P.json, substitute_pairs_low_phosphorus.json, FOOK_인대체사전.csv")
