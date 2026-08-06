# -*- coding: utf-8 -*-
"""
validation_scenarios_FOOK.py — STEP1~2 reward/beta ablation용 별도 validation 시나리오
(기존 120개 최종평가 CSV와 완전히 분리, 겹치지 않음). STEP3에서 본격적인 train/val 세트를
새로 만들 때까지 이 소규모 세트로 ablation 비교에 쓴다.
"""
import os, sys, csv
sys.path.insert(0, r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code')
sys.path.insert(0, r'E:\final')

FRESH_MAINS = ['가자미구이', '감자조림', '건파래볶음', '고구마맛탕', '궁중떡볶이', '낙지채소볶음',
               '날치알달걀찜', '느타리버섯볶음', '단호박찜', '감자채당근볶음']
FRESH_ING_KEYWORDS = ['미역', '깻잎', '느타리', '단호박', '메추리알', '팽이버섯', '숙주', '연근',
                       '우엉', '브로콜리']
# 120csv와 겹치지 않는 별도 키 목록(145~190cm 범위지만 다른 구체값)
HEIGHTS = [147, 150, 153, 156, 158, 160, 162, 164, 166, 168, 170, 173, 176, 178, 180, 182, 184, 186, 188]
SEXES = ['남', '여']


def build_validation_scenarios():
    import app_core_FOOK as core
    scenarios = []
    sid = 0
    ing_verified = []
    for kw in FRESH_ING_KEYWORDS:
        cand, note, b, anchor, warn = core.make_meal(ingredient=kw, W=60)
        if anchor is not None:
            ing_verified.append((kw, str(anchor)))
    seen = set(); ing_dedup = []
    for kw, m in ing_verified:
        if m not in seen:
            ing_dedup.append((kw, m)); seen.add(m)

    # 20개: 랜덤모드
    for i in range(20):
        h = HEIGHTS[i % len(HEIGHTS)]
        sex = SEXES[i % 2]
        w = round(core.F.standard_weight(h, sex), 1)
        sid += 1
        scenarios.append({'sid': f'V{sid:03d}', 'height': h, 'sex': sex, 'weight': w,
                           'mode': 'random', 'anchor_or_ing_input': None})
    # 10개: 메뉴지정(신규 앵커)
    for i in range(10):
        h = HEIGHTS[(i + 5) % len(HEIGHTS)]
        sex = SEXES[(i + 1) % 2]
        w = round(core.F.standard_weight(h, sex), 1)
        sid += 1
        scenarios.append({'sid': f'V{sid:03d}', 'height': h, 'sex': sex, 'weight': w,
                           'mode': 'menu', 'anchor_or_ing_input': FRESH_MAINS[i % len(FRESH_MAINS)]})
    # 10개: 재료지정(신규 재료)
    for i in range(10):
        h = HEIGHTS[(i + 9) % len(HEIGHTS)]
        sex = SEXES[i % 2]
        w = round(core.F.standard_weight(h, sex), 1)
        sid += 1
        kw = ing_dedup[i % len(ing_dedup)][0] if ing_dedup else FRESH_ING_KEYWORDS[i % len(FRESH_ING_KEYWORDS)]
        scenarios.append({'sid': f'V{sid:03d}', 'height': h, 'sex': sex, 'weight': w,
                           'mode': 'ingredient', 'anchor_or_ing_input': kw})
    return scenarios


if __name__ == '__main__':
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    scenarios = build_validation_scenarios()
    print(f'validation 시나리오 {len(scenarios)}개 생성')

    # 120csv와 겹치는 입력 조합 없는지 확인
    csv_path = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code\final_service_benchmark_out\final_service_benchmark_120_realistic_weight.csv'
    with open(csv_path, encoding='utf-8-sig') as f:
        existing = list(csv.DictReader(f))
    existing_keys = {(row['height'], row['sex'], row['mode'], row['anchor_or_ing_input']) for row in existing}
    overlap = [s for s in scenarios if (str(s['height']), s['sex'], s['mode'], s['anchor_or_ing_input']) in existing_keys]
    print(f'120csv와 겹치는 시나리오: {len(overlap)}개 (0이어야 함)')
    assert len(overlap) == 0, overlap

    out_csv = r'E:\final\rl_v2_work\validation_scenarios.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
        w.writeheader(); w.writerows(scenarios)
    print(f'저장: {out_csv}')
