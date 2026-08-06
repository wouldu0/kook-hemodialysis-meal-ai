# -*- coding: utf-8 -*-
"""
side_dish_diagnosis_FOOK.py — 부찬(slot3) 다양성이 파이프라인 어느 단계에서 좁아지는지 진단.
C_mask100+RL(epoch260) 체크포인트, 실제 make_meal() 게이트 로직을 그대로 재현(순서·기준 불변경).

실제 게이트 적용 순서(app_core_FOOK.make_meal() 코드 그대로, 2026-07-27 재확인):
  S0 생성 직후(gen_batch)
  S1 dup_today/clash/seafood_overload/high_p_overload 계산 — 전부 "원본(adjust 전) menus" 기준
  S2 F.adjust() 적용(레버) → before, after, inst 획득
  S3 unrealistic_reason(inst) 계산 — adjust 후 inst 기준
  S4 passes(after, b) 계산(영양5종) — adjust 후 after 기준
  S5 위 6개 조건을 동시에 AND 결합 → ok
  S6 후보 목록을 앞에서부터 훑다가 첫 ok=True에서 즉시 채택, 끝까지 하나도 없으면 부분점수 최고 채택

★ 중요: 원본 코드는 게이트를 "순차 필터"(살아남은 것만 다음 게이트로)가 아니라, 6개 조건을
  전부 계산해서 동시에 AND로 합친다 — 어떤 게이트도 다른 게이트의 입력을 바꾸지 않는다(전부
  같은 F.adjust() 결과 하나를 놓고 독립적으로 판정). 그래서 이 스크립트의 "게이트별 통과/탈락"도
  전체 후보 풀에 대해 각 조건을 독립적으로 적용한 결과다 — "S3을 통과한 것만 S4로 넘어간다"가
  아니라 "S2 상태 하나에 대해 6개 조건을 각각 계산한다"는 뜻으로 해석해야 한다.

F.adjust() 반환값(FOOK_adjust_levers.py:1111 정의 확인):
  return before, after, inst, p_ok
  - before: 레버 적용 전 영양 총계(dict, totals(원본 inst))
  - after : 레버 적용 후 영양 총계(dict) — 실제 통과판정(passes)에 쓰는 값
  - inst  : 레버 적용 후 재료 인스턴스 리스트(메뉴명 변경/치환 반영) — unrealistic_reason 등이 봄
  - p_ok  : lever_phosphorus 자체 성공여부(레버가 "나는 성공했다"고 보고하는 값, app_core는
            이 값을 신뢰하지 않고 passes(after,b)로 직접 재검증함 — 코드주석에 명시돼있음)

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python side_dish_diagnosis_FOOK.py
"""
import os, sys, io, csv, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from collections import Counter, defaultdict
from scipy.spatial.distance import jensenshannon

FINAL = r'E:\final'
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, FINAL)

from Model import Encoder, Decoder
from train_FOOK_soupmask_1000 import build_data, SOUP_POS

OUT_DIR = os.path.join(CODE, 'side_dish_diagnosis_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_CONDITIONS = [('생선구이', '고등어구이', 55), ('육류', '제육불고기', 152), ('두부콩류', '두부양념조림', 36)]
SEED_ROWS = [11, 12, 6, 36, 7]        # 사전등록 10개 중 앞 5개(screening 규모)
N_CALLS = 30
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
SIDE_SLOT = 3   # 부찬


def entropy_of(counter):
    vals = np.array(list(counter.values()), dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stage_stats(counter):
    total = sum(counter.values())
    if total == 0:
        return {'n': 0, 'unique': 0, 'top1_menu': None, 'top1_ratio': 0.0, 'top5_ratio': 0.0, 'entropy': 0.0}
    top1_menu, top1_cnt = counter.most_common(1)[0]
    return {'n': total, 'unique': len(counter), 'top1_menu': top1_menu, 'top1_ratio': top1_cnt / total,
            'top5_ratio': sum(v for _, v in counter.most_common(5)) / total, 'entropy': entropy_of(counter)}


def load_model(ckpt_dir, num_tokens):
    kwargs = {'num_tokens': num_tokens, 'embed_dim': 128, 'fc_dim': 64,
              'fully-connected_layer': 'GRU', 'attention': True}
    enc = Encoder(**kwargs, batch_size=10)
    dec = Decoder(**kwargs, batch_size=10)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    assert ck, f'체크포인트 없음: {ckpt_dir}'
    tf.train.Checkpoint(encoder=enc, decoder=dec).restore(ck).expect_partial()
    print('로딩:', ck)
    return enc, dec


def gen_batch_traced(core, encoder, decoder, num_tokens, mask_id, food_dict,
                      fixed_seed_row_7tok, anchor_token, n, temp):
    """gen_batch()와 동일 로직 + 부찬(slot3) 샘플링 시점의 확률(model_side_probability) 기록."""
    seeds = np.tile(fixed_seed_row_7tok, (n, 1)).astype(np.int64)
    seeds[:, SOUP_POS] = mask_id           # C_mask100+RL: 국 위치 항상 마스킹(불변식, 안 건드림)
    fixed = {2: anchor_token}
    seeds[:, 3] = anchor_token

    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden0 = tf.zeros([n, encoder.units])
    enc_output, enc_hidden = encoder(seeds_tf, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int)
    res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]
    side_probs = np.zeros(n)

    for j in range(5):
        outputs, dec_hidden, _ = decoder(seeds_tf[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for bi in range(n):
            if j in fixed:
                res[bi, j + 1] = fixed[j]
                continue
            p = probs[bi].copy()
            for t in core.SPECIAL:
                if t < num_tokens: p[t] = 0.0
            if mask_id is not None: p[mask_id] = 0.0
            for t in core.BLOCK_TOK:
                if t < num_tokens: p[t] = 0.0
            for t in used[bi]:
                if t < len(p): p[t] = 0.0
            slot_ok = core.SLOT_OK[j]
            if num_tokens > len(slot_ok):
                slot_ok = np.append(slot_ok, np.zeros(num_tokens - len(slot_ok)))
            masked = p * slot_ok
            for gi in used_grp[bi]:
                idx = np.array([g for g in core.GRP_TOK[gi] if g < num_tokens])
                if len(idx): masked[idx] = 0.0
            if masked.sum() > 0:
                p = masked
            p = np.clip(p, 1e-12, None)
            p = p ** (1.0 / temp)
            p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            if j == SIDE_SLOT:
                side_probs[bi] = float(p[tok])
            res[bi, j + 1] = tok
            used[bi].add(tok)
            gi = core.TOK_GRP.get(tok)
            if gi is not None:
                used_grp[bi].add(gi)
    menus_list = [[food_dict[int(t)] for t in r if int(t) not in core.SPECIAL and t != mask_id] for r in res]
    return menus_list, side_probs


def adjusted_side(orig_menus, inst):
    present = set(i['menu'] for i in inst)
    kept = [m if m in present else None for m in orig_menus]
    added = [m for m in dict.fromkeys(i['menu'] for i in inst) if m not in orig_menus]
    it = iter(added)
    final = [m if m is not None else next(it, m) for m in kept]
    return final[SIDE_SLOT]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    os.chdir(cwd)
    b = F.meal_bounds(60)

    _, food_dict, diet_np, _, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    orig_diet_np_np = diet_np.numpy()

    enc, dec = load_model(RL_CKPT_DIR, num_tokens)

    seed_info = []
    for sid, row_idx in enumerate(SEED_ROWS):
        row = orig_diet_np_np[row_idx].copy()
        seed_info.append((sid, row, row_idx))

    trace_rows = []
    cid_counter = 0

    for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
        anchor_token = core.name2idx[anchor_menu]
        for sid, base_row, row_idx in seed_info:
            np.random.seed(RNG_SEED)
            for call_id in range(N_CALLS):
                menus_list, side_probs = gen_batch_traced(core, enc, dec, num_tokens, mask_id, food_dict,
                                                            base_row, anchor_token, TRIES, TEMP)
                cand_records = []
                for ci, menus in enumerate(menus_list):
                    cid_counter += 1
                    if len(menus) != 5:
                        cand_records.append(None)
                        continue
                    original_side = menus[SIDE_SLOT]
                    dup_today = False   # used_today=set() 고정(단일끼 평가, 하루연속 아님) — 원본과 동일 조건
                    clash = core._has_ingredient_clash(menus)
                    overload = core._has_seafood_overload(menus)
                    p_overload = core._has_high_p_overload(menus)
                    before, after, inst, p_ok = F.adjust(list(menus), b, anchor=anchor_menu)
                    unreal = F.unrealistic_reason(inst)
                    nut_flags = (b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                                after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax'])
                    nutrition_all_pass = all(nut_flags)
                    reality_pass = unreal is None
                    duplicate_pass = not clash
                    dup_today_pass = not dup_today
                    seafood_pass = not overload
                    high_p_pass = not p_overload
                    ok = nutrition_all_pass and reality_pass and dup_today_pass and duplicate_pass and seafood_pass and high_p_pass

                    adj_side = adjusted_side(menus, inst)
                    side_changed = (original_side != adj_side)

                    # 첫 실패 게이트(소스코드 AND식 순서 그대로: 영양,현실성,dup_today,중복,해산물,고인)
                    gate_checks = [('nutrition', nutrition_all_pass), ('reality', reality_pass),
                                   ('dup_today', dup_today_pass), ('duplicate_clash', duplicate_pass),
                                   ('seafood_overload', seafood_pass), ('high_p_overload', high_p_pass)]
                    first_fail = next((g for g, ok2 in gate_checks if not ok2), None)

                    cand_records.append({
                        'anchor_type': cond_name, 'anchor_menu': anchor_menu, 'seed_id': sid, 'seed_row_idx': row_idx,
                        'call_id': call_id, 'candidate_id': cid_counter,
                        'rice': menus[0], 'soup': menus[1], 'main_dish': menus[2], 'side_dish': menus[SIDE_SLOT],
                        'kimchi': menus[4],
                        'original_side_dish': original_side, 'adjusted_side_dish': adj_side,
                        'side_dish_changed_by_adjustment': side_changed,
                        'calories': after['E'], 'protein': after['protein'], 'sodium': after['Na_season'],
                        'potassium': after['K'], 'phosphorus': after['P'],
                        'nutrition_all_pass': nutrition_all_pass, 'reality_pass': reality_pass,
                        'duplicate_pass': duplicate_pass, 'dup_today_pass': dup_today_pass,
                        'seafood_overload_pass': seafood_pass, 'high_p_overload_pass': high_p_pass,
                        'survived': ok, 'failure_reason': None if ok else first_fail,
                        'first_failure_gate': first_fail,
                        'model_side_probability': side_probs[ci],
                        'p_ok_lever_reported': p_ok,
                    })

                # 최종선택(원본 순서: 첫 ok=True 즉시채택, 없으면 부분점수 최고)
                valid = [c for c in cand_records if c is not None]
                selected_idx = None
                best_score, best_idx = -1, None
                for idx, c in enumerate(valid):
                    if c['survived']:
                        selected_idx = idx
                        break
                    score = sum([b['Elo'] <= c['calories'] <= b['Ehi'],
                                 b['Plo'] <= c['protein'] <= b['Phi'],
                                 c['potassium'] < b['Kmax'], c['phosphorus'] < b['Pmax'],
                                 c['sodium'] <= b['Namax']])
                    if c['reality_pass']: score += 0.5
                    if c['dup_today_pass']: score += 0.3
                    if c['duplicate_pass']: score += 0.3
                    if c['seafood_overload_pass']: score += 0.3
                    if c['high_p_overload_pass']: score += 0.3
                    if score > best_score:
                        best_score, best_idx = score, idx
                if selected_idx is None:
                    selected_idx = best_idx
                for idx, c in enumerate(valid):
                    c['final_selected'] = (idx == selected_idx)

                trace_rows.extend(valid)
        print(f'[{cond_name}] 완료')

    # ── ① side_dish_candidate_trace.csv (S0/S2/S9/S10 요약 — 후보당 1행, 단계별 필드는 컬럼으로 표현) ──
    trace_csv = os.path.join(OUT_DIR, 'side_dish_candidate_trace.csv')
    fieldnames = ['anchor_type', 'anchor_menu', 'seed_id', 'seed_row_idx', 'call_id', 'candidate_id',
                  'rice', 'soup', 'main_dish', 'side_dish', 'kimchi',
                  'original_side_dish', 'adjusted_side_dish', 'side_dish_changed_by_adjustment',
                  'calories', 'protein', 'sodium', 'potassium', 'phosphorus',
                  'nutrition_all_pass', 'reality_pass', 'duplicate_pass', 'dup_today_pass',
                  'seafood_overload_pass', 'high_p_overload_pass', 'survived', 'failure_reason',
                  'first_failure_gate', 'final_selected', 'model_side_probability', 'p_ok_lever_reported']
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trace_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f'\n① {trace_csv} ({len(trace_rows)}행)')

    # ── ② side_dish_stage_metrics.csv ──
    stage_rows = []
    stages_def = [
        ('S0_생성직후', lambda r: r['side_dish']),
        ('S2_adjust후', lambda r: r['adjusted_side_dish']),
        ('S9_최종생존(ok)', lambda r: r['adjusted_side_dish'] if r['survived'] else None),
        ('S10_최종선택', lambda r: r['adjusted_side_dish'] if r['final_selected'] else None),
    ]
    for stage_name, extractor in stages_def:
        for cond_name, _, _ in ANCHOR_CONDITIONS + [('전체', None, None)]:
            rs = trace_rows if cond_name == '전체' else [r for r in trace_rows if r['anchor_type'] == cond_name]
            vals = [extractor(r) for r in rs]
            vals = [v for v in vals if v is not None]
            c = Counter(vals)
            st = stage_stats(c)
            orig_retention = np.mean([r['original_side_dish'] == r['adjusted_side_dish'] for r in rs]) if rs else None
            change_rate = 1 - orig_retention if orig_retention is not None else None
            stage_rows.append({
                'stage': stage_name, 'anchor_type': cond_name, 'candidate_count': st['n'],
                'survival_rate': st['n'] / len(rs) if rs else None,
                'unique_side_count': st['unique'], 'top1_side': st['top1_menu'], 'top1_ratio': st['top1_ratio'],
                'top5_ratio': st['top5_ratio'], 'entropy': st['entropy'],
                'original_side_retention_rate': orig_retention, 'side_change_rate': change_rate,
            })
    stage_csv = os.path.join(OUT_DIR, 'side_dish_stage_metrics.csv')
    with open(stage_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        w.writeheader(); w.writerows(stage_rows)
    print(f'② {stage_csv}')

    # ── ③ side_dish_gate_impact.csv ──
    gate_names = ['nutrition_all_pass', 'reality_pass', 'dup_today_pass', 'duplicate_pass',
                  'seafood_overload_pass', 'high_p_overload_pass']
    gate_rows = []
    base_c = Counter(r['adjusted_side_dish'] for r in trace_rows)
    base_stats = stage_stats(base_c)
    for gate in gate_names:
        input_count = len(trace_rows)
        pass_rows = [r for r in trace_rows if r[gate]]
        fail_rows = [r for r in trace_rows if not r[gate]]
        pass_c = Counter(r['adjusted_side_dish'] for r in pass_rows)
        pass_stats = stage_stats(pass_c)
        gate_rows.append({
            'gate_name': gate, 'input_count': input_count, 'pass_count': len(pass_rows),
            'fail_count': len(fail_rows), 'fail_rate': len(fail_rows) / input_count,
            'unique_before': base_stats['unique'], 'unique_after': pass_stats['unique'],
            'unique_loss': base_stats['unique'] - pass_stats['unique'],
            'entropy_before': base_stats['entropy'], 'entropy_after': pass_stats['entropy'],
            'entropy_loss': base_stats['entropy'] - pass_stats['entropy'],
            'top1_ratio_before': base_stats['top1_ratio'], 'top1_ratio_after': pass_stats['top1_ratio'],
            'top1_increase': pass_stats['top1_ratio'] - base_stats['top1_ratio'],
        })
    gate_csv = os.path.join(OUT_DIR, 'side_dish_gate_impact.csv')
    with open(gate_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        w.writeheader(); w.writerows(gate_rows)
    print(f'③ {gate_csv}')

    # ── ④ side_dish_menu_survival.csv ──
    menu_stats = defaultdict(lambda: {'generated': 0, 'adjusted_kept': 0, 'gate_pass': 0, 'final_selected': 0,
                                       'nutrients': [], 'fail_gates': Counter()})
    for r in trace_rows:
        m0 = r['side_dish']
        menu_stats[m0]['generated'] += 1
        m_adj = r['adjusted_side_dish']
        if m0 == m_adj:
            menu_stats[m0]['adjusted_kept'] += 1
        menu_stats[m_adj]['nutrients'].append(r)
        if r['survived']:
            menu_stats[m_adj]['gate_pass'] += 1
        if r['final_selected']:
            menu_stats[m_adj]['final_selected'] += 1
        if r['first_failure_gate']:
            menu_stats[m_adj]['fail_gates'][r['first_failure_gate']] += 1

    survival_rows = []
    for menu, d in menu_stats.items():
        nutrients = d['nutrients']
        survival_rows.append({
            'side_dish': menu, 'generated_count': d['generated'], 'adjusted_count': len(nutrients),
            'gate_pass_count': d['gate_pass'], 'final_selected_count': d['final_selected'],
            'generation_to_selection_rate': d['final_selected'] / d['generated'] if d['generated'] else 0,
            'most_common_failure_gate': d['fail_gates'].most_common(1)[0][0] if d['fail_gates'] else None,
            'mean_sodium': float(np.mean([n['sodium'] for n in nutrients])) if nutrients else None,
            'mean_potassium': float(np.mean([n['potassium'] for n in nutrients])) if nutrients else None,
            'mean_phosphorus': float(np.mean([n['phosphorus'] for n in nutrients])) if nutrients else None,
            'mean_protein': float(np.mean([n['protein'] for n in nutrients])) if nutrients else None,
            'mean_calories': float(np.mean([n['calories'] for n in nutrients])) if nutrients else None,
        })
    survival_rows.sort(key=lambda x: -x['generated_count'])
    survival_csv = os.path.join(OUT_DIR, 'side_dish_menu_survival.csv')
    with open(survival_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(survival_rows[0].keys()))
        w.writeheader(); w.writerows(survival_rows)
    print(f'④ {survival_csv} ({len(survival_rows)}종)')

    # ── ⑤ side_dish_failure_reasons.csv ──
    fail_rows = []
    for cond_name, _, _ in ANCHOR_CONDITIONS:
        rs = [r for r in trace_rows if r['anchor_type'] == cond_name]
        for gate in gate_names:
            fails = [r for r in rs if not r[gate]]
            if not fails:
                continue
            affected = Counter(r['adjusted_side_dish'] for r in fails).most_common(5)
            fail_rows.append({
                'anchor_type': cond_name, 'gate_name': gate, 'failure_reason': f'{gate}=False',
                'count': len(fails), 'ratio': len(fails) / len(rs),
                'top_affected_side_dishes': '; '.join(f'{m}({c})' for m, c in affected),
            })
    fail_csv = os.path.join(OUT_DIR, 'side_dish_failure_reasons.csv')
    with open(fail_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(fail_rows[0].keys()))
        w.writeheader(); w.writerows(fail_rows)
    print(f'⑤ {fail_csv}')

    # ── 콘솔 요약 ──
    print('\n===== 단계별 부찬 다양성(전체) =====')
    for row in stage_rows:
        if row['anchor_type'] == '전체':
            print(f"  {row['stage']}: 고유{row['unique_side_count']} top1={row['top1_ratio']*100:.1f}% "
                  f"entropy={row['entropy']:.2f} 생존율={row['survival_rate']*100:.1f}%")

    print('\n===== 게이트별 영향 =====')
    for g in gate_rows:
        print(f"  {g['gate_name']}: 탈락률={g['fail_rate']*100:.1f}%  "
              f"고유손실={g['unique_loss']}  entropy손실={g['entropy_loss']:.2f}  top1증가={g['top1_increase']*100:.1f}%p")

    return trace_rows, stage_rows, gate_rows, survival_rows


if __name__ == '__main__':
    main()
