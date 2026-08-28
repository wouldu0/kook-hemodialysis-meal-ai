"""
make_meal()이 48회 시도 후에도 완전 통과 후보를 못 찾았을 때(실패 폴백 경로) 사용자에게
내려주는 note/warning 문구를 확인한다.

수정 전에는 이 경로에서 note에 내부 디버그 정보([48회 완전통과 실패 → 최선 X/5])와
warning 내용을 그대로 또 이어붙여서, 화면에 "AI 안내"와 "⚠ 나트륨 안내" 두 박스가
같은 문구로 중복 표시되고 내부 점수(부동소수점 raw 값)까지 노출됐다(2026-08).

- note: 순수 생성 안내만(앵커 선택 이유, 잡곡→흰쌀밥 대체 사유 등) — 실패 시에도 동일해야 함.
- warning: 영양 기준을 못 맞춘 항목 안내(나트륨 전용이 아니라 열량·단백질·칼륨·인·나트륨 전체).
- 시도 횟수·candidate score 같은 내부 정보는 서버 print 로그로만 남고 note에는 안 들어감.

app_core_FOOK.py는 TF 모델을 모듈 최상단에서 로딩해 비용이 커서(다른 test_*.py와 같은 이유)
직접 import하지 않고 소스 문자열 수준에서 확인한다.
"""
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
CORE_SRC = (BACKEND_DIR / 'app_core_FOOK.py').read_text(encoding='utf-8')


def _extract_function_source(src: str, name: str) -> str:
    m = re.search(rf'^def {re.escape(name)}\(', src, re.MULTILINE)
    assert m, f'{name}() 정의를 찾지 못했습니다.'
    lines = src[m.start():].splitlines(keepends=True)
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() == '' or line[:1] in (' ', '\t'):
            out.append(line)
        else:
            break
    return ''.join(out)


MAKE_MEAL_SRC = _extract_function_source(CORE_SRC, 'make_meal')


def test_failure_path_returns_plain_note_without_debug_suffix():
    # note는 note 변수 그대로 반환 — 다른 문자열과 이어붙이지 않는다.
    assert re.search(r"return best, note, b, anchor, warn\s*\n", MAKE_MEAL_SRC), (
        "make_meal()의 실패 폴백 반환문이 'return best, note, b, anchor, warn' 형태가 아닙니다."
    )
    # 예전에 note에 붙이던 디버그 접미사 패턴이 반환문에서 사라졌는지 확인.
    assert "return best, note + f' [" not in MAKE_MEAL_SRC
    assert "\\n  ⚠ ' + warn if warn else ''" not in MAKE_MEAL_SRC


def test_warn_stays_a_separate_field_untouched():
    # warning 문구 자체(각 영양소별 warns.append(...))는 이번 정리 대상이 아니므로 그대로 있어야 함.
    assert "warn = ' '.join(warns)" in MAKE_MEAL_SRC
    assert "단백질이 기준(≥{b[\"Plo\"]:.0f}g)에 모자라요" in MAKE_MEAL_SRC


def test_debug_info_moved_to_server_log_only():
    # 시도 횟수·최선 score는 사용자 note가 아니라 print()로만 남는다(서버 로그 전용).
    assert "print(f'  [make_meal]" in MAKE_MEAL_SRC
    assert "완전통과 실패" in MAKE_MEAL_SRC  # 로그 문자열 자체엔 남아있어야 함(제거 대상 아님)
    assert "best_score" in MAKE_MEAL_SRC


def test_success_path_note_unchanged():
    # 통과하는 후보를 즉시 반환하는 성공 경로는 이번 수정과 무관해야 한다(그대로 유지).
    assert "return cand, note, b, anchor, _total_na_warning(after)" in MAKE_MEAL_SRC


def test_resample_and_scoring_logic_untouched():
    # 후보 선택/resample 로직 자체는 건드리지 않았음을 확인(개수·점수식 그대로).
    assert "def make_meal(menu=None, ingredient=None, W=60, tries=48, temp=0.8" in MAKE_MEAL_SRC
    assert "for menus in gen_batch(tok_anchor, n=tries, temp=temp):" in MAKE_MEAL_SRC
    assert "score = sum([b['Elo'] <= after['E'] <= b['Ehi']" in MAKE_MEAL_SRC


def test_meal_result_forwards_note_and_warning_without_reprocessing():
    src = _extract_function_source(CORE_SRC, 'meal_result')
    assert "'note': note.strip()," in src
    assert "'warning': warn," in src
