"""
meal_result()의 raw_menus_display 필드를 확인한다.

'식단을 생성했어요' 화면(MealResultPage)은 레버 재구성 전 메뉴를 보여줘야 하는데(영양
판정 화면이 nutrition_before를 쓰는 것과 같은 원칙, 2026-08), 기존 raw_menus는 표시명
정리(F.display_menu_name, 예: '돈까스소스'->'돈까스+소스')가 안 된 내부 원본 이름이라
그대로 노출하면 안 된다. raw_menus_display = [disp(m) for m in raw_menus]가 맞는지
소스 레벨에서 확인한다(app_core_FOOK.py는 TF 모델 로딩 비용이 있어 다른 test_*_route*.py와
같은 이유로 직접 import하지 않는다).
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


def test_meal_result_adds_raw_menus_display_without_removing_raw_menus():
    src = _extract_function_source(CORE_SRC, 'meal_result')
    assert "'raw_menus': menus," in src               # 기존 필드(하루 중복 체크용) 그대로 유지
    assert "'raw_menus_display': [disp(m) for m in menus]," in src


def test_day_result_forwards_meal_result_dict_as_is():
    # day_result()는 meal_result()가 만든 dict를 그대로 out에 담으므로(label만 추가),
    # raw_menus_display도 별도 코드 없이 끼니마다 자동으로 딸려간다.
    src = _extract_function_source(CORE_SRC, 'day_result')
    assert "r = meal_result(" in src
    assert "out.append(r)" in src
