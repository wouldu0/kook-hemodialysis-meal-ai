import os
import sys

# FOOK_adjust_levers.py는 backend/ 바로 아래에 있고 패키지가 아니라서,
# pytest가 어느 디렉터리에서 실행되든 backend/를 sys.path에 넣어줘야 import가 된다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
