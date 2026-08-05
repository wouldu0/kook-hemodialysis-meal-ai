# 이 폴더의 SQL은 실행하지 마세요

`002_user_features.sql`은 `server_FOOK.py`가 실제로 사용하는 컬럼/테이블 구조와 다릅니다
(예: `shopping_cart_items` 대신 `shopping_lists`+`shopping_list_items`, `user_profiles`
컬럼명 불일치 등). 이전 통합 작업 중 만들어진 초안으로 보이며, 실행 중인 코드는
`../001_full_schema.sql` 하나만 참조합니다.

향후 이 초안의 기능(알레르기 필드, 리프레시 토큰, 이벤트 로깅 등)을 실제로 도입하려면
`server_FOOK.py`와 `001_full_schema.sql`을 함께 맞춰 새 마이그레이션으로 다시 작성하세요.
