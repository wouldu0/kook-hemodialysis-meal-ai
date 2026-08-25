import { useParams } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { RecipeBody } from "../../components/meal/RecipeBody";

// 찜한 메뉴 하나만 보는 화면. RecipePage(방금 생성한 식단의 레시피 보기)와 달리
// "이번 한 끼의 다른 메뉴" 같은 관련 없는 목록을 같이 보여주지 않고, 딱 그 메뉴의
// 재료·조리과정·영양성분과 "이 메뉴로 새 식단 만들기" 버튼만 보여준다.
export function MenuDetailPage() {
  const { name = "" } = useParams();
  const decoded = decodeURIComponent(name);
  return (
    <Shell header={false}>
      <BackHeader title={decoded} dot />
      <RecipeBody menuName={decoded} showMakeMealButton />
    </Shell>
  );
}
