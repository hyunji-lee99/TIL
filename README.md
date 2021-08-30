# TIL
2021년 8월 30일
.html - 구동 -> 웹 브라우저(크롬, 엣지, 익스플로러, 사파리, ..)
마크업 언어 -> 제목, 본문 등의 구조와 표현을 설정할 수 있는 언어 (어떤 액션을 명령하는 프로그래밍 언어와는 다름)

[구조] HTML : 웹 문서의 기본적인 골격을 담당
[표현] CSS : 각 요소들의 레이아웃, 스타일링을 담당
[동작] JavaScript : 동적인 요소(사용자와의 인터랙션)을 담당
HTML, CSS를 나누어서 쓰면 좋은 점? HTML 파일 하나를 두 개의 CSS 파일로 두 가지 스타일링 가능! 반대로, HTML 파일 여러 개를 CSS 파일로 하나로 같은 스타일링 가능

웹 표준 -> HTML5는 W3C에서 공식표준화, 이후 WHATWG(애플, 모질라, 구글, ms)에 의해 HTML living standard( HTML5 개선안, 일반적으로 HTML5로 불림) 표준화
                웹 표준을 준수하여 작성하면 운영체제, 브라우저마다 의도된 대로 보여지는 웹 페이지를 만들 수 있음
웹 접근성 -> 장애를 가진 사람과 가지지 않은 사람 모두 웹을 이용할 수 있게 하는 방식. 그 외에도 작은 화면을 가진 스마트기기나 일시적인 장애, 인터넷이 느린 사람을 위해서도 존재
웹 호환성(cross browsing) -> 웹 브라우저 버전, 종류와 관계없는 웹사이트 접근, 웹 표준 준수를 통한 브라우저 호환성 확보 가능

웹 페이지를 구성하고 있는 요소 하나하나를 "태그"라는 표기법으로 작성
오프닝 태그 -> \<p> 내용 \</p> <- 클로징 태그
오프닝 태그, 내용, 클로징 태그 통틀어 "요소(element)"라고 함
태그는 대소문자 구분하지 않음. but, html5에서는 모두 소문자로 작성하는 것을 권장

빈 요소(empty elements) -> text로 이루어진 내용이 없는 요소 e.i) 이미지\<img src="">, 수평선\<hr>, 줄바꿈\<br> 등
\<p>\</p>처럼 내용을 작성하지 않는다고 해서 빈 요소가 되는 것이 아님. 즉, 빈 요소로 정해진 것들만 빈 요소임.
이런 경우 닫는 태그를 명시하지 않아도 됨. self-closing element, void element, single tag,...등으로 불림.
빈 요소 뒤에 슬래쉬( / )를 넣은 경우가 있음.(현재는 optional, 과거 xhtml에선 반드시 넣어줬어야 함.) 

요소의 중첩(nesting) -> 요소 안에 다른 요소가 들어가는 포함관계 성립 가능.
e.i) \<ul>
        \<li>하나</li>
        \<li>둘</li>
        \<li>셋</li>
    \</ul>
포함관계를 나타내기 위해 "들여쓰기"를 사용한다.

\<!-- 주석 -->
