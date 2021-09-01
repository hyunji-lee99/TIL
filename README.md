# TIL
**2021년 8월 30일**<br>
.html - 구동 -> 웹 브라우저(크롬, 엣지, 익스플로러, 사파리, ..)<br>
마크업 언어 -> 제목, 본문 등의 구조와 표현을 설정할 수 있는 언어 (어떤 액션을 명령하는 프로그래밍 언어와는 다름)

[구조] HTML : 웹 문서의 기본적인 골격을 담당<br>
[표현] CSS : 각 요소들의 레이아웃, 스타일링을 담당<br>
[동작] JavaScript : 동적인 요소(사용자와의 인터랙션)을 담당<br>
HTML, CSS를 나누어서 쓰면 좋은 점? HTML 파일 하나를 두 개의 CSS 파일로 두 가지 스타일링 가능! 반대로, HTML 파일 여러 개를 CSS 파일로 하나로 같은 스타일링 가능

웹 표준 -> HTML5는 W3C에서 공식표준화, 이후 WHATWG(애플, 모질라, 구글, ms)에 의해 HTML living standard( HTML5 개선안, 일반적으로 HTML5로 불림) 표준화<br>
                웹 표준을 준수하여 작성하면 운영체제, 브라우저마다 의도된 대로 보여지는 웹 페이지를 만들 수 있음.<br>
웹 접근성 -> 장애를 가진 사람과 가지지 않은 사람 모두 웹을 이용할 수 있게 하는 방식. 그 외에도 작은 화면을 가진 스마트기기나 일시적인 장애, 인터넷이 느린 사람을 위해서도 존재.<br>
웹 호환성(cross browsing) -> 웹 브라우저 버전, 종류와 관계없는 웹사이트 접근, 웹 표준 준수를 통한 브라우저 호환성 확보 가능

웹 페이지를 구성하고 있는 요소 하나하나를 "태그"라는 표기법으로 작성. <br>
오프닝 태그 -> \<p> 내용 \</p> <- 클로징 태그<br>
오프닝 태그, 내용, 클로징 태그 통틀어 "요소(element)"라고 함. <br>
태그는 대소문자 구분하지 않음. but, html5에서는 모두 소문자로 작성하는 것을 권장. <br>

빈 요소(empty elements) -> text로 이루어진 내용이 없는 요소 e.i) 이미지\<img src="">, 수평선\<hr>, 줄바꿈\<br> 등<br>
\<p>\</p>처럼 내용을 작성하지 않는다고 해서 빈 요소가 되는 것이 아님. 즉, 빈 요소로 정해진 것들만 빈 요소임.<br>
이런 경우 닫는 태그를 명시하지 않아도 됨. self-closing element, void element, single tag,...등으로 불림.<br>
빈 요소 뒤에 슬래쉬( / )를 넣은 경우가 있음.(현재는 optional, 과거 xhtml에선 반드시 넣어줬어야 함.)

요소의 중첩(nesting) -> 요소 안에 다른 요소가 들어가는 포함관계 성립 가능<br>
e.i) \<ul><br> 
        \<li>하나</li><br> 
        \<li>둘</li><br>
        \<li>셋</li><br>
    \</ul><br>
포함관계를 나타내기 위해 "들여쓰기"를 사용한다.

\<!-- 주석 -->

**2021년 8월 31일**<br>
html 문서의 구조
\<!doctype html> : 이 문서의 타입이 html이라는 뜻. 생략해도 문제는 없으나 오랜 시간동안 이것을 선언하는 것이 관습화되어 있어서 남아있는 것임.<br>
\<html>\</html>: 페이지 전체 내용을 감싸는 최상위(root) 요소<br>
\<head>\</head>:웹브라우저 화면에 직접적으로 나타나지 않는 페이지의 정보 e.i) meta tag(문서의 일반적인 정보와 문자 인코딩을 명시), title 등<br>
\<body>\</body>:웹브라우저 화면에 나타나는 모든 콘텐츠

*mdn : firefox 브라우저를 만든 모질라에서 html 태그에 대한 정보 많음. 태그 검색 시 뒤에 붙여서 검색하면 확인 가능

head 태그 : \<html> 요소의 첫 번째 자식으로 배치. 주 목적은 기계 처리를 위한 정보이고, 사람이 읽을 수 있는 정보가 아님. title, script, style sheet 등이 속하고, 제목은 반드시 하나만.<br>
html5 호환 브라우저는 \<head>가 없는 경우 자동으로 생성. 하지만 구형 브라우저는 지원X -> 반드시 적어야 한다!<br>
body 태그 : html 문서의 내용을 나타냄. 한 문서에 하나의 body태그 사용. 다양한 attribute 적용 가능하지만, 적합하지 않음. css나 js로 적용하는 것이 좋음.<br>

태그를 구분짓는 특성(\<body>영역에 들어가는 태그 中)
1. 구획을 나누는 태그 : 단독으로 사용하면 눈에 보이지 않음. 여러 가지 요소들을 묶어서 그룹화. 즉, 레이아웃을 위한 태그이며 컨테이너 역할을 할 수 있음.
2. 그 자체로 요소인 태그 : 단독으로 사용해도 눈으로 확인 가능. 

1. 블록(block) : 블록 레벨 요소는 언제나 새로운 줄에서 시작하고, 좌우 양쪽으로 최대한 늘어나 가능한 모든 너비를 차지함. 때문에 옆에 어떤 요소도 올 수 없음. 
2. 인라인(inline) : 인라인 요소는 줄의 어느 곳에서나 시작할 수 있고, 바로 이전 요소가 끝나는 지점부터 시작하여 요소의 내용만큼만 차지함.
태그가 아무런 스타일링이 되어 있지 않다면 왼쪽 상단부터 채워짐. 블록, 인라인 특성은 바꿀 수 있음!  
같은 형태의 다른 요소를 안에 포함할 수 있음. (블록 > 블록, 인라인 > 인라인). 
대부분의 블록 요소는 다른 인라인 요소를 포함할 수 있으나 그 반대는 불가능!

콘텐츠 카테고리. 
1. 플로우 콘텐츠(flow contents) : 일부 메타데이터 콘텐츠를 제외하고 거의 모든 요소가 속함. 보통 텍스트나 임베디드 콘텐츠를 포함.
2. 섹션 콘텐츠(section contents) : 웹 문서의 구획을 나눌 때 사용.
3. 헤딩 콘텐츠(heading contents) : 섹션의 제목과 관련된 요소.
4. 프레이징 콘텐츠(phrasing contents) : 문단에서 텍스트를 마크업할 때 사용. e.i) 텍스트의 사이즈나 자간 등
5. 임베디드 콘텐츠(embeded contents) : 이미지나 비디오 등 외부 소스를 가져오거나 삽입할 때 사용되는 요소.
6. 인터랙티브 콘텐츠(interactive contents) : 사용자와의 상호작용을 위한 컨텐츠 요소. 
7. 메타데이터 콘텐츠(metadata contents) : head 태그에 들어가는 문서의 정보나 다른 문서를 가리키는 링크 등을 나타내는 요소.

**2021년 9월 1일**<br>
VScode 단축키(맥 버전)
현재 창 닫기 cmd+w<br>
닫은 창 다시 열기 cmd+shift+t<br>
에디터 확대 cmd+(+)<br>
에디터 축소 cmd+(-)
  
들여쓰기  탭 또는 cmd+] <- 후자는 전체 문장 들여쓰기<br>
내어쓰기 shift+tab 또는 cmd+[<br>
아래에 행 삽입 cmd+enter<br>
위에 행 삽입  cmd+shift+enter<br>
현재 행 이동 opt+방향키<br>
현재 행 복사 opt+shift+방향키<br>
현재 행 삭제 cmd+shift+k<br>
주석 토글  cmd+/<br>

h태그 : h1-h6 숫자가 클수록 레벨이 낮음.(프레이징 콘텐츠에 속함)<br>
웹 브라우저가 h1-h6까지 제목의 정보를 사용해서 자동으로 문서 콘텐츠의 표(목차)를 작성함.<br>
제목 단계를 건너뛰는 것을 피해야 함. 예를 들어, h2->h1->h3 <br>
글씨 크기를 조정하기 위해서 h태그를 사용하지 말아야 함. 글씨 사이즈를 설정하고 싶으면 css의 font-size 속성을 사용하면 됨. -> 웹 브라우저마다 h 태그의 글씨 크기를 다르게 설정했기 때문에 브라우저 종류마다 글씨 크기가 달라질 수 있기 때문에!<br>
페이지 당 하나의 \<h1>을 사용해야 함. h1은 전체 페이지의 목적을 설명해야 하기 때문에! 또, 구글이나 네이버같은 검색엔진이 검색 결과를 표시할 때 내부 정보를 수집해서 결과를 띄어주게 되는데 검색 엔진들이 웹페이지들을 돌아다니면서 검색 결과를 만들어냄. 검색엔진이 h1을 먼저 찾기 때문에 페이지를 정확하게 찾아낼 수 있음. 

p태그 : 하나의 문단을 나타냄. 책에서는 문장들의 집합을 문단이라고 하지만 html에선 텍스트뿐만 아니라 이미지나 입력 폼 등 서로 관련있는 콘텐츠 무엇이든 가능.<br>
블록 레벨 요소이므로 인라인 레벨 요소를 포함할 수도 있음. <br>
\<p>첫 번째 문단입니다.<br>
첫 번째 문단입니다.<br>
첫 번째 문단입니다.<br>
첫 번째 문단입니다.<br>
\</p><br>
위 예시와 같이 p태그 안에 개행을 포함해서 입력을 해도 결과에서는 개행을 무시하고 출력됨.<br>
\<p>Lorem ipsum dolor sit, amet consectetur adipisicing elit. In, laudantium minus nisi vero amet id nam pariatur quis facere iusto consectetur! Quaerat aspernatur autem tempore veniam minus incidunt sit vero!\</p>
\<p>Lorem ipsum dolor sit, amet consectetur adipisicing elit. In, laudantium minus nisi vero amet id nam pariatur quis facere iusto consectetur! Quaerat aspernatur autem tempore veniam minus incidunt sit vero!\</p>
위 예시와 같이 p태그 여러 개를 사용하면 결과에서는 문단 사이에 여백이 생김. 이러한 여백은 텍스트 내용의 한 줄 너비만큼 차지함. 이러한 여백을 정확히 몇 px로 할 지 정하고 싶으면 css 스타일링으로 여백을 줄 수 있음. <br>
e.i) p { <br>
margin-bottem : 2px;<br>
}<br>
빈 p태그를 이용해서 여백을 만들려고 하면 안됨. 스크린 리더가 어떤 p 태그를 읽고 다음 p 태그를 읽으려고 할 때 p 태그가 비어있으면 작동이 어색할 수 있음.

*lorem 사용해서 임의의 긴 문장을 입력할 수 있음.<br>
*html에선 여러 개의 스페이스가 있어도 하나의 스페이스로 처리함.<br>

br태그 : 텍스트 안에 줄바꿈을 생성함. 즉, line-break할 때 사용. 빈 요소이기 때문에 클로징태그 필요없음. <br>
문단과 문단 사이의 여백을 많이 주고 싶다고 해서 br태그를 많이 사용하면 안됨. 여백을 많이 주어 나누고 싶은 문단을 각각의 p태그로 나누고 css 스타일링을 사용해서 여백을 늘려주어야 함.

block quote, q태그 : 둘 다 인용을 목적으로 함. block quote 태그는 블록 요소이고, 텍스트가 긴 인용문에 사용함. q태그는 인라인 요소이며, 텍스트가 짧은 인용문에 사용함. <br>
<img width="1164" alt="스크린샷 2021-09-02 오전 1 06 53" src="https://user-images.githubusercontent.com/58133945/131708464-9941f19d-b855-4642-ae8b-b64f4b81df99.png">
<img width="1087" alt="스크린샷 2021-09-02 오전 1 23 04" src="https://user-images.githubusercontent.com/58133945/131708493-06daf758-54e7-48ce-8db8-baa443a76ac7.png">
block quote는 일반 p태그와 다르게 앞쪽에 여백이 들어감. block quote 안에 여러 개의 p태그를 넣을 수 있음.<br>
p 태그 내부에 block quote를 작성하면 안됨. p태그는 내부에 있는 요소가 block 요소이면 자동으로 p태그가 닫힘. 즉, "~같이 이야기한다."에서 p태그가 닫힌 걸로 인식됨. 그래서 앞에서 닫았는데 뒤에서  또 닫은 걸로 인식하기 때문에 웹페이지가 정확하게 인식을 하지 못함.<br>
q 태그는 기본 스타일링으로 앞뒤 쌍따옴표가 들어감. 인라인 요소기 때문에 개행이 되지 않고 연결되어 출력됨. <br>
block quote 태그와 q 태그가 공통으로 사용할 수 있는 attribute는 인용문의 출처 문서나 메시지를 가리키는 url, 인용문의 맥락 또는 출처 정보를 가리킬 용도인 cite를 사용할 수 있음. attribute는 사용자의 눈에는 보이지 않기 때문에 브라우저만 알고 있는 정보로 사용됨. 출처를 알아야 할 필요가 있을 때 이러한 attribute를 이용해서 알 수 있는 것임. 
