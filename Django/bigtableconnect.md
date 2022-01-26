# Google Bigtable과 Django 연동하기
gcp에서 제공하는 NoSQL 데이터베이스인 Bigtable은 대용량 데이터를 처리하는 데 특화되어 있습니다.
빅데이터를 처리하기 위해 기존의 postgreSQL을 사용중이던 Django 프로젝트를 Google Bigtable 연동으로 변경해보겠습니다.

우선, Google Cloud Console의 프로젝트 선택기 페이지에서 Google Cloud 프로젝트를 선택하거나 만들어줍니다.
[프로젝트 선택](https://console.cloud.google.com/projectselector2/home/dashboard?hl=ko&_ga=2.5748403.1298516868.1642988026-1283658739.1640136893&_gac=1.181893589.1643073008.Cj0KCQiAubmPBhCyARIsAJWNpiNAr6CQwzzCy9UFABFHO7mP4sBhIlqtILtPbDKoh0ItiVzMUnDvRzIaAoK2EALw_wcB)

그리고, 해당 cloud 프로젝트에 결제가 사용 설정되어 있는지 확인하고 API를 설정합니다. [API 설정](https://console.cloud.google.com/apis/enableflow?apiid=bigtable.googleapis.com,bigtableadmin.googleapis.com&;redirect=https:%2F%2Fconsole.cloud.google.com&hl=ko&_ga=2.13104695.1298516868.1642988026-1283658739.1640136893&_gac=1.57035480.1643073008.Cj0KCQiAubmPBhCyARIsAJWNpiNAr6CQwzzCy9UFABFHO7mP4sBhIlqtILtPbDKoh0ItiVzMUnDvRzIaAoK2EALw_wcB&project=eminent-glider-213307)

[서비스 계정](https://console.cloud.google.com/projectselector2/iam-admin/serviceaccounts/create?supportedpurview=project&hl=ko&_ga=2.5036595.1298516868.1642988026-1283658739.1640136893&_gac=1.19364938.1643073008.Cj0KCQiAubmPBhCyARIsAJWNpiNAr6CQwzzCy9UFABFHO7mP4sBhIlqtILtPbDKoh0ItiVzMUnDvRzIaAoK2EALw_wcB)을 만들어줍니다. 계정 이름을 설정하고 프로젝트에 대한 액세스 권한(bigtable 관리자)을 부여합니다.
![스크린샷 2022-01-25 오후 1.31.59](https://i.imgur.com/FifDXhs.png)

그리고 위에서 만든 서비스 계정에서 키를 생성해주고, 생성한 키를 json 파일로 다운받습니다.
그 다음엔 터미널에서
```
export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
```
를 입력해서 사용자 인증 정보를 제공합니다.

다음으로 [cloud sdk를 설치 및 초기화](https://cloud.google.com/sdk/docs/install?hl=ko)합니다.
이전에 cloud SDK를 설치한 경우엔 <code>gcloud components update</code>를 이용해서 최신 버전인지 확인합니다.

이제 django와 연동할 google bigtable 인스턴스를 만들어보겠습니다.
Google Cloud Console에서 Bigtable > 인스턴스 만들기 페이지를 엽니다.
인스터스 이름, 스토리지 유형과 region 등을 설정해주고 만들기를 클릭해서 인스턴스를 만들어줍니다.
![스크린샷 2022-01-25 오후 1.47.45](https://i.imgur.com/0qifQvb.png)

이제 위에서 생성한 Bigtable 인스턴스가 잘 생성됐는지, 잘 작동하는지 확인해보겠습니다.
터미널에서 <code>.cbtrc</code>파일을 만들고, bigtable 인스턴스를 만든 프로젝트의 ID로 바꿔서 프로젝트와 인스턴스를 사용하도록 <code>cbt</code>를 구성합니다.
혹시 cbt가 설치되어 있지 않다면
```
gcloud components update
gcloud components install cbt
```
위 코드를 터미널에서 실행해서 cbt를 설치해줍니다.
그리고 다음을 입력합니다.
```
echo project = project-id > ~/.cbtrc
echo instance = bigtable instance-id >> ~/.cbtrc
```
그리고 .cbtrc 파일이 올바르게 설정됐는지 확인합니다.
```
project = project-id
instance = bigtable instance-id
```
위 내용이 출력된다면 올바르게 설정된 것입니다.

cbt를 이용해서 간단하게 bigtable이 잘 작동하는지 테스트해보겠습니다.
우선 table을 만들어줍니다.
```
cbt createtable my-table
```
실행했을 때 operation not permmited 에러가 난다면 콘솔창을 한 번 껐다가 다시 켜서 시도해보시면 됩니다.
<code>cbt ls</code>로 만든 테이블을 나열해보면 테이블이 잘 생성되었는지 확인할 수 있습니다.
그리고 column family를 1개 추가해보겠습니다.
```
cbt createfamily my-table cf1
```
그리고 <code>cbt ls my-table</code>해보면
```
Family Name	GC Policy
-----------	---------
cf1		<never>
```
위와 같이 columnfamily가 추가된 것을 확인할 수 있습니다. 다음으로 columnfamily <code>cf1</code> 및 column qualifier <code>c1</code>을 사용해서 <code>test-value</code> 값을 <code>r1</code> 행에 입력합니다.
```
cbt set my-table r1 cf1:c1=test-value
```
그리고 <code>cbt read my-table</code>을 사용해서 테이블에 추가한 데이터를 읽습니다. 그러면 결과값은 아래와 같이 나올 것입니다.
```
----------------------------------------
r1
  cf1:c1                                   @ 2022/01/25-14:21:22.251000
    "test-value"

```
그리고 이러한 테스트용 bigtable이 요금이 청구되지 않도록 <code>.cbtrc</code> 파일을 삭제합니다.
먼저 위에서 생성한 테이블 <code>my-table</code>을 삭제하겠습니다.
```
cbt deletetable my-table
```
그리고 인스턴스를 삭제합니다.
```
cbt deleteinstance djangotable
```
마지막으로, <code>.cbtrc</code> 파일을 삭제합니다.
```
rm ~/.cbtrc
```

이제 google bigtable을 생성하고, 삭제하는 등 잘 작동하는 것을 확인했습니다.

다시 bigtable 인스턴스를 만들어서 django 프로젝트와 연결해보겠습니다.
이번 실습은 react로 짜여진 front에서 데이터를 post하면 해당 데이터를 django api로 받아서 bigtable에 저장하는 실습입니다.

우선, 장고에서 <code>GOOGLE_APPLICATION_CREDENTIALS</code>을 가질 수 있도록 <code>settings.py</code> 파일에서 다음 코드를 입력해줍니다. json path는 위에서 서비스 계정을 만들고 다운받은 json 파일의 경로를 작성해주면 됩니다.
```
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="json path"
```
그리고 google bigtable에 데이터를 보낼 django view를 만들어보겠습니다. 앱 디렉토리에 <code>views.py</code>에 다음과 같은 코드를 작성합니다.
```
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from google.cloud import bigtable
from google.cloud.bigtable import column_family
from google.cloud.bigtable import row_filters
import datetime

class BigtableView(APIView):
#해당 api로 post, get, put, delete 요청이 들어오면 처리하는 DRF 라이브러리

    def post(self,request):
        project_id = 'project-id'
        instance_id = 'instance-id'
        table_id = 'table-id'

        rows=[]
        client = bigtable.Client(project=project_id, admin=True)
        instance = client.instance(instance_id)
        timestamp = datetime.datetime.utcnow()

        # 테이블 만들기
        table = instance.table(table_id)
        max_versions_rule = column_family.MaxVersionsGCRule(2)
        #columnfamily rule 설정
        column_family_id = "dummy"
        column_families = {column_family_id: max_versions_rule}
        if not table.exists():
        #테이블이 없으면
            table.create(column_families=column_families)
            #테이블 생성
        else:
        #있으면
            print("Table {} already exists.".format(table_id))
            #안내문 출력

        # row 만들기
        bigtable_data = request.data #프론트에서 post한 데이터를 bigtable_data 변수에 저장
        for i, value in enumerate(bigtable_data['text']):
        #여러 개의 데이터를 조회하면서 행을 정의하고, 생성
            row_key="data{}".format(i).encode()
            #row_key 정의
            row=table.direct_row(row_key)
            #행을 정의
            row.set_cell(column_family_id,"r", value)
            rows.append(row)
            #생성한 행을 rows 배열에 저장

        table.mutate_rows(rows)
        #rows 배열의 값을 일괄로 쓰면서 한 번에 여러 행을 db에 저장

        return Response(bigtable_data, status=status.HTTP_201_CREATED)


    def get(self,request):
        return Response("test ok",status=200)

    def put(self, request):
        return Response("test ok", status=200)

    def delete(self, request):
        return Response("test ok", status=200)

```
django에서 post로 받은 데이터를 처리해서 bigtable로 보내는 코드입니다. 그리고 api 요청을 받을 url을 앱 프로젝트의 <code>urls.py</code>에 추가해줍니다.
```
path('bigtable',views.BigtableView.as_view()),
```

이제 front에서 post로 데이터를 보내는 코드를 작성해보겠습니다. [react, django 연동해서 데이터 보내기](https://github.com/hyunji-lee99/TIL/blob/main/Django/project2.md)
django와 연동한 front에서 <code>App.js</code> 파일을 다음과 같이 수정합니다.
```
import React,{useEffect} from "react";

function App() {
   useEffect(()=>{
    (async () => {
            const response =
                await fetch('http://127.0.0.1:8000/api/bigtable', {
                    method: 'POST',
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(
                        {text:send}
                    ),
                })
                    .then(((result) => console.log(result)))
                    .catch(error=>console.log(error));
            }
    )();
    },[])

  return (
    <div className="App">

    </div>
  );
}

 const send=
  [
      "daib","research","bigtable","test","django"
   ]


export default App;
```
우리는 send라는 문자열 배열을 데이터로 전송하고자 합니다.

이제 django 서버를 실행하고, front를 실행해서 데이터를 전송해보겠습니다.
```
backend/
python manage.py runserver
```
```
frontend/
npm run start
```
그리고 데이터가 잘 전송됐는지 <code>cbt</code>로 확인해보겠습니다.

터미널을 켜고 다음을 실행해보면
```
cbt read test_bigtable
```
![스크린샷 2022-01-26 오전 11.01.18](https://i.imgur.com/7aWSI9c.png)

post했던 데이터가 잘 들어왔음을 확인할 수 있습니다!
