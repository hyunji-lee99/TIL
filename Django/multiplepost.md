# 여러 개의 데이터를 한 번에 데이터베이스로 전송하기

postgreSQL이 데이터베이스로 작동하는 Django를 위해서 아래 게시글을 참고해주세요.
[PostgreSQL과 Django 연동하기]('Django/postgreSQL.md')

우선, 다음과 같은 여러 개의 데이터를 Post할 때 한 번에 데이터베이스에 분리해서 저장하고자 합니다.
```
const data=[
        {
            "title":"test2",
            "content":"content2"
        },
        {
            "title":"test3",
            "content":"content3"
        },
        {
            "title":"test4",
            "content":"content4"
        },
        {
            "title":"test5",
            "content":"content5"
        },
        {
            "title":"test6",
            "content":"content6"
        },
        {
            "title":"test7",
            "content":"content7"
        },
        {
            "title":"test8",
            "content":"content8"
        },
        ]
```
프론트에서 Post할 때 loop를 이용해서 다음과 같이 데이터를 전송하는 방법도 있지만, 서버와의 연결을 여러 번 해야하기 때문에 실제로 사용하긴 어려운 방법입니다.
```
for(i=0;i<data.length;i++){
const response =
               await fetch('http://127.0.0.1:8000/api/', {
                   method: 'POST',
                   headers: {
                       "Content-Type": "application/json",
                   },
                   body: (
                       JSON.stringfy({
                           title=data[i].title
                           content=data[i].content
                         })
                   ),
               })
                   .then(((result) => console.log(result)));
           }
 }
```
그래서 프론트에서 Post를 해줄 땐 여러 개의 데이터를 배열로 묶어서 한 번에 보내고 백엔드에서 나눠서 저장하는 방법을 사용하려고 합니다.
즉,
<img src="https://i.imgur.com/3ZVunuG.png" width="300px">=><img src="https://i.imgur.com/BdVwCMT.png" width="300px">
왼쪽 데이터처럼 데이터를 뭉쳐서 보내고, 오른쪽 데이터처럼 분리해서 저장해보겠습니다.

우선, 이렇게 배열에 담긴 여러 개의 json 데이터를 저장하려면 Django model의 필드를 JSONB field를 사용합니다.
JSONB field는 postgreSQL에서 지원하는 필드입니다. <code>app/models.py</code>에서 다음과 같이 필드를 import해줍니다.
```
from django.contrib.postgres.fields.jsonb import JSONField as JSONBField
```
그리고 다음과 같이 model을 선언해줍니다.
```
class Post(models.Model):
    data = JSONBField(default=list)
```
그리고 한 번에 받은 json 데이터들을 나눠서 저장할 model도 선언해줍니다.
```
class Sep(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
```
그리고 <code>app/serializers.py</code>에 각각의 serializer를 작성합니다.
```
from rest_framework import serializers
from .models import Post,Sep

class PostSerializer(serializers.ModelSerializer):
    class Meta:
        fields=(
            'data',
        )
        model=Post

class SepSerializer(serializers.ModelSerializer):
    class Meta:
        fields=(
            'title','content',
        )
        model=Sep
```

그리고 이런 수정사항들을 데이터베이스에도 반영합니다.
```
python manage.py makemigrations app
python manage.py migrate app
```
이제 DRF의 generics의 APIView를 이용해서 프론트로부터 받은 데이터를 받아서 화면에 출력하거나 상세화면을 보고, 삭제하거나 수정, 추가도 가능하도록 해보겠습니다. ```app/views.py```에 다음과 같은 코드를 작성합니다.
```
from rest_framework import generics
from .models import Post, Sep
from .serializers import PostSerializer, SepSerializer

class ListPost(generics.ListCreateAPIView):
    #데이터를 한꺼번에 받아와서 json 형태로 저장한 Post 모델의 데이터의 목록을 출력합니다.
    queryset = Post.objects.all()
    serializer_class = PostSerializer

class DetailPost(generics.RetrieveUpdateDestroyAPIView):
    #Post 모델의 데이터들을 불러와서 상세화면으로 보여주고, 데이터 수정,삭제가 가능합니다.
    queryset = Post.objects.all()
    serializer_class = PostSerializer

#데이터를 나눠서 Sep 모델에 저장하고 데이터베이스에 반영하는 코드
class SeperatePost(generics.ListCreateAPIView):
    #loop 돌려서 Post의 데이터들을 분리해줍니다.
    item = list(Post.objects.all().values())
    #Post.objects.all().values()는 queryset을 반환하기 때문에 list로 형변환해줍니다.
    for i in item:
        for j in i["data"]:
            ele=Sep(title=j["title"], content=j["content"]) #모델 객체에 데이터를 저장하고
            ele.save() #모델 객체를 데이터베이스에 저장합니다.
    queryset = Sep.objects.all() #Sep 모델에 저장된 데이터 모두 가져와서 ListCreateAPIView로 출력합니다.
    serializer_class = SepSerializer
```
그리고 url 등록을 해줍니다.
<code>app/urls.py</code>에 다음과 같이 view에 접속할 url을 선언해줍니다.
```
from django.urls import path
from . import views

urlpatterns=[
    path('', views.ListPost.as_view()),
    path('<int:pk>/', views.DetailPost.as_view()),
    path('sep', views.SeperatePost.as_view()),
]
```
그리고 app의 url들을 project 파일의 urls.py에서 선언해줍니다.
```
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('app.urls')), #app의 urls들을 모두 포함한다는 의미, import해줘야 합니다.
]
```
그리고 ```python manage.py runserver```로 서버를 실행하고 프론트에서 데이터를 보내고, http://127.0.0.1:8000/api 에선 뭉쳐서 한꺼번에 보낸 데이터를, http://127.0.0.1:8000/api/sep 에선 분리되서 저장된 데이터를 확인할 수 있습니다. postgreSQL의 데이터베이스에도 잘 들어왔는지 확인해보겠습니다.
![스크린샷 2022-01-18 오전 10.38.25](https://i.imgur.com/ukPNnGQ.png)
![스크린샷 2022-01-18 오전 10.38.36](https://i.imgur.com/DDJdquC.png)

위와 같이 잘 들어왔음을 확인할 수 있습니다😄
