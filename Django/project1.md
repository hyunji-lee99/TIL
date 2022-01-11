# django를 이용해서 react로 데이터 가져오기
우선, 작업 디렉토리(djangodb로 이름지어서 생성했습니다)를 생성해서 가상환경을 만들어주고, 가상환경을 실행했습니다. pip를 업그레이드하고, 장고를 설치해줍니다.
backend 디렉토리를 생성해서 그 안에 django project를 생성합니다.
```
python3 -m venv myvenv
source myvenv/bin/activate
python3 -m pip install --upgrade pip
pip install django~=2.0.0
mkdir backend
cd backend
django-admin startproject djangoapi .
```
장고 프로젝트를 생성할 때 주의할 점은 새로운 디렉토리가 아니라 현재 디렉토리에 장고 프로젝트를 생성할 수 있도록 django-admin 명령 마지막에 마침표를 꼭 붙여야 합니다.
여기까지 수행하면 djangodb 작업 디렉토리 안에 myvenv 디렉토리와 backend 디렉토리가 있고, backend 디렉토리의 구조는 다음와 같습니다.
```
├── djangoapi
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── manage.py
```

지금부턴 api로 호출시킬 app을 만들고 db를 초기화하겠습니다. manage.py가 있는 backend 디렉토리에서 아래 명령어를 실행합니다.
```
python manage.py startapp post
python manage.py migrate
```
<code>migrate</code> 명령 실행 시
```
TypeError: argument of type 'PosixPath' is not iterable
```
라는 오류가 발생할 수 있습니다. 이러한 경우엔 <code>settings.py</code> 파일에서 데이터베이스 부분 코드를 다음과 같이 수정해줍니다.
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```

앱 <code>post</code>를 생성해주었기 때문에 <code>settings.py</code> 파일에서 INSTALLED_APPS에 'post'를 추가해줍니다.
```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'post',
]
```
다음, 앱이 잘 작동하는지 확인해보기 위해서 django의 개발용 웹서버를 실행해봅니다.
```
python manage.py runserver
```
![스크린샷 2022-01-10 오전 10.46.03](https://i.imgur.com/Z8mnzOH.png)
잘 작동합니다.

이제 모델을 정의하고 app을 활용할 수 있도록 해봅니다.
backend/post/models.py에 Post 모델을 작성해줍니다.
```
from django.db import models

# Create your models here.
class Post(models.Model):
    title=models.CharField(max_length=200)
    content=models.TextField()

    def __str__(self):
        """A string representation of the model"""
        return self.title
```
이제 데이터베이스에 새 모델을 추가해보겠습니다.
```
python manage.py makemigration post
```
=> Migrations for 'post':
  post/migrations/0001_initial.py
    - Create model Post

```
python manage.py migrate post
```
=> Operations to perform:
  Apply all migrations: post
Running migrations:
  Applying post.0001_initial... OK

작성한 모델이 성공적으로 데이터베이스에 반영되었습니다.

모델을 장고 관리자에서 추가하거나 수정, 삭제할 수 있습니다.
post/admin.py를 열어서 다음과 같이 작성합니다.
```
from django.contrib import admin
from .models import Post

# Register your models here.
admin.site.register(Post)
```
다음으로 <code>createsuperuser</code> 명령어를 이용해서 관리자를 생성합니다.
```
python manage.py createsuperuser
=> Username (leave blank to use 'username'): username
Email address: user@email.com
Password:
Password (again):
Superuser created successfully.
```
서버를 한 번 재부팅하고 admin 사이트로 접속해서 제대로 작동하는지 확인합니다.
http://127.0.0.1:8000/admin/

우리가 작성한 post 모델에 여러 개의 포스트를 추가해봅시다.
![스크린샷 2022-01-10 오전 11.13.46](https://i.imgur.com/Jall8um.png)![스크린샷 2022-01-10 오전 11.13.51](https://i.imgur.com/GN5iTpy.png)

여기까지 했으면 djano-rest-framework를 사용할 것입니다.
```
pip install djangorestframework
```
설치가 완료되면 <code>settings.py</code> 파일에 새로운 app이 추가되었음을 알립니다.
```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
	'post',
    'rest_framework',
]

REST_FRAMEWORK={
    'DEFAULT_PERMISSION_CLASSES':[
        'rest_framework.permissions.AllowAny'
    ]
}
```
이제 api app을 추가했고, api 루트로 들어왔을 때 데이터를 보낼 수 있도록 해야합니다.
어떤 형식의 데이터를 보낼지는 post app 디렉토리 안의 <code>view.py</code>에서 정해집니다.

django+react 앱은 api요청을 통해서 데이터를 주고 받는데 api 요청 및 반환값은 대부분 데이터포멧이 JSON으로 되어있/습니다. 그래서 반환값을 JSON으로 직렬화(Serialize)해주는 것이 필요합니다. 이때 필요한 것이 DRF(Django Rest Framework)의 serializers 입니다.
만들어져 있는 파일이 아니기 때문에 직접 만들어 작성해야 합니다.

post app 디렉토리 안에 <code>serializers.py</code> 파일을 만들어줍니다.
```
from rest_framework import serializers
from .models import Post

class PostSerializer(serializers.ModelSerializer):
    class Meta:
        fields=(
            'id',
            'title',
            'content',
        )
        model=Post
```
이제 views.py를 작성합니다.
```
from django.shortcuts import render
from rest_framework import generics
# Create your views here.

from .models import Post
from .serializers import PostSerializer

class ListPost(generics.ListCreateAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

class DetailPost(generics.RetrieveUpdateDestroyAPIView):
    queryset=Post.objects.all()
    serializer_class=PostSerializer
```
이제 데이터를 보낼 준비는 끝났습니다. api요청이 왔을 때 Post 데이터를 보내야 하므로 urls.py 파일을 생성해서 작성하겠습니다.
```
from django.urls import path
from . import views

urlpatterns=[
    path('',views.ListPost.as_view()),
    path('<int:pk>/',views.DetailPost.as_view()),
]
```
post 내부의 urls는 정의했기 때문에 루트 디렉토리에서 urls.py에 이 내용을 반영합니다.
```
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('post.urls')),
]
```
api 요청이 제대로 작동하는지 확인해보려면 runserver를 실행하고
http://127.0.0.1:8000/api/ 로 접속하면
![스크린샷 2022-01-10 오후 12.59.23](https://i.imgur.com/rLZuRmk.png)
아까 /admim 에서 추가해준 데이터가 뜨는 것을 볼 수 있습니다.
참고로 주소 뒤에 아이디 값을 작성해주면 해당 아이디의 데이터만 볼 수 있습니다.
즉, http://127.0.0.1:8000/api/2/ 로 접속하면 다음과 같이 해당 데이터만 볼 수 있습니다.
![스크린샷 2022-01-10 오후 1.02.04](https://i.imgur.com/nUGBzYY.png)

여기까지 하면 django api 서버의 준비는 완료되었습니다. 마지막으로 script 태그 안에 api를 통한 데이터의 접근제어를 위해 HTTP 접근제어 규약(CORS: Cross-Origin Resource Sharing)을 추가해야 합니다. 기존의 HTTP 요청에선 <code>img</code>나 <code>link</code> 태그 등으로 다른 호스트의 css나 이미지파일 등의 리소스를 가져오는 것이 가능하지만 <code>script</code> 태그로 쌓여진 코드에서 다른 도메인에 대한 요청은 Same-Origin policy에 의해서 접근이 불가능합니다. 때문에 react에서 이를 가능하게 하려면 제약 해제가 필요합니다.

이를 위해서 프로젝트 디렉토리에서 다음 명령을 실행합니다.
```
pip install django-cors-headers
```
그리고 <code>settings.py</code>에 다음 내용을 추가합니다.
```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
	  'post',
    'rest_framework',
    'corsheaders', #추가
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware', #추가
    'django.middleware.common.CommonMiddleware', #추가
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ORIGIN_WHITELIST = (
    'http://localhost:3000',
)
```

이제 간단한 react 앱에서 django에서 전송해주는 데이터를 받아보겠습니다.
backend 디렉토리를 만들었던 디렉토리 djangodb에서 create-react-app을 통해서 react앱을 만들어줍니다.
```
create-react-app frontend
```
(create-react-app을 설치하지 않았다면, npm install -g create-react-app을 먼저 실행합니다.)
그 다음 frontend 디렉토리로 이동해서 <code>npm start</code>로 리액트 앱이 정상적으로 작동하는지 확인합니다.
정상적으로 작동한다면 3000번 포트에 react가 실행된 화면이 뜹니다.
![스크린샷 2022-01-10 오후 1.19.17](https://i.imgur.com/9BFOOCi.png)

이제부턴 frontend 요청을 처리할 웹서버와 backend api 요청을 처리할 두 개의 웹서버가 작동돼야 합니다. 하지만 아마도 react 앱을 만드는 과정에서 django 웹서버가 중지됐을 것입니다. 다시 runserver를 통해서 서버를 켜줍니다.

이제 react 앱에서 보내주는 django 데이터를 받아볼 차례입니다. create-react-app 명령어로 생성된 frontend 디렉토리에서 fetch API를 이용해서 데이터를 받아오는 코드를 작성합니다.
fetch API는 react에서 Request나 Response와 같은 HTTP 파이프라인을 구성하는 요소를 조작할 수 있도록 해주는 API입니다.

frontend 디렉토리에 src/app.js 파일을 다음과 같이 수정합니다.
```
import React,{useState, useEffect} from 'react';

function App() {
  const [datas,setData]=useState([])
  useEffect(()=>{
    (async () => {
        const Tdata = await (
            await fetch('http://127.0.0.1:8000/api/')
        ).json();
        setData(Tdata);
        }
    )();

  },[])

  return (
    <div>
      {datas.map(data => <div key={data.id}>
        <h1>{data.title}</h1>
        <p>{data.content}</p>
      </div>)}
    </div>
  );
}

export default App;
```
그러면 React앱에 다음과 같이 우리가 admin에서 입력했던 데이터들이 나타납니다.
![스크린샷 2022-01-10 오후 5.24.19](https://i.imgur.com/HUwEdDY.png)
