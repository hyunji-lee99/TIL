# Django 시작하기
### 가상환경
프로젝트 디렉토리 안에 가상환경을 생성해줍니다.
```
python3 -m venv myvenv
```
가상환경에 접속해줍니다.
```
source myvenv/bin/activate</code>
```

### 장고 설치
장고를 설치하하는 데 필요한 pip가 최신버전인지 확인합니다.
```
python3 -m pip install --upgrade pip
```
장고를 설치해줍니다.
```
pip install django~=2.0.0
```
다음으로 git(git-scm.com)을 설치하고, pythonanywhere(www.pythonanywhere.com)에 무료 계정인 초보자로 회원가입합니다.

 ### 누군가 서버에 웹사이트를 요청한다면?
 웹 서버에 요청이 오면 장고로 전달됩니다. 장고 urlresolver는 웹페이지의 주소를 가져와 무엇을 할지 확인합니다. urlresolver는 패턴 목록을 가져와 URL과 맞는지 처음부터 하나식 대조해 식별합니다. 만약 일치하는 패턴이 있으면 장고는 해당 요청을 관련된 함수(view)에 넘겨줍니다.

### 장고 프로젝트 만들기
```
django-admin startproject mysite .
```
명령어 끝에 ```.```(마침표)를 입력하는 것 주의합니다.

<code>django-admin.py</code>는 스크립트로 디렉토리와 파일들을 생성합니다. 스크립트 실행 후엔 아래와 같이 새로 만들어진 디렉토리 구조를 볼 수 있습니다.
```
djangogirls
├───manage.py
└───mysite
        settings.py
        urls.py
        wsgi.py
        __init__.py
```
<code>manage.py</code>는 스크립트인데, 사이트 관리를 도와주는 역할을 합니다. 이 스크립트로 다른 설치 작업없이, 컴퓨터에서 웹 서버를 시작할 수 있습니다.
<code>settings.py</code>는 웹사이트 설정이 있는 파일입니다.
<code>urls.py</code> 파일은 urlresolver가 사용하는 패턴 목록을 포함하고 있습니다.

### 설정 변경
<code>settings.py</code>에서 <code>TIME_ZONE</code>의 값을 변경해줍니다.
```
TIME_ZONE='Asia/Seoul'
```
다음으로 정적파일 경로를 추가해 줍니다. 파일의 끝으로 내려가서, <code>STATIC_URL</code> 항목 바로 아래에 <code>STATIC_ROOT</code> 항목을 추가합니다.
```
STATIC_URL = '/static/'
STATIC_ROOT=os.path.join(BASE_DIR, 'static')
```

<code>DEBUG</code>가 <code>True</code>이고 <code>ALLOWED_HOSTS</code>가 비어 있으면, 호스트는 <code>['localhost', '127.0.0.1', '[::1]']</code>에 대해서 유효합니다. 애플리케이션을 배포할 땐 pythonanywhere을 사용하기 때문에 호스트 이름과 일치시켜 주기 위해서 다음 설정을 아해와 같이 변경해줘야 합니다.
```
ALLOWED_HOSTS=['127.0.0.1','.pythonanywhere.com']
```
### 데이터베이스 설정하기
사이트 내에 데이터를 저장하기 위한 다양한 데이터베이스 소프트웨어 중 <code>sqlite3</code>를 사용하겠습니다.
이미 <code>mysite/settings.py</code> 파일 안에 설치가 되어있습니다.
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```
데이터베이스를 생성하기 위해선 콘솔 창에서 아래 코드를 실행해야 합니다.
```
python magage.py migrate
```
성공적으로 실행되면,
```
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying sessions.0001_initial... OK
```
위와 같은 메시지가 출력됩니다.

### 서버 실행
프로젝트 디렉토리에서 콘솔창에 ```python manage.py runserver``` 명령을 실행하면 웹 서버를 바로 시작할 수 있습니다.
웹 서버가 성공적으로 실행되면,
```
Performing system checks...

System check identified no issues (0 silenced).
January 04, 2022 - 13:41:40
Django version 2.0.13, using settings 'mysite.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```
위와 같은 메시지가 출력됩니다.
웹사이트가 잘 작동하는지 확인해보기 위해서 사용하는 브라우저에서 주소(http://127.0.0.1:8000/)로 들어가보면,

![스크린샷 2022-01-04 오후 1.43.37](https://i.imgur.com/lcNwu9B.png)
위와 같은 웹사이트가 나타납니다.
웹 서버를 실행하는 동안은 추가 명령을 입력할 수 있는 새로운 명령어 프롬포트가 표시되지 않기 때문에 새 터미널에 명령어를 입력할 순 있지만 실행되지 않습니다. 웹 서버가 들어오는 요청을 대기하기 위해 지속적으로 실행하고 있기 때문입니다.
웹 서버가 실행되는 동안엔 새 터미널 창을 열고 가상환경을 실행하면 명령어를 입력하고 실행할 수 있습니다.

# 장고 모델
기본적으로 객체지향설계는 객체 속성(propertise), 메소드(method)로 구현됩니다.
블로그를 만든다고 가정하고, 블로그 글 모델을 만든다면 어떤 속성들을 가져야 할까?
```
Post(게시글)
--------
title(제목)
text(내용)
author(글쓴이)
created_date(작성일)
published_date(게시일)
```
또, 블로그 글로 할 수 있는 건은 출판하는 것. 즉, publish 메소드가 있어야 합니다.

장고 안의 모델은 객체의 특별한 종류입니다. 이 모델을 저장하면 그 내용이 데이터베이스에 저장됩니다. 쉽게 말해 데이터베이스 안의 모델이란 엑셀 스프레드시트와 같다고 할 수 있습니다. 엑셀 스프레드시트는 열(필드)와 행(데이터)로 구성되어 있는 것처럼 모델도 마찬가지입니다.

### 어플리케이션 만들기
프로젝트 디렉토리 내부에 별도의 어플리케이션을 만드려면 콘솔창에서 다음 명령어를 실행합니다.
```
python manage.py startapp blog
```
이제 blog 디렉토리가 생성되고 그 안에 여러 파일도 같이 들어갈 것입니다.
```
djangogirls
    ├── mysite
    |       __init__.py
    |       settings.py
    |       urls.py
    |       wsgi.py
    ├── manage.py
    └── blog
        ├── migrations
        |       __init__.py
        ├── __init__.py
        ├── admin.py
        ├── models.py
        ├── tests.py
        └── views.py
```
애플리케이션을 생성하면 장고에 사용한다고 알려줘야 합니다. 이 역할을 하는 파일이 ```mysite/settings.py```입니다.
이 파일 안에서 아래와 같이 ```INSTALLED_APPS```에 'blog'를 추가해줍니다.
```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog',
]
```
### 블로그 글 모델 만들기
모든 Model 객체는 ```blog/models.py```파일에 선언하여 모델을 만듭니다.
```
from django.conf import settings
from django.db import models
from django.utils import timezone


class Post(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(
            default=timezone.now)
    published_date = models.DateTimeField(
            blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title
```
모델을 정의하는 코드, <code>class</code>는 객체를 정의한다는 의미, <code>Post</code>는 모델의 이름, <code>models</code>는 Post가 장고 모델임을 의미합니다.

<code>models.CharField</code> - 글자 수가 제한된 텍스트를 정의할 때 사용합니다.
<code>models.TextField</code> - 글자 수에 제한이 없는 긴 텍스트를 위한 속성입니다.
<code>models.DateTimeField</code> - 날짜와 시간을 의미합니다.
<code>models.ForeignKey</code> - 다른 모델에 대한 링크를 의미합니다.

<code>def publish(self):</code>는 메소드를 의미합니다. 메소드 이름을 붙일 때는 공백 대신, 소문자와 언더스코어를 사용해야 합니다.

### 데이터베이스에 모델을 위한 테이블 만들기
이제 데이터베이스에 새 모델 Post를 추가할 것입니다. 먼저 장고 모델에 몇 가지 변화가 있다는 것을 알려야 합니다.
```
python manage.py makemigrations blog
```
입력하면 데이터베이스에 반영할 수 있도록 마이그레이션 파일(migration file)이라는 것을 준비해 두었습니다.
 ```
 python manage.py migrate blog
 ```
 위 명령을 실행해, 실제 데이터베이스에 모델 추가를 반영하겠습니다.

# 장고 관리자
방금 막 모델링한 글들을 장고 관리자에서 추가하거나 수정, 삭제할 수 있습니다.
<code>blog/admin.py</code>를 열어서 내용을 다음과 같이 바꿉니다.
```
from django.contrib import admin
from .models import Post

admin.site.register(Post)
```
코드에서 알 수 있듯이 Post 모델을 가져오고 있습니다. 관리자 페이지에서 만든 모델을 보려면 <code>admin.site.register(Post)</code>로 모델을 등록해야 합니다.

브라우저를 열고 http://127.0.0.1:8000/admin/ 에 접속하면 로그인 페이지를 볼 수 있습니다.
이 페이지에 로그인하기 위해선, 모든 권한을 가지는 슈퍼 사용자를 생성해야 합니다.
```
python manage.py createsuperuser
```
콘솔창에서 위 명령어를 입력합니다.
메시지가 나타면 사용자 이름, 이메일 주소 및 암호를 입력하면 슈퍼 사용자로 등록됩니다.  

# 배포하기
인터넷 상에 서버를 제공하는 업체들 중에 비교적 배포 과정이 간단하고 방문자가 많지 않은 소규모 애플리케이션을 위한 무료 서비스를 제공하는 pythonanywhere을 사용합니다.
로컬 컴퓨터는 개발 및 테스트를 수행하고, 개발이 완료되면 프로그램 복사본을 GitHub에 저장합니다. 그리고 웹사이트는 pythonanywhere에 있고 GitHub에서 코드 사본을 업데이트합니다.

Git은 repositoty에 특정한 파일들 집합의 변화를 추적하여 관리합니다.
콘솔창에서 프로젝트 디렉터리 아래 다음 명령어를 실행합니다.
```
git init
git config --global user.name "username"
git config --global user.email "useremail"
```
git 저장소 초기화는 프로젝트를 시작할 때 딱 한 번만 필요합니다. git에서 변경점 추적을 제외할 특정 파일명을 <code>.gitignore</code>에 추적받지 않게 합니다.
기본 디렉토리에 <code>.gitignore</code>라는 파일을 만들면 됩니다.
```
*.pyc
*~
__pycache__
myvenv
db.sqlite3
/static
.DS_Store
```

그리고 난 다음, 우리가 지금까지 만든 파일들을 git에 업로드 합니다.
```
git status
git add --all .
git status
git commint -m "memomemo"
git remote add origin https://github.com/<your-github-username>/my-first-blog.git
git push -u origin master
```

### pythonanywhere
pythonanywhere에서 git 저장소를 clone해줍니다.
```
git clone https://github.com/<your-github-username>/my-first-blog.git
```

pythonanywhere에서 가상환경을 생성해줍니다.
```
cd my-first-blog
virtualenv --python=python3.6 myvenv
source myvenv/bin/activate
pip install django~=2.0
```

pythonanywhere에서 데이터베이스를 생성해줍니다.
```
python manage.py migrate
python manage.py createsuperuser
```
이제 코드도 pythonanywhere에 있고, 가상환경도 준비가 되었고, 정적 파일들도 모여있고, 데이터베이스도 초기화되었습니다.
이젠 웹앱으로 배포할 수 있습니다. 대시보드로 와서 ```Web->Add a new web app```을 선택합니다.
도메인 이름을 확정한 후, 대화창에 수동설정을 클릭하세요, PYTHON 3.6을 선택하고 다음을 클릭하면 마법사가 종료됩니다.

가상환경을 설정하려면 "가상환경(virtualenv)" 섹션에서 ```가상환경 경로를 입력해주세요 (Enter the path to a virtualenv)```라고 쓰여있는 빨간색 글자를 클릭하고 우리가 만든 가상환경을 선택합니다. 이동 경로 저장을 하려면 파란색 박스에 체크 표시를 하고 클릭하세요.

#### WSGI 파일 설정하기
장고는 "WSGI 프로토콜"을 사용해서 작동합니다. 이 프로토콜은 파이썬을 이용한 웹사이트를 서비스하기 위한 표준입니다. WSGI 설정 파일을 수정해서 우리가 만든 장고 프로젝트를 pythonanywhere에서 인식하게 합니다.

WSGI 설정 파일 링크(페이지 상단에 있는 Code 섹션 내 <code>/var/www/<your-username>_pythonanywhere_com_wsgi.py</code>)를 클릭하면 에디터를 볼 수 있습니다.
모든 내용을 삭제하고 아래 내용을 넣으세요.
```
import os
import sys

path = '/home/<your-PythonAnywhere-username>/my-first-blog'  # PythonAnywhere 계정으로 바꾸세요.
if path not in sys.path:
    sys.path.append(path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'

from django.core.wsgi import get_wsgi_application
from django.contrib.staticfiles.handlers import StaticFilesHandler
application = StaticFilesHandler(get_wsgi_application())
```
이 파일은 PythonAnywhere에게 웹 어플리케이션의 위치와 Django 설정 파일명을 알려주는 역할을 합니다.
Reload버튼을 누르면 페이지 최상단에 배포된 링크를 발견할 수 있을겁니다.

# Django View
첫 뷰를 만들어봅시다.
<code>blog/views.py</code>에 view를 추가하는 코드를 작성합니다.
```
from django.shortcuts import render

# Create your views here.
def post_list(request):
    return render(request, 'blog/post_list.html', {'posts':posts})
```
이 함수는 post_list라는 함수를 만들었습니다. 이 함수는 요청(request)를 넘겨받아 render 메소드를 호출합니다. <code>blog/post_list.html</code> 템플릿(아직은 작성하지 않았습니다. 곧 작성할 거에요!)을 보여줍니다.

이렇게 작성한 뷰를 사용하기 위해선 <code>URLconf</code>를 만들어서 뷰를 URL에 맵핑해야 합니다. 장고는 <code>URLconf(URL configuration)</code>을 사용합니다. URLconf는 장고에서 URL과 일치하는 뷰를 찾기 위한 패턴들의 집합입니다. blog 디렉터리 안에 URLconf를 만들기 위해서 urls.py라는 이름의 파일을 만들어서 아래 두 줄을 추가합니다.
```
from django.urls import path
from . import views
```
이 두 줄로 장고함수인 path와 blog 애플리케이션에서 사용할 모든 <code>views</code>를 가져왔습니다.
다음으로, URL 패턴을 추가합니다.
```
urlpatterns=[
  path('',views.post_list,name='post_list')
]
```
이제 <code>post_list</code>라는 <code>view</code>가 루트 URL에 할당되었습니다. 이 URL 패컨은 빈 문자열에 매칭되고, 장고 URL 확인자(resolver)는 전체 URL 경로에서 접두어에 포함되는 도메인 이름(i.e. http://127.0.0.1:8000/)을 무시하고 받아들입니다. 이 패턴은 장고에게 누군가 웹사이트에 'http://127.0.0.1:8000/'로 접속했을 때 <code>views.post_list</code>를 보여줍니다.

다음으로, 루트 URLconf가 <code>blog.urls</code> 모듈을 볼 수 있도록 해줘야 합니다. <code>mysite/urls.py</code> 파일에 <code>django.conf.urls.include</code>를 import하고 urlpatterns 리스트 안에 <code>include()</code>를 넣어 줍니다.
```
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('blog.urls')),
]
```
<code>include()</code> 함수는 루트 URLconf가 다른 URLconf를 참조할 수 있도록 합니다. 지금부터 장고는  http://127.0.0.1:8000/ 로 들어오는 모든 접속 요청을 <code>blog.urls</code>로 전송해 추가 명령을 찾습니다.

# Django Templates
템플릿은 <code>blog/templates/blog</code>디렉토리에 저장됩니다. 먼저 <code>blog</code> 디렉토리 안에 하위 디렉토리인 <code>templates</code>를 생성합니다. 그리고 <code>templates</code> 디렉토리 내 <code>blog</code>라는 하위 디렉토리를 생성합니다. post_list.html과 같은 html 파일에 웹에 보여주고자 하는 내용을 작성하면 됩니다.
```
blog
└───templates
 └───blog
```

# Django shell
```
python manage.py shell
```
위 명령을 콘솔에서 실행합니다. 실행하면 다음과 같은 인터랙티브 콘솔로 들어갑니다.
```
(InteractiveConsole)
>>>
```
파이썬 프롬프트와 비슷하지만, 장고만의 명령을 내릴 수 있기도 하고, 파이썬의 모든 명령어를 여기서 사용할 수 있습니다.

### 모든 객체 조회하기
아래 코드처럼 모든 객체를 조회하는 코드를 작성하면 NameError가 발생합니다.
```
>>> Post.objects.all()
Traceback (most recent call last):
      File "<console>", line 1, in <module>
NameError: name 'Post' is not defined
```
객체를 조회하려면 먼저 import를 해줘야 합니다.
```
>>> from blog.models import Post
```
Post 모델을 blog.models에서 불러왔습니다. 이제 우리가 등록했던 모든 글이 출력됩니다.
```
>>> Post.objects.all()
<QuerySet [<Post: my post title>, <Post: another post title>]>
```
이러한 게시글들은 장고 관리자 인터페이스로 만들었던 것들이비낟. 그런데, 파이썬으로 글을 포스팅하는 방법도 있습니다.

### 객체 생성하기
데이터베이스에 새 글 객체를 저장하는 방법입니다.
우선 작성자로서 User 모델의 인스턴스를 가져와서 전달해줘야 합니다.
```
>>> from django.contrib.auth.models import User
>>> me=User.objects.get(username='ola')
```
위와 같이 <code>사용자이름(username)</code>가 'ola'인 User 인스턴스를 받아왔습니다.
```
>>> Post.objects.create(author=me, title='Sample title',text='Test')
```
이제 파이썬으로 데이터베이스에 새로운 게시글을 등록했습니다.
제대로 작동했는지 확인해봅시다.
```
>>> Post.objects.all()
<QuerySet [<Post: my post title>, <Post: another post title>, <Post: Sample title>]>
```
같은 방식으로 여러 개의 글을 추가해보세요.

장고 쿼리셋의 중요한 기능은 데이터를 필터링하는 것입니다. 예를 들어, 제목(title)에 'title'이라는 글자가 들어간 글들만 뽑아내고 싶다면?
```
>>> Post.objects.filter(title__contains='title')
[<Post: Sample title>, <Post: 4th title of post>]
```
잘 출력됩니다. 여기서 title과 contains 사이에 있는 밑줄(_)은 2개입니다. 장고 ORM은 필드 이름("title")과 연산자와 필터("contains")를 밑줄 2개로 구분해서 사용합니다.
파이썬 콘솔에서 추가한 게시물은 먼저 게시하려는 게시물의 인스턴스를 얻어서 <code>publish</code> 메소드를 사용해서 게시합니다.
```
>>> post = Post.objects.get(title="Sample title")
>>> post.publish()
```

# 템플릿 동적 데이터
콘텐츠(데이터베이스 안에 저장되어 있는 모델)가져와서 템플릿에 넣어 보여주는 것을 해보려고 합니다.
뷰는 모델과 템플릿을 연결하는 역할을 합니다. <code>post_list</code>를 뷰에서 보여주고 이를 템플릿에 전달하기 위해서는 모델을 가져와야 합니다. 일반적으론 뷰가 템플릿에서 모델을 선택하도록 만들어야 합니다.

<code>blog/views.py</code> 파일을 열어서 <code>post_list</code> 뷰 내용을 봅시다.
```
from django.shortcuts import render

def post_list(request):
  return render(request,'blog/post_list.html',{})
```
<code>models.py</code> 파일에 정의된 모델을 가져옵니다. ```from .models import Post```를 추가합니다.
models 앞에 마침표<code>.</code>는 현재 디렉토리 또는 애플리케이션을 의미합니다. 동일한 디렉토리 내에 views.py, models.py가 있기 때문에 <code>. 파일명</code> 형식으로 내용을 가져올 수 있습니다.
Post 모델에서 블로그 글을 가져오기 위해선 쿼리셋이 필요합니다.
```
from django.shortcuts import render
from django.utils import timezone
from .models import Post

def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {})
```

<code>blog.views.py</code> 파일 내에서 posts로 쿼리셋의 이름을 짓고 이를 이용해서 데이터를 정렬합니다.  
이제 posts 쿼리셋을 템플릿에 보내는 방법을 알아봅시다.
render 함수의 <code>{}</code> 부분에 매개변수 <code>{'posts': posts}</code>를 추가합니다.
```
from django.shortcuts import render
from django.utils import timezone
from .models import Post

def post_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {'posts': posts})
```
이제 템플릿으로 돌아가서 매개변수로 전달한 쿼리셋을 보이게 해봅시다.
이때, 장고에 내장된 <b>템플릿 태그(template tag)</b>라는 기능을 사용합니다.

### 템플릿 태그란?
브라우저는 파이썬 코드를 이해할 수 없기 때문에 HTML에 파이썬 코드를 바로 넣을 순 없습니다. 템플릿 태그는 파이썬을 HTML로 바꿔주어, 빠르고 쉽게 동적인 웹사이트를 만들어 줍니다.

위에서 글 목록이 있는 posts 변수를 템플릿으로 넘겨주었습니다. 이제 넘겨진 posts 변수를 받아 HTML에 나타나도록 해보겠습니다.
장고 템플릿 안에 있는 값을 출력하려면 중괄호 안에 변수 이름을 넣어 표시해야 합니다.
```
{{ post }}
```
post_list.html에서 적절한 위치에 <code>{{ posts }}</code>를 넣어줍니다.
장고가 <code>{{ posts }}</code>를 객체 목록으로 이해하고 처리합니다. 목록을 보여주는 방식으로 <code>for loop</code>를 이용할 수 있습니다.
```
{% for post in posts %}
    {{ post }}
{% endfor %}
```
다음과 같은 형식으로 적용할 수 있습니다.
```
<div>
    <h1><a href="/">Django Girls Blog</a></h1>
</div>

{% for post in posts %}
    <div>
        <p>published: {{ post.published_date }}</p>
        <h1><a href="">{{ post.title }}</a></h1>
        <p>{{ post.text|linebreaksbr }}</p>
    </div>
{% endfor %}
```

# 템플릿 확장하기
장고에선 <b>템플릿 확장(template extending)</b>이 가능합니다. 웹사이트 안의 서로 다른 페이지에서 HTML의 일부를 동일하게 재사용할 수 있다는 뜻입니다.
이 방법을 사용하면 동일한 정보/레이아웃을 사용하고자 할 때, 모든 파일마다 같은 내용을 반복해서 입력할 필요가 없게 됩니다. 또, 뭔가 수정할 부분이 생겼을 때, 각각 모든 파일을 수정할 필요 없이 딱 한번만 수정하면 됩니다.

### 기본 템플릿 생성하기
기본 템플릿은 웹사이트 내 모든 페이지에 확장되어 사용되는 가장 기본적인 템플릿입니다.
<code>blog/templates/blog/</code>에 <code>base.html</code> 파일을 만듭니다.
```
blog
└───templates
    └───blog
            base.html
            post_list.html
```

<code>post_list.html</code>에 있는 모든 내용을 <code>base.html</code>에 아래 내용을 복사해 붙여넣습니다.
```
{% load static %}
<html>
    <head>
        <title>Django Girls blog</title>
        <link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css">
        <link href='//fonts.googleapis.com/css?family=Lobster&subset=latin,latin-ext' rel='stylesheet' type='text/css'>
        <link rel="stylesheet" href="{% static 'css/blog.css' %}">
    </head>
    <body>
        <div class="page-header">
            <h1><a href="/">Django Girls Blog</a></h1>
        </div>

        <div class="content container">
            <div class="row">
                <div class="col-md-8">
                {% for post in posts %}
                    <div class="post">
                        <div class="date">
                            {{ post.published_date }}
                        </div>
                        <h1><a href="">{{ post.title }}</a></h1>
                        <p>{{ post.text|linebreaksbr }}</p>
                    </div>
                {% endfor %}
                </div>
            </div>
        </div>
    </body>
</html>
```

그 다음 <code>base.html</code>에서 <code>{% for post in posts %}</code>부터 <code>{% endfor %}</code>까지의 코드를 다음 코드로 바꿉니다.
```
{% block content %}
{% endblock %}
```
이 코드의 의미는 우리가 block을 만들었다는 의미입니다. 템플릿 태그 <code>{% block %}</code>으로 HTML 내에 들어갈 수 있는 공간을 만들었습니다.

<code>blog/templates/blog/post_list.html</code> 파일을 열어서 <code>{% for post in posts %}</code>부터 <code>{% endfor %}</code>까지만 남기고 전부 지웁니다. 그러면 <code>post_list.html</code> 파일의 내용은 다음 코드만 남을 겁니다.
```
{% for post in posts %}
    <div class="post">
        <div class="date">
            {{ post.published_date }}
        </div>
        <h1><a href="">{{ post.title }}</a></h1>
        <p>{{ post.text|linebreaksbr }}</p>
    </div>
{% endfor %}
```
이 코드를 모든 컨텐츠 블록에 대한 템플릿의 일부로 사용합니다. 이 파일에 블록 태그를 추가해봅시다.
블록 태그가 <code>base.html</code> 파일의 태그와 일치해야 합니다. 또, 콘텐츠 블록에 속한 모든 코드를 포함하게 만들어야 합니다.
이를 위해서 post_list.html 파일의 코드를 <code>{% block content %}</code>와 <code>{% endblock %}</code>로 묶어줍니다.
```
{% block content %}
    {% for post in posts %}
        <div class="post">
            <div class="date">
                {{ post.published_date }}
            </div>
            <h1><a href="">{{ post.title }}</a></h1>
            <p>{{ post.text|linebreaksbr }}</p>
        </div>
    {% endfor %}
{% endblock %}
```
이제 마지막으로, 두 템플릿을 연결해줘야 합니다. 확장 태그를 파일 맨 처음에 추가합니다.
```
{% extends 'blog/base.html' %}

{% block content %}
    {% for post in posts %}
        <div class="post">
            <div class="date">
                {{ post.published_date }}
            </div>
            <h1><a href="">{{ post.title }}</a></h1>
            <p>{{ post.text|linebreaksbr }}</p>
        </div>
    {% endfor %}
{% endblock %}
```
