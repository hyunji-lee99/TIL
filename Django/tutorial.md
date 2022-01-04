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
