# React와 Django 연동
백엔드를 Django로, 프론트엔드를 React로 개발하고 싶어서 찾아보던 중 좋은 방법이 있어서 정리합니다.
백엔드는 API로 프론트엔드로 데이터를 전달해주고, 프론트엔드에선 데이터를 받아서 화면을 구성해줍니다.

우선, 프로젝트 디렉토리를 만들고
```
mkdir djangoreact
```
장고를 설치해줍니다.
```
pip install django~=2.0.0
```
장고 프로젝트를 만들어줍니다.
```
django-admin startproject djangoreact
```
가상환경을 설치한 후에, 가상환경을 사용합니다.(참고로 가상환경을 끄려면 deactivate를 입력하면 됩니다.)
```
python3 -m venv myvenv
source myvenv/bin/activate
```

프로젝트를 만든 디렉토리로 가서 react 프로젝트를 생성해줍니다. (react가 없다면 npm install -g create-react-app으로 설치합니다.)
```
cd djangoreact
create-react-app frontend
```

<code>manage.py</code> 파일이 있는 경로로 이동해서 앱을 생성합니다.
```
python manage.py startapp api
```

Django와 React 연결에 필요한 pip 패키지를 설치합니다.
```
pip install django-rest-framework #django->react로 데이터 전달을 위한 API
pip install django-cors-headers #CORS 오류를 방지하기 위해 꼭 설치해야 합니다.
```

설치가 완료되었으면 <code>settings.py</code> 파일에 아래와 같이 코드를 수정해줍니다.
```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'api',
    'rest_framework',
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ORIGIN_WHITELIST = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

CORS_ALLOW_CREDENTIALS = True
```
다음으로 <code>settings.py</code> 파일 내의 static 경로를 변경해야 합니다.
```
import os #맨 위에 추가
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [#경로변경
            os.path.join(BASE_DIR, 'frontend', 'build'),
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

STATICFILES_DIRS = [#경로추가
    os.path.join(BASE_DIR, 'frontend', 'build', 'static'),
]
```

react 템플릿을 적용하기 위해서 경로를 명령어로 다시 한 번 알려줘야 합니다.
<code>manage.py</code> 파일과 <code>urls.py</code> 파일에 코드를 변경합니다.
<manage.py>
```
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoreact.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    try:
        if sys.argv[2] == 'react':
            project_root = os.getcwd()
            os.chdir(os.path.join(project_root, "frontend"))
            os.system("export NODE_OPTIONS=--openssl-legacy-provider")
            os.system("npm run build")
            os.chdir(project_root)
            sys.argv.pop(2)
    except IndexError:
        execute_from_command_line(sys.argv)
    else:
        execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
```

<urls.py>
```
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path('.*', TemplateView.as_view(template_name='index.html')),
]
```

셋팅이 완료되었고,
```
python manage.py runserver react
```
실행해주면 장고에서 react 템플릿을 build하여 템플릿이 변경되는 것을 볼 수 있습니다.
