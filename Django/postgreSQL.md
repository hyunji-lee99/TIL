# postgreSQL, pgAdmin4 설치, Django 연동

~~https://www.postgresql.org/~~
~~에서 postgreSQL을 설치해줍니다.~~
![스크린샷 2022-01-13 오후 3.04.11](https://i.imgur.com/YCmaXNq.png)
~~응용프로그램에 위치한 postgreSQL 디렉토리에서 SQL shell을 열어줍니다.~~

공식사이트에서 다운받아서 설치하려고 하니 auth 오류가 계속 나서🤬 방법을 찾다찾다가 원래 공식사이트에서 다운받는 것이 오류가 많다는 것을 알았습니다. 그래서 homebrew로 설치하기로 했습니다🤯.

```
brew install postgresql
```
설치가 완료되면 postgreSQL을 실행합니다.
```
pg_ctl -D /usr/local/var/postgres start && brew services start postgresql
```
설치가 잘 되었는지 ```postgres -V```로 확인합니다.
=> ```postgres (PostgreSQL) 14.1```
잘 설치되었습니다!

이제 비밀번호를 설정하겠습니다.
먼저 postgresql에 접속합니다.
```
psql postgres
```
Django에서 사용할 새로운 데이터베이스를 생성합니다.
```
create database 데이터베이스명
```
이제 생성한 데이터베이스로 접속해서 몇가지 설정을 해보겠습니다.
```
\c 데이터베이스명
create user username with password 'password';
alter role root set client_encoding to 'utf-8';
alter role root set timezone to 'Asia/Seoul';
grant all privileges on database 데이터베이스명 to username;
```

그리고 나중에 django와 연동해서 psycopg2를 설치할 때 문제가 생길 수 있기 때문에 openssl을 설치해주고, 다음 명령어를 수행해서 경로를 추가해줍니다.
```
brew install openssl
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"
```

이제 postgreSQL 설치는 끝났습니다.

이제 postgreSQL을 더 편하게 쓰게 해주는 pgAdmin4를 설치해보려고 합니다.
https://www.pgadmin.org/download/
위 링크로 접속해서 dmg파일을 다운받아서 설치합니다.

접속해서 Servers에서 우클릭, create를 설정하면 다음과 같은 창이 뜹니다.
여기선 Name을 설정해줍니다.
![스크린샷 2022-01-14 오전 9.41.42](https://i.imgur.com/R2WPu2W.png)
저는 Django로 설정했습니다.
그리고 Connection 탭에서도 다음과 같이 설정해줍니다.

여기선 위와 같이 작성해주고, username과 password는 아까 \du 명령어에서 확인했던 이름과 설정한 비밀번호를 입력합니다.
![스크린샷 2022-01-14 오전 9.41.46](https://i.imgur.com/z94gJjZ.png)
그러면 정상적으로 서버가 생성됩니다!

Servers > LocalDev > Databases 에서 우클릭하면 Create > Database를 선택해서 새로운 데이터베이스를 생성할 수 있습니다.

이제 Django와 postgreSQL을 연동할 준비를 마쳤습니다!

우선 Django 프로젝트 디렉토리에서 settings.py 파일의 DATABASES 부분을 다음과 같이 수정합니다.
```
SECRET_KEY='secret key를 입력하세요.'

DATABASES={
    'default':
            {'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'DB명',
            'USER': 'postgre User명',
            'PASSWORD': '비밀번호',
            'HOST': 'localhost',
            'PORT': '5432', }
}
```

그리고 파이썬에서 postgreSQL을 연결하는 라이브러리인 psycopg2를 설치합니다.
```
pip3 install psycopg2
```

venv 가상환경 안에서도 설치해주셔야 합니다!
그리고 migrate 해줍니다.
```
python manage.py migrate
```
그리고 서버를 실행해보면 정상작동함을 알 수 있습니다 :)


<!-- 그런데 settings.py 파일에 입력을 해두면 나중에 github에 업로드하거나 누군가 나의 코드를 봤을 때 데이터베이스의 비밀번호와 정보가 들어있기 때문에 보안상 위험이 될 수 있다고 생각했습니다. 그래서 settings 파일을 하나 더 만들어서 분리해주기로 했습니다.

프로젝트 디렉토리에 DbSettings.py라는 이름의 파일을 만들고, 그 안에 위 코드를 작성합니다.
그리고 settings.py 파일에는 다음과 같이 작성합니다.
```
from (프로젝트 디렉토리명) import DbSettings

SECRET_KEY=DbSettings.SECRET_KEY

DATABASES=DbSettings.DATABASES
``` -->
