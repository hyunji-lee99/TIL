파이썬의 모듈과 패키지
모듈 : 특정 목적을 가진 함수, 자료의 모임
import 키워드를 이용해서 모듈을 사용함. e.i) import random -> random 모듈 불러옴.
모듈.함수(매개변수,..) 또는 모듈.변수 형식으로 사용함. e.i) random.randrange(start, stop)
사용자가 원하는 내용이 담긴 모듈을 제작할 수 있음. (.py 확장자 이용) 예를 들어, cal.py파일안에 def plus(a,b)라는 함수를 만들면 import cal을 이용해서 다른 모듈에 cal 모듈을 불러와서 cal.plus(2,4)처럼 활용할 수 있음. 

패키지 : 모듈을 편리하게 관리하기 위해서 모듈을 폴더(directory)로 구분하여 관리하는 것 
import를 이용해서 폴더를 불러온 후, 함수를 실행. e.i) import user.cal -> user 패키지 안에 cal 모듈을 불러옴.
from(모듈명) import(함수명) 사용하는 방식도 있음. e.i) from user.cal import plus -> user 패키지 안에 cal  모듈에서 plus함수를 불러옴.
이러한 방식을 모듈 안에 함수나 변수 사용 시 .(dot)를 써주지 않아도 됨. 

웹페이지의 정보를 쉽게 가져올 수 있는 urllib 패키지를 제공함. urllib.request.urlopen 함수는 해당 url의 html 파일을 가져옴. 

numpy : 파이썬에서 대규모 다차원 배열을 다룰 수 있게 도와주는 라이브러리
반복문 없이 배열을 처리할 수 있기 때문에 빠른 연산을 지원하고, 메모리를 효율적으로 사용할 수 있어서 빅데이터 분석 등에 많이 쓰임. import numpy as np 와 같이 import 키워드를 사용해서 불러오며, np라는 별칭을 붙여줌. 관습적으로 numpy는 np로 사용함. 

파이썬 리스트와 달리 같은 데이터 타입만 저장할 수 있는 numpy 배열의 데이터 타입을 지정하는 dtype은 다음 예시와 같이 사용함.
e.i) arr=np.array([0,1,2,3,4], dtype=float)
astype을 이용해서 배열의 데이터 타입을 바꿀 수 있음. e.i) arr.astype(int)
dtype의 종류로는 int, float, str, bool이 있음. 
ndarray의 차원 관련 속성으로는 ndim(ndimentsion), shape가 있음. 배열의 size는 배열에 들어가있는 요소의 개수이고, len()은 배열의 열의 개수 즉, 가로길이를 뜻함.

2차원 배열을 만드려면 array의 shape를 변경해서 만들어줄 수 있음. 2차원 배열의 각 행과 열을 슬라이싱해서 출력하는 것은 matrix[0:2,1:4] 이런 식으로 작성함. 
슬라이싱에서 arr[::2]와 같이 작성하면 마지막 2는 2 간격으로 잘라준다는 뜻이며, 처음부터 끝까지의 원소를 2간격으로 잘라주는 것임. 

boolean indexing은 배열의 각 요소의 선택 여부를 boolean mask를 이용하여 지정하는 방식임.
1. 조건에 맞는 데이터를 가져옴 
2. 참/거짓인지 알려줌. 
예를 들어, x=np.arange(7)일 때, print(x<3)이면 t t t f f f f와 같은 배열을 출력함. 또, print(x[x<3])이면 0 1 2를 출력함. 

fancy indexing은 배열의 각 요소 선택을 index 배열을 전달하여 지정하는 방식임. 
예를 들어, 1차원 배열일 때, x[[1,3,5]]를 출력하면 1,3,5번째에 위치한 원소를 출력하고, 2차원 배열일 때, x[[0,2]]를 출력하면 0번째,2번째 행에 위치한 원소를 출력함. 

pandas : 구조화된 데이터를 효과적으로 처리하고 저장, array 계산에 특화된 numpy를 기반으로 설계. import pandas as pd 와 같이 import 키워드를 사용해서 불러오며, pd라는 별칭을 붙여줌. 관습적으로 pandas는 pd로 사용함.
1. series : numpy의 array가 보강된 형태, data와 index를 가지고 있음.
             0 1 <-data
             1 2
index-> 2 3
             3 4
값을 ndarray 형태로 가지고 있음. dtype 인자로 데이터 타입을 지정할 수 있음. name 속성을 이용해서 series 이름을 지정할 수도 있음. 또, 인덱스를 지정할 수 있고 인덱스로 접근 가능함. 즉, index를 꼭 숫자로 하는 것이 아니라 알파벳, 특정 단어 등으로 지정 가능. 
e.i) data=pd.Series([1,2,3,4], index=['a','b','c','d']) , 딕셔너리를 이용한 series 생성도 가능함.  
e.i) dict={'korea' : 5180,
        'japan' : 12718,
        'china' : 141500,
        'usa' : 32676}
country=pd.Series(dict)

2. DataFrame : 여러 개의 series가 모여서 행과 열을 이룬 데이터
또, 딕셔너리를 활용해서 데이터프레임을 생성할 수 있음. 딕셔너리를 생성해서 DataFrame의 인자로 할당하고, set_index를 이용해서 인덱스를 설정해줄 수 있음. 데이터프레임의 shape, size, ndim, values 속성은 각 인덱스와 컬럼명은 데이터의 크기로 산정하지 않음. 데이터프레임은 to_csv,to_excel(저장경로)를 이용해서 파일로 저장할 수 있으며, read_csv,read_excel(저장경로)를 통해서 데이터프레임을 불러올 수 있음. 
e.i) pd.DataFrame({'population':population,'gdp':gdp}) population -> 컬럼명에 population 딕셔너리를 불러오고, gdp 컬럼명에 gdp 딕셔너리를 불러옴.

### 데이터 선택
데이터 프레임 안에 데이터를 선택하는 방법으로 **인덱싱과 슬라이싱**이 있음. 
loc : 명시적 인덱스를 참조하는 인덱싱/슬라이싱
e.i) country.loc['china'], country.loc['japan':'korea',:'population']
iloc : 파이썬 스타일의 정수 인덱싱/슬라이싱
e.i) country.iloc[0], country[1:3,:2]
또, **컬럼명을 활용**해서 데이터프레임에서 데이터 선택이 가능함. 
e.i) country['gdp'] -> 시리즈로 추출 출력 ,  country[['gdp']] -> 데이터프레임으로 추출 출력
masking연산이나 query 함수를 활용해서 **조건에 맞는** 데이터 프레임 행 추출도 가능함. 
e.i) country[country['population']<10000], country.query("population>100000")
두 가지 이상의 조건을 동시에 만족해야 하는 경우는 다음과 같이 작성
masking=df[(df['A']<0.5)&(df['B']>0.3)]
query=df.query("A<0.5 and B>0.3")

### 데이터 변경
시리즈도 numpy array처럼 **연산자 활용**이 가능함. 이를 이용해서 데이터프레임에 새로운 컬럼을 추가해줄 수도 있음. 
e.i) gdp_per_capita=country['gdp']/country['population']
country['gdp per capita']=gdp_per_capita
**리스트로 추가하거나 딕셔너리로 추가**할 수 있음
df = pd.DataFrame(columns=['이름','나이','주소'])  #데이터프레임 생성
df.loc[0] = ['길동', '26', '서울']  #리스트로 데이터 추가
df.loc[1] = {'이름':'철수', '나이':'25', '주소':'인천'}  #딕셔너리로 데이터 추가
df.loc[1, '이름'] = '영희'  #명시적 인덱스 활용하여 데이터 수정
**nan값으로 초기화** 한 새로운 컬럼을 추가할 수 있음. 
e.i) df['전화번호']=np.nan
df.loc[0,'전화번호']='01012345678'
데이터프레임에서 **컬럼 삭제** 후 원본 변경
e.i) df.drop('전화번호',axis=1,inplace=true) -> axis=1은 열(col) 방향, axis=0은 행(row) 방향으로 삭제함. inplace=true면 원본 변경, inplace=false면 원본 변경x, 여기서 주의할 점은 inplace 사용 시 원본데이터는 따로 저장해두고 true로 설정해야 함!

### 데이터 프레임 정렬
1. index값 기준으로 정렬 
정렬할 방향이 열 기준인지 행 기준인지 axis로 정해주고, ascending으로 오름차순(default)으로 정렬할지 내림차순으로 정렬할지 정해줌. 
axis=1은 열 인덱스 기준, axis=0은 행 인덱스 기준. ascending=true(오름차순) or false(내림차순)
e.i) df=df.sort_index(axis=0)

2. 컬럼값 기준으로 정렬
정렬한 기준이 될 컬럼명을 입력해주고, 오름차순(default)/내림차순을 결정해서 작성함. 
e.i) df.sort_values('col1',ascending=True)
여러 개의 컬럼을 한 번에 정렬할 수 있음. 
e.i) df.sort_values(['col1','col2'], ascending=[True, False]) -> col2 컬럼 기준 오름차순으로 정렬 후, col1 컬럼 기준 내림차순 정렬 (기존의 내림차순 col2를 망치지 않는 선에서 col1기준 오름차순으로 정렬)
* NaN 값은 가장 마지막 값으로 정렬

### 데이터프레임 분석용 함수
1. 집계함수 
count: 데이터 개수 확인 가능
axis 속성을 이용해서 0(열 기준), 1(행 기준)으로 데이터 개수 확인 방향을 정해줌. 기본값으로 NaN 값은 제외하고 개수를 셈. 데이터 타입은 int
e.i) df.count(axis=0)
max,min : 최대, 최소값 확인 가능
e.i) df.max(), df.min() -> 인자를 정해주지 않으면, 기본값으로 열 기준으로 계산함. 데이터 타입은 float으로 나타남. 
sum, mean : 합계 및 평균 계산
e.i) df.sum=0, df.mean() -> 기본값으로 열 기준으로 계산함. 데이터 타입은 float
**axis, skipna** 인자를 활용해서 계산 방향과 NaN값을 계산에 포함할지 여부를 결정할 수 있음. 기본값은 열 기준으로 계산하는 것이며, axis=0을 인자로 전달하면 행 기준으로 계산함. skipna=False를 작성하면 NaN을 계산에 포함. 

NaN값이 존재하는 column의 평균을 구해서 NaN을 대체할 수 있음. 
e.i) B_avg=df['math'].mean()
df['math']=df['math'].**fillna**(B_avg) -> math 열의 NaN값들을 열 값들의 평균치로 대체할 수 있음. 

group by : 조건부로 집계를 하고 싶은 경우 사용.
e.i) df.groupby('key').sum() -> key 컬럼을 기준으로 다른 컬럼 값을 합함. 
df.groupby(['key','data1']).sum() ->  두 개 이상의 조건을 동시에 묶어서 계층적으로 묶어줄 수도 있음. 

aggregate를 통해서 집계를 한번에 계산할 수 있음.
e.i) **df.groupby('key').aggregate(['min',np.median,max])** -> key 컬럼값을 기준으로 그룹화하고, 다른 데이터의 최대, 중간, 최소값을 한 번에 계산.
**df.groupby('key').aggregate({'data1':'min','data2':np.sum})** ->  key 컬럼값을 기준으로 그룹화하고, data1은 최소값을, data2는 더한 값을 구함. 

filter를 통해서 그룹속성을 기준으로 데이터 필터링할 수 있음.
e.i) def filter_by_mean(x):
        return x['data2'].mean() > 3
    df.groupby('key').filter(filter_by_mean)

apply, lambda를 통해서 묶인 데이터에 함수 적용할 수 있음. 
e.i) df.groupby('key').apply(lambda x: x.max()-x.min())

get_group으로 묶인 데이터에서 key값으로 데이터를 가져올 수 있음.
e.i) df.groupby('시도').get_group('충남') -> 컬럼 '시도'의 값이 '충남'인 데이터들만 가져옴.
len(df.groupby('시도').get_group('충남')) -> 충남에 있는 대학의 개수가 몇개일까?

matplotlib 그래프
## line plot
e.i) fig, ax=plt.subplots() -> 인자가 없으면 1개의 figure을 만듦. 
x=np.arrange(15)
y=x**2 ->  2제곱
ax={
x,y,linestyle=":",marker="*",color="#524F41"
}

linestyle의 종류는 - (solid,기본값), -- (dashed), -. (dashdot), : (dotted)가 있음.
하나의 fig에 여러 개의 라인을 그릴 수있는데, ax.plot(x,y,linestyle="-",color="0.8")와 같은 형식으로 추가할 수 있음. 여기서 color="0.8"은 0부터 1까지 gray scale임.
marker의 종류는 점(.) , 원(o) , 삼각형(v,>,^,<) , 사각형(s) , 별표(*), 삼각선(1,2,3,4), 오각형(p), 육각형(H,h)가 있음.
축 경계를 조정하는 방법은 ax.set_xlim, ax.set_ylim을 이용해서 축이 어디까지 포함할 지 축의 경계를 지정할 수 있음.
범례(legend)는 x와 y에 label을 셋팅해주고, ax.legend(loc='upper right',shadow=True,fancybox=True,borderpad=2)와 같이 범례를 그래프에 추가해줄 수 있음. 여기서 borderpad는 범례가 작성된 박스의 크기를 설정하는 역할을 함.  
set_xlabel(),set_ylabel()으로 x축과 y축에 라벨을 추가해줄 수 있음. 

## bar plot 그래프
e.i) fig, ax=plt.subplots(figsize=(12,4)) -> fig의 사이즈를 가로세로 순으로 설정해줌.
ax.bar(x,x*2) 

## histogram(도수분포표)
e.i) fig,ax=plt.subplots()
data=np.random.randn(1000)
ax.hist(data, bins=50) -> data값 1000개의 분포를 50개의 막대로 나타냄.


fig, axes=plt.subplots(1,2,figsize=(8,4)) 여기서 1,2는 하나의 fig에 1*2 모양으로 그래프를 그리는 것. 즉, 한 fig 안에 가로로 두 개의 그래프를 출력할 수 있고 각 그래프를 인덱싱으로 추출할 수 있음.

#Data set
x = np.array(["축구", "야구", "농구", "배드민턴", "탁구"])
y = np.array([18, 7, 12, 10, 8])
z = np.random.randn(1000) 

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

#Bar 그래프
axes[0].bar(x, y)
#히스토그램
axes[1].hist(z, bins = 50)

matplotlib로 pyplot을 그릴 때, 기본 폰트는 한글을 지원하지 않기 때문에
import matplotlib.font_manager as fm
fname='./NanumBarunGothic.ttf'
font = fm.FontProperties(fname = fname).get_name()
plt.rcParams["font.family"] = font
위와 같은 코드로 한글을 지원하는 폰트로 변경해주어야 함. 

##  Matplotlib with pandas
pandas 라이브러리에 read_csv() 함수를 통해서 파일을 가져와서 그래프로 분석할 수 있음. 
scatter 함수는 점을 흩뿌리는 듯한 분포 그래프를 그릴 수 있음. x값, y값, marker, color를 인자로 넘겨서 설정해줄 수 있음. 
e.i) df = pd.read_csv("./data/pokemon.csv")

#공격 타입 Type 1, Type 2 중에 Fire 속성이 존재하는 데이터들만 추출해보세요.
fire = df[(df["Type 1"]=="Fire") | (df["Type 2"]=="Fire")]
#공격 타입 Type 1, Type 2 중에 Water 속성이 존재하는 데이터들만 추출해보세요.
water = df[(df["Type 1"]=="Water") | (df["Type 2"]=="Water")]

fig, ax = plt.subplots()
#왼쪽 표를 참고하여 아래 코드를 완성해보세요.
ax.scatter(fire['Attack'], fire['Defense'],
    marker='*', color="R", label="Fire", s=50)
ax.scatter(water['Attack'], water['Defense'],
    marker='.', color="B", label="Water", s=25)

ax.set_xlabel("Attack")
ax.set_ylabel("Defense")
ax.legend(loc="upper right")
