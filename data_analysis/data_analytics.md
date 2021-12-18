# 국내 코로나 환자 데이터를 활용한 데이터 분석
read_csv()로 가져온 데이터를 info() 함수를 이용해서 불러온 데이터의 정보를 요약해서 출력함. 
1. 데이터를 정제하는 방법-비어있는 column 지우기
모든 컬럼 값이 null인 column을 삭제함. -> info() 함수에서 Non-Null Count가 0인 열을 삭제한다고 할 수 있음. 주의할 점은 원본에서 삭제하는 것이 아니라 새로운 변수에 저장하고, 새로운 변수에서 건드려야 함!
corona_del_col=corona_all.drop(columns=['국적','환자정보','조치사항'])
-> corona_del_col.info()로 제대로 삭제됐는지 확인.

2. 데이터 시각화
**확진일 데이터 전처리하기**
확진일 데이터가 월.일 형태의 날짜 형식임을 알 수 있고, 월별, 일별 분석을 위해서 문자열 형식의 데이터를 나누어 숫자 형 데이터로 변환
확진일 데이터를 month, day 데이터로 나눔
데이터 프레임에 month와 day를 바로 추가하지 않고, 리스트를 만들어서 임시로 데이터를 저장해둠.
month=[]
day=[]
for data in corona_del_col['확진일']:
    month.append(data.split('.')[0])
    day.append(data.split('.')[1])
    => 여기서 split 함수의 원리는 data를 '.'을 기준으로 나누는데 인덱스 0번째 데이터를 month에 추가하고, 1번째 데이터를 day에 추가한다는 것.
그 후에,
corona_del_col['month']=month
corona_del_col['day']=day
corona_del_col['month'].astype('int64')
corona_del_col['month'].astype('int64')
위와 같이 리스트에 저장한 데이터를 corona_del_col 데이터에 새로운 컬럼을 추가해서 저장해줌.

**월별 확진자 수 출력**
위에서 만든 month데이터를 바탕으로 달별 확진자 수를 막대그래프로 출력함.
막대그래프의 x축을 정리하기 위해서 리스트를 생성.
order=[]
for i in range(1,11): -> 데이터에서 1월부터 10월까지의 데이터만 존재하기 때문에 10월까지만 x축에 포함시켜줌.
    order.append(str(i)) -> 문자열로 저장하기 위해 str로 형 변환

seaborn 라이브러리의 countplot 함수를 사용해서 출력함.-> 파이썬에서 따로 계산을 해서 그려주는 것이 아니라, 원본데이터에서 바로 계산해서 그려줌.
sns.set(style="darkgrid")
ax=sns.countplot(x="month",data=corona_del_col,palette="Set2",order=order) -> palette는 그래프의 색상을 정해주는 속성

series의 plot 함수를 사용한 출력 방법도 있음.
corona_del_col['month'].values_counts().plot(kind='bar') -> 여기서 values_counts() 함수는 각 데이터를 세어서 내림차순으로 정리하는 함수임.

**8월달 일별 확진자 수 출력**
   막대그래프의 x축을 정리하기 위해서 리스트를 생성.
   order2=[]
   for i in range(1,32):
        order2.append(str(i))

seaborn 모듈의 countplot 함수를 사용
sns.set(style='darkgrid')
ax=sns.countplot(x='day',data=corona_del_col[corona_del_col['month']==8], palette='rocket_r',order=order2)

**8월 평균 일별 확진자 수를 구하세요. (8월 총 확진자 수/31)**
corona_del_col[corona_del_col['month']==8]['day'].count()/31 -> month==8인 day들의 개수
**corona_del_col[corona_del_col['month'] == '8']['day'].value_counts().mean()**

**지역별 확진자 수 출력**
import matplotlib.font_manager as fm
font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = **fm.findSystemFonts**(fontpaths=font_dirs) 
for font_file in font_files:
    **fm.fontManager.addfont**(font_file)
    한글 출력을 위해서 폰트 옵션을 설정함.
sns.set(font="NanumBarunGothic", 
rc={"axes.unicode_minus":False}, -> 축의 값이 -값이면 -부호가 깨져보이기 위해 이를 방지하기 위해서 유니코드 -는 false로 설정하는 것.
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_del_col, palette="Set2")

**지역 이상치 데이터 처리**
'종랑구'라는 잘못된 데이터와 '한국'이라는 지역과는 맞지 않는 데이터가 있음을 알 수 있음.
기존 지역 데이터 특성에 맞도록 종랑구-> 중랑구, 한국->기타로 데이터를 변경해야 함.
replace 함수를 이용해서 해당 데이터를 변경함.
이상치가 처리된 데이터는 새로운 데이터 프레임에 저장함.
corona_out_region=corona_del_col.replace({'종랑구':'중랑구','한국':'기타'})

**8월달 지역별 확진자 수 출력**
논리연산을 활용해서 해당 조건에 맞는 데이터 추출
**corona_out_region[corona_out_region['month']==8]**

plt.figure(figsize=(20,10))
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_out_region[corona_del_col['month'] == '8'], palette="Set2")

**월별 관악구 확진자 수 출력**
corona_out_region['month'][corona_out_region['지역'] == '관악구'] -> 지역이 관악구인 확진자들의 month 값
plt.figure(figsize=(10,5))
sns.set(style="darkgrid")
ax = sns.countplot(x="month", data=corona_out_region[corona_out_region['지역'] == '관악구'], palette="Set2", order = order)

**서울 지역에서 확진자를 지도에 출력** -ing
지도를 출력하기 위해 folium 라이브러리를 사용함
import folium
map_osm=folium.Map(location=[37.529622,126.984307],zoom_start=11) -> 위도와 경도를 지정하고, 초기 화면에 크기를 지정함
확진자의 수를 지도에 추가하려면 각 행정지역의 지역좌표가 필요하기 때문에, 공공데이터에서 서울시 행정구역 시군 정보 데이터를 불러와서 사용함.
CRS=pd.read_csv("./data/서울시 행정구역 시군구 정보 (좌표계_ WGS1984).csv")
for문을 사용해서 지역마다 확진자를 원형 마커를 사용해서 지도에 출력함.
corona_out_region의 지역 컬럼에는 '타시도','기타' 등 위치를 특정할 수 없는 데이터가 존재하므로 해당 컬럼을 삭제함. 
corona_seoul = corona_out_region.drop(corona_out_region[corona_out_region['지역'] == '타시도'].index) -> 특정 index를 특정할 때는 이와 같이 사용.
corona_seoul = corona_seoul.drop(corona_out_region[corona_out_region['지역'] == '기타'].index)
서울 중심지 중구를 가운데 좌표로 잡아 지도를 출력함.
map_osm = folium.Map(location=[37.557945, 126.99419], zoom_start=11)
지역 정보를 set 함수를 이용해서 25개 고유의 지역을 뽑아냄.
for region in set(corona_seoul['지역']): -> set함수는 여러 값들 가진 데이터에서 중복을 삭제해서 집합으로 저장함.
        count=len(corona_seoul[corona_seoul['지역']==region]) -> 해당 지역의 데이터 개수를 count에 저장함.
        CRS_region=CRS[CRS['시군구명_한글']==region] -> 시군구명_한글 컬럼 값이 region인 데이터를 뽑아냄.
        marker=follium.CircleMaker([CRS_region['위도'],CRS_region['경도']] -> 위치
        radius=count/10+10,  -> 범위
        color="#3186cc", -> 선 색상
        fill_color='#3186cc', -> 면 색상
        popup=' '.join((region,str(count),'명'))) -> 팝업 설정
        marker.add_to(map_osm) -> 생성한 원형마커를 지도에 추가

**6월에 확진자가 가장 많이 나온 지역을 구하세요**
top=corona_out_region[corona_del_col['month']==6]['지역'].value_counts()
quiz2=top.index[0]

## 지하철 승하차 인원 분석하기
 metro_all 데이터프레임에서 호선명 데이터 확인
 sorted(list(set(metro_all['호선명']))) -> set함수로 호선명 중복을 제거해서 리스트로 나타내고, sorted 함수로 오름차순 정렬
 metro_all 데이터프레임에서 지하철역 데이터 확인
 sorted(list(set(metro_all['지하철역']))) 
 
 1. 데이터 정제
 **2021년 6월 승하차 인원만 추출**
 metro_recent = metro_all[metro_all['사용월']==202106] -> 수집된 데이터 중 가장 최근인 6월에 수집한 데이터만 추출하고 불필요한 컬럼을 제거.
 metro_recent=metro_recent.drop(columns={'작업일자'})->불필요한 작업일자 컬럼 제거
 
 2. 데이터 시각화
 2021년 6월 데이터만 추출한 metro_recent를 활용해서 다양한 데이터 시각화 및 혼잡도 분석
 
 **호선 별 이용객 수 출력** -> 어렵다!!!!!!!
 이용객 수가 가장 많은 호선 순으로 막대그래프를 출력
 metro_line = metro_recent.groupby(['호선명']).mean().reset_index() -> 호선명 컬럼을 기준으로 groupby해주고, 그룹화된 각 컬럼별로 각 mean()함수로 평균을 계산하고, reset_index()로 인덱스 초기화, 하나의 값에 대해서 함수를 여러가지 쓰고 싶을 때 이와 같이 연결해서 사용할 수 있음. 
 metro_line = metro_line.drop(columns='사용월').set_index('호선명') -> 6월 데이터만 사용하기 때문에 사용월 컬럼은 필요가 없기 때문에 삭제하고, 호선명으로 index 설정
 metro_line = metro_line.mean(axis=1).sort_values(ascending=False) -> 가로축을 기준으로 평균을 계산하고, sort_values를 이용해서 내림차순 정렬

 plt.figure(figsize=(20,10))
 plt.rc('font', family="NanumBarunGothic") -> rc함수로 폰트설정
 plt.rcParams['axes.unicode_minus'] = False -> 유니코드 -값은 삭제!

 metro_line.plot(kind=('bar'))
 plt.show()
 
 **특정 호선에서 역별 평균 승하차 인원 데이터 추출**
 이용객이 가장 많은 2호선 기준으로 역별 평균 승하차 인원 데이터를 추출
 line = '2호선'
 metro_st = metro_recent.groupby(['호선명','지하철역']).mean().reset_index() -> metro_recent 데이터를 호선명과 지하철역을 기준으로 묶음. reset_index로 metro_recent의 인덱스를 초기화해줌.
 metro_st_line2 = metro_st[metro_st['호선명']==line] -> metro_st 중 호선명이 2호선인 데이터만 추출
 metro_st_line2
 
 #승차 인원 컬럼만 추출
 metro_get_on = pd.DataFrame()
 metro_get_on['지하철역'] = metro_st_line2['지하철역']
 for i in range(int((len(metro_recent.columns)-3)/2)): -> 3번째 인덱스부터 홀수 번째에만 승차인원이 존재하기 때문에 전체 열의 개수에서 3을 빼고 2로 나눈 개수만큼만 반복
 metro_get_on[metro_st_line2.columns[3+2*i]] = metro_st_line2[metro_st_line2.columns[3+2*i]] -> 3번째, 5번째, 7번째 등.. 승차인원이 있는 컬럼만 활용!!, metro_st_line2.columns[3+2*i] 는 3,5,7,..번째의 컬럼명을 반환함.
 metro_get_on = metro_get_on.set_index('지하철역')
 metro_get_on
 
 #하차 인원 컬럼만 추출
 metro_get_out=pd.DataFrame()
 metro_get_out['지하철역']=metro_st_line2['지하철역']
 for i in range(int((len(metro_recent.columns)-3)/2)):
    metro_get_out[metro_st_line2.columns[4+2*i]]=metro_st_line2[metro_st_line2.columns[4+2*i]]
  metro_get_out = metro_get_out.set_index('지하철역')
  
  #역 별 평균 승하차 인원을 구한 후 정수로 형 변환하여 데이터프레임으로 저장
  df = pd.DataFrame(index = metro_st_line2['지하철역']) -> index만 가지는 데이터프레임이 생성될 뿐, 아직 빈 데이터프레임임.
  df['평균 승차 인원 수'] = metro_get_on.mean(axis=1).astype(int)
  df['평균 하차 인원 수'] = metro_get_out.mean(axis=1).astype(int)
  
  **평균 승하차 인원 수 내림차순으로 막대그래프 출력**
  top10_on = df.sort_values(by='평균 승차 인원 수', ascending=False).head(10) -> df를 평균 승차 인원 수 기준으로 내림차순 정렬함.

  plt.figure(figsize=(20,10))
  plt.rc('font', family="NanumBarunGothic")
  plt.rcParams['axes.unicode_minus'] = False

  plt.bar(top10_on.index, top10_on['평균 승차 인원 수'])
  for x, y in enumerate(list(top10_on['평균 승차 인원 수'])): -> enumerate 리스트 값을 열거하며, x에는 인덱스를, y에는 값이 들어감.
      if x == 0: -> 인덱스가 들어가는 x가 0이라는 것은 가장 큰 값이라는 의미이므로 가장 큰 값은 빨간색으로 표시해줌! 
      plt.annotate(y, (x-0.15, y), color = 'red') ->  y는 값이고, x-0.15, y는 표시가 들어가는 위치인데, 문구를 좀 더 자연스럽게 해주기 위해서 -0.15해주고, 빨간색으로 강조표시
      else:
          plt.annotate(y, (x-0.15, y))

  plt.title('2021년 6월 평균 승차 인원 수 Top10')
  plt.show()
  
  top10_off = df.sort_values(by='평균 하차 인원 수', ascending=False).head(10)

  plt.figure(figsize=(20,10))
  plt.rc('font', family="NanumBarunGothic")
  plt.rcParams['axes.unicode_minus'] = False

  plt.bar(top10_off.index, top10_off['평균 하차 인원 수'])
  for x, y in enumerate(list(top10_off['평균 하차 인원 수'])):
      if x == 0:
          plt.annotate(y, (x-0.15, y), color = 'red')
      else:
          plt.annotate(y, (x-0.15, y))

  plt.title('2021년 6월 평균 하차 인원 수 Top10')
  plt.show()
  
  **특정 호선의 혼잡 정도와 위치좌표 데이터 병합**
  #특정 호선의 역별 평균 승하차 인원 수와 지하철 역 위치 좌표를 데이터프레임으로 반환하는 함수입니다.
  def get_nums_and_location(line, metro_st):
      
    # 특정 호선의 데이터만 추출합니다.
      metro_line_n = metro_st[metro_st['호선명']==line] 
      
      # 승차 인원 컬럼만 추출합니다.
      metro_get_on = pd.DataFrame()
      metro_get_on['지하철역'] = metro_line_n['지하철역']
      for i in range(int((len(metro_recent.columns)-3)/2)):
          metro_get_on[metro_line_n.columns[3+2*i]] = metro_line_n[metro_line_n.columns[3+2*i]]
      metro_get_on = metro_get_on.set_index('지하철역')
      
      # 하차 인원 컬럼만 추출합니다.
      metro_get_off = pd.DataFrame()
      metro_get_off['지하철역'] = metro_line_n['지하철역']
      for i in range(int((len(metro_recent.columns)-3)/2)):
          metro_get_off[metro_line_n.columns[4+2*i]] = metro_line_n[metro_line_n.columns[4+2*i]]
      metro_get_off = metro_get_off.set_index('지하철역')
      
      # 역 별 평균 승하차 인원을 구한 후 정수로 형 변환하여 데이터프레임으로 저장합니다.
      df = pd.DataFrame(index = metro_line_n['지하철역'])
      df['평균 승차 인원 수'] = metro_get_on.mean(axis=1).astype(int)
      df['평균 하차 인원 수'] = metro_get_off.mean(axis=1).astype(int)
      
      # 지하철역 명 동일하도록 설정합니다. -> 두 데이터의 지하철명을 통일
      temp = []
      df = df.reset_index() 
      for name in df['지하철역']:
          temp.append(name.split('(')[0]+'역') -> 괄호를 기준으로 나누고, 앞에 데이터에다 '역' 추가
      df['지하철역'] = temp
      
      # 지하철역 명을 기준으로 두 데이터프레임 병합합니다.
      df = df.merge(subway_location, left_on='지하철역', right_on='지하철역') -> df를 subway_location데이터와 병합할건데, 왼쪽 테이블은 '지하철역' 기준으로, 오른쪽 테이블은 '지하철역' 기준으로 병합함.
      
      return df
      
**특정 호선의 혼잡 정도를 지도에 출력**
import folium
#특정 위도, 경도 중심으로 하는 OpenStreetMap을 출력
map_osm = folium.Map(location = [37.529622, 126.984307], zoom_start=12)
map_osm

#특정 호선의 역별 평균 승하차 인원 수와 위치좌표 데이터만 추출합니다.
rail = '6호선'
df = get_nums_and_location(rail, metro_st)

#서울의 중심에 위치하는 명동역의 위도와 경도를 중심으로 지도 출력합니다.
latitude = subway_location[subway_location['지하철역']=='명동역']['x좌표']
longitude = subway_location[subway_location['지하철역']=='명동역']['y좌표']
map_osm = folium.Map(location = [latitude, longitude], zoom_start = 12)

#각 지하철 역의 위치별로 원형마커를 지도에 추가합니다.
for i in df.index:
    marker = folium.CircleMarker([df['x좌표'][i],df['y좌표'][i]],
                        radius = (df['평균 승차 인원 수'][i]+1)/3000, # 인원 수가 0일 때 계산오류 보정
                        popup = [df['지하철역'][i],df['평균 승차 인원 수'][i]], 
                        color = 'blue', 
                        fill_color = 'blue')
    marker.add_to(map_osm)

map_osm

 # 자동차 리콜 데이터 분석
 import numpy as np 
 import pandas as pd 
 import matplotlib.pyplot as plt
 !pip install seaborn==0.9.0. -> seaborn의 버전이 0.9.0인지 확인.
 import seaborn as sns
 print(sns.__version__)
 #missingno라는 라이브러리가 설치되어 있을 경우 import
 try: 
     import missingno as msno
 #missingno라는 라이브러리가 설치되어 있지 않을 경우 설치 후 import
 except: 
     !pip install missingno
     import missingno as msno
 
 상위 5개 데이터를 불러오는 것은 df.head()로 하지만, 하위 5개 데이터를 불러오는 것은 df.tail()로 불러옴.

1. 데이터 정제
**결측치 확인**
missingno.matrix() 함수를 이용해서 결측치를 시각화할 수 있음.
sns.set(font="NanumBarunGothic", 
rc={"axes.unicode_minus":False}) -> seaborn의 set함수를 이용해서 폰트를 설정해주고, rc에서 axes.unicode_minus로 - 기호가 깨질 가능성을 예방함.
msno.matrix(df) -> 결측치가 있는 부분은 흰색으로 줄이 가게 나타남. 
plt.show() -> 해당 데이터에선 모두 검은색으로, 흰색 실선이 없음. 즉, 결측치가 없음.

각 열 별로 결측치의 갯수를 반환함.
df.isna().sum() ->결측치면 1을 반환, 아니면 0을 반환하기 때문에 sum()으로 각 컬럼의 결측치 갯수를 알 수 있음.

**중복값 확인**
duplicated() 함수를 이용해서 중복값(모든 컬럼의 값이 모두 일치하는 행이 두 개 이상 있으면 중복값이라고 함)을 확인할 수 있음.
df[df.duplicated(keep=False)] -> 중복값을 출력해줌. keep: first(default)는 처음 발견된 데이터를 제외하고, 중복데이터를 마크하거나 삭제함, last는 마지막으로 발견된 데이터를 제외하고 중복데이터를 마크하거나 삭제함, false는 모든 중복데이터를 마크하거나 삭제함.
df = df.drop_duplicates() -> 중복값 삭제!

**기초적인 데이터 변형**
생산기간, 생산기간.1, 리콜개시일 열이 모두 object타입, 즉 문자열로 인식되고 있음. 분석을 위해 연도, 월, 일을 각각 정수형으로 저장함.
추가적으로 분석의 편리를 위해서 열 이름은 영어로 변경
def parse_year(s):
    return int(s[:4])
def parse_month(s):
    return int(s[5:7])
def parse_day(s):
    return int(s[8:])

#Pandas DataFrame에서는 row별로 loop를 도는 것이 굉장히 느리기 때문에, apply() 함수를 이용하여 벡터 연산을 진행합니다.
df['start_year'] = df['생산기간'].apply(parse_year)
df['start_month'] = df['생산기간'].apply(parse_month)
df['start_day'] = df['생산기간'].apply(parse_day)

df['end_year'] = df['생산기간.1'].apply(parse_year)
df['end_month'] = df['생산기간.1'].apply(parse_month)
df['end_day'] = df['생산기간.1'].apply(parse_day)

df['recall_year'] = df['리콜개시일'].apply(parse_year)
df['recall_month'] = df['리콜개시일'].apply(parse_month)
df['recall_day'] = df['리콜개시일'].apply(parse_day)
-> 생산기간, 생산기간.1, 리콜개시일이 yyyy-mm-dd 형식으로 저장되어 있는 것을 yyyy를 int형으로 바꿔주고, mm을 int형으로, dd를 int형으로 바꿔서 영어 컬럼명으로 변경해서 추가해줌. int형으로 바꿔주는 함수를 연,월,일별로 선언해서 apply함수를 이용해서 각 컬럼값에 적용하고, 새로 추가할 영어 칼럼에 추가해줌.

df.=df.drop(columns=['생산기간','생산기간.1','리콜개시일']).rename(columns={'제작자':'manufacturer','차명':'model','리콜사유':'cause'})
-> drop함수로 불필요한 열을 버리고, rename 함수로 열 이름을 재정의 함. 

본 분석에선  2020년의 데이터만을 대상으로 하므로, 그 외에 데이터가 있다면 삭제해줘야 함.
df.recall_year.min(),df.recall_year.max() -> recall_year 컬럼값을 기준으로 최대, 최소를 출력
df=df[df['recall_year']==2020] -> recall_year가 2020년인 데이터만을 남겨줌

2. 데이터 시각화
**제조사별 리콜 현황 출력**
제조사별 리콜 건수 분포를 막대그래프로 확인.
df.groupby("manufacturer").count()['model'].sort_values(ascending=False) -> manufacturer을 기준으로 그룹으로 묶어주고, 각 manufacturer의 model 개수를 세서, 내림차순으로 정렬함. groupby하면 해당 컬럼이 인덱스가 되는듯?
pd.DataFrame(df.groupby("manufacturer").count()["model"].sort_values(ascending=False)).rename(columns={"model": "count"}) -> 깔끔하게 볼 수 있도록 데이터프레임으로 변경하고, model 컬럼명을 count로 바꿔줌.
tmp = pd.DataFrame(df.groupby("manufacturer").count()["model"].sort_values(ascending=False)).rename(columns={"model": "count"}) -> order를 지정해주기 위해서 tmp변수 사용
plt.figure(figsize=(20,10))
#한글 출력을 위해서 폰트 옵션을 설정합니다.
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="manufacturer", data=df, palette="Set2", order=tmp.index)
plt.xticks(rotation=270) -> xtick에 해당하는 값들이 수평으로 나오면 인덱스를 알아보기 힘들기 때문에 다음과 같이 변경
plt.show()

**모델별 리콜 현황 출력**
pd.DataFrame(df.groupby("model").count()["start_year"].sort_values(ascending=False)).rename(columns={"start_year": "count"}).head(10) -> 모델별로 count(임의의 칼럼으로 카운팅해주면 됨)를 해서 내림차순으로 정렬하고, start_year를 count로 바꿔서 데이터프레임으로 만들어줌.
#모델이 굉장히 많기 때문에 상위 50개 모델만 뽑아서 시각화를 진행.
tmp = pd.DataFrame(df.groupby("model").count()["manufacturer"].sort_values(ascending=False))
tmp = tmp.rename(columns={"manufacturer": "count"}).iloc[:50] -> tmp에 데이터프레임을 저장하고, iloc함수를 이용해서 인덱스 기준으로 슬라이싱해서 저장함. -값도 적용가능한데, -1이면 뒤에서 첫번째, -2면 뒤에서 두번째 등으로 표현.
#그래프의 사이즈를 조절합니다.
plt.figure(figsize=(10,5))

#seaborn의 countplot 함수를 사용하여 출력합니다.
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
        #df['model']=df.model은 같은 결과를 줌
        ax = sns.countplot(x="model", data=df[df.model.isin(tmp.index)], palette="Set2", order=tmp.index) -> df에 model 컬럼에 접근해서, model명이 상위 50개를 포함한 tmp안에 있는 애들만 가져와라.
plt.xticks(rotation=270)
plt.show()

**월별 리콜 현황 출력**
pd.DataFrame(df.groupby("recall_month").count()["start_year"].sort_values(ascending=False)).rename(columns={"start_year": "count"}) -> recall_month는 인덱스로 들어감!!! 이걸 해제하고, 컬럼으로 취급해주려면 reset_index를 사용하면 됨.
#그래프의 사이즈를 조절합니다.
plt.figure(figsize=(10,5))

#seaborn의 countplot 함수를 사용하여 출력합니다.
sns.set(style="darkgrid")
ax = sns.countplot(x="recall_month", data=df, palette="Set2")
plt.show()

**생산연도별 리콜 현황 출력**
tmp = pd.DataFrame(df.groupby("start_year").count()["model"]).rename(columns={"model": "count"}).reset_index()

#그래프의 사이즈를 조절합니다.
plt.figure(figsize=(10,5))

#seaborn의 lineplot 함수를 사용하여 출력합니다.
sns.set(style="darkgrid")
sns.lineplot(data=tmp, x="start_year", y="count")
plt.show()

**4분기 제조사별 리콜 현황 출력**
df[df.recall_month.isin([10,11,12])].head() -> 4분기인 10월, 11월, 12월 리콜 현황 데이터를 뽑아볼 수 있으며, df.recall_month가 10,11,12 중 하나인 데이터들을 뽑아내고 상위 5개를 출력함.
plt.figure(figsize=(20,10))
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="manufacturer", data=df[df.recall_month.isin([10,11,12])], palette="Set2") -> 제조사별로 4분기에 해당하는 리콜 건 수를 계산해서 그래프로 출력
plt.xticks(rotation=270)
plt.show()

**하반기(7-12월) 생산연도별 리콜 현황 출력**
df[df.recall_month>=7].head() -> 해당 컬럼을 지정하여 시리즈 형태로 출력할 수 있음. 
plt.figure(figsize=(10,5))
sns.set(style="darkgrid")
ax = sns.countplot(x="start_year", data=df[df.recall_month>=7], palette="Set2")
plt.show()

**워드 클라우드를 이용한 리콜 사유 시각화** 
워드 클라우드 생성을 도와주는 패키지를 가져옴.
try:
    from wordcloud import WordCloud, STOPWORDS
except:
    !pip install wordcloud
    from wordcloud import WordCloud, STOPWORDS 
#문법적인 성분들을 배제하기 위해 stopwords들을 따로 저장해둡니다.
set(STOPWORDS) -> 영어를 사용할 땐 상관없지만, 한글을 쓸 땐 적합하지 않음. 예시로 몇 개의 stopword들을 수기로 저장해보자.
#손으로 직접 리콜 사유와 관련이 적은 문법적 어구들을 배제해보겠습니다.
spwords = set(["동안", "인하여", "있는", "경우", "있습니다", "가능성이", "않을", "차량의", "가", "에", "될", "이",
               "인해", "수", "중", "시", "또는", "있음", "의", "및", "있으며", "발생할", "이로", "오류로", "해당"])
               # 리콜 사유에 해당하는 열의 값들을 중복 제거한 뒤 모두 이어붙여서 text라는 문자열로 저장합니다.
               text = ""
               for c in df.cause.drop_duplicates(): -> drop_duplicates()는 중복값을 처리해주며, 1개의 단일 값만 남기고 나머지 중복값은 모두 제거함.
                   text += c
               text[:100]
 #한글을 사용하기 위해서는 폰트를 지정해주어야 합니다.
 wc1 = WordCloud(max_font_size=200, stopwords=spwords, font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                 background_color='white', width=800, height=800) -> 워드클라우드 생성
 wc1.generate(text) -> 워드클라우드를 text에 적용

 plt.figure(figsize=(10, 8))
 plt.imshow(wc1)
 plt.tight_layout(pad=0)
 plt.axis('off')
 plt.show()
 
 # 유가 데이터 분석하기
 전국 주유소 10000개의 번호와 기간, 지역, 상표, 셀프여부, 휘발유 가격 등이 컬럼으로 구성되어 있으며 7일간의 조사 과정때문에 한 주유소당 7개의 행을 차지하고 있음.
 import numpy as np 
 import pandas as pd 
 import seaborn as sns
 sns.set_style('darkgrid') -> 그래프가 훨씬 예쁘기 때문에 추가해서 사용해주는 것이 좋음
 import matplotlib.pyplot as plt
 import matplotlib.font_manager as fm

 font_dirs = ['/usr/share/fonts/truetype/nanum', ]
 font_files = fm.findSystemFonts(fontpaths=font_dirs)
 for font_file in font_files:
     fm.fontManager.addfont(font_file)

 plt.rcParams['font.family'] = 'NanumBarunGothic'
 plt.rcParams['axes.unicode_minus']=False
 
f18 = pd.read_csv(f'./data/과거_판매가격(주유소)_2018.csv')
#0번 row 제거
f18 = f18.drop(0) 
#변수별 null값 확인 결과 null 없음
f18.isna().sum() 
#include='all': 카테고리형 변수도 정보 제공
f18.describe(include='all') -> describe(include='all')을 이용해서 데이터에 대한 다양한 정보를 얻을 수 있음. unique 번호의 개수 등을 통해서 데이터의 개수를 알 수 있고, 다음과 같이 데이터를 가공해야 할 부분을 추측해볼 수 있음. 
describe 점검 포인트:
unique 번호가 11673개이며 최대 7번까지 기록되었음
기간이 수치로 인식되고 있음
unique 지역 개수가 229이어서 너무 많음
unique 상표 개수가 9개이므로 적절함
unique 셀프여부 개수가 2개이며, 셀프여부는 각각 절반정도 비중을 차지함
휘발유 min이 0임

이러한 점검포인트를 바탕으로 다음과 같이 데이터를 정제함
1. 기간을 datetime 형태로 변환
f18['기간'] = f18['기간'].apply(lambda x:pd.to_datetime(str(int(x)))) 
2. 지역 변수 중 첫 지역 구분만 컬럼 형성 즉, 충북 청주시 -> 충북. 모든 행이 oo xx시 이렇게 지역구분이 된 것인지 모르기 때문에, len(x.split())으로 공백기준으로 잘라주고 값을 확인.
region_len = f18['지역'].apply(lambda x: len(x.split())) 
print(f"min: {min(region_len)},max: {max(region_len)}")
-> 확인해보니 딱 한 개의 데이터만 x가 1임. 데이터 분석에 크게 문제가 없으므로 생략
f18['지역2'] = f18['지역'].apply(lambda x:x.split()[0])
import collections
collections.Counter(f18['지역2'])
3. 휘발유 값이 0인 row삭제
f18 = f18.loc[f18['휘발유']!=0,:]
4. 주유소별 데이터 정합성 확인(7일동안 변화 없었다는 전제)
unique_count = f18.groupby('번호')[['지역','상표','셀프여부']].nunique() -> 번호로 그룹바이해본 후에 지역, 상표, 셀프여부의 unique한 값 개수를 찾아봄.
unique_count.head()
target = unique_count.loc[(unique_count!=1).sum(axis=1)!=0] -> 행 별로 값이 한 개라도 1이 아니면 1 반환 => 어렵다!!!!!
target
f18 = f18.loc[~f18['번호'].isin(target.index)] -> 편의를 위해서 target에 포함된 데이터는 삭제. ~을 이용해서 target index에 포함되지 않는 값들만 가져옴.
5. 주유소별 데이터 통합
f18 = f18.groupby('번호')\ -> 번호를 기준으로 그룹바이 하는데.
.agg({'지역':'first','지역2':'first','상표':'first','셀프여부':'first','휘발유':'mean'})\ -> 지역, 지역2, 상표, 셀프여부는 모두 첫번째 값으로 그룹바이하고, 휘발유는 평균값으로 그룹바이
.reset_index()

모든 년도의 데이터들을 위 과정과 같이 cleansing 및 feature engineering 함수 생성 및 전체 년도에 적용
def preprocess(df):
df_copy=df.copy() # 필터링 전

df = df.drop(0)
df['기간'] = df['기간'].apply(lambda x:pd.to_datetime(str(int(x))))
df['지역2'] = df['지역'].apply(lambda x:x.split()[0])
df = df.loc[df['휘발유']!=0,:]
unique_count = df.groupby('번호')[['번호','지역','상표','셀프여부']].nunique()
target = unique_count.loc[(unique_count!=1).sum(axis=1)!=0,:]
df = df.loc[~df['번호'].isin(target.index),:]
df = df.groupby('번호')\
    .agg({'지역':'first','지역2':'first','상표':'first','셀프여부':'first','휘발유':'mean'})\
    .reset_index()

out = set(df_copy['번호']).difference(set(df['번호'])) # 필터링 후 -> 원본 데이터의 번호 컬럼의 값과 필터링한 df 데이터의 번호 컬럼을 차집합해서 저장.
return(df,out) -> 필터링한 데이터인 df와 이상한 데이터 값이 들어있었던 out을 반환

f_dict = dict()
out_all = set() # 이상치 발견한 주유소 번호 저장
for year in range(2018,2022):
    df = pd.read_csv(f'./data/과거_판매가격(주유소)_{year}.csv')
    f_dict[year], out = preprocess(df) -> key값이 year이고, value가 해당 년도의 주유소 데이터인 딕셔너리 
    out_all.update(out)

**연도별 데이터 outer join**
데이터프레임에서 외부조인을 하려면 키값을 설정해줘야 하는데, 모든 컬럼에서 휘발유만 빼고 모두 키값으로 설정해줌. 즉, 번호, 지역, 지역2, 상표, 셀프여부가 모두 일치할 때 년도별 휘발유값을 조인해서 하나의 행을 나타내는 것.
key = list(f_dict[2018].columns)
key.remove('휘발유')
m1 = pd.merge(f_dict[2018],f_dict[2019],on=key,how='outer',suffixes=('_2018', '_2019')) -> join 방식에는 대표적으로 inner(키 값 기준 양쪽 데이터프레임의 교집합을 타냄.), left, outer(양쪽 데이터프레임 기준 합집합을 나타내는 것임) 등이 있는데 suffixes는 양쪽 데이터프레임에 같은 키값이 있는 경우 두 키 값을 어떻게 구분할지 나타내는 것임.
m2 = pd.merge(f_dict[2020],f_dict[2021],on=key,how='outer',suffixes=('_2020', '_2021'))
m = pd.merge(m1,m2,on=key,how='outer')
m.groupby('번호').size().sort_values(ascending=False).head() -> 각 행의 개수를 나타내는 size로 agg하고 내림차순으로 조회하면 끝자리가 752인 데이터는 행이 4개나 됨.
m.loc[m['번호']=='A0019752'] -> 그 이유는 상표가 매년 바뀌었기 때문
(m.groupby('번호').size()>1).sum() -> 이런 경우가 얼마나 많은지 조회하기 위해서 행의 개수가 여러개인 번호를 조회해보니까 1338개나 됨. 개수가 너무 많아서 상표를 키값에서 제외해주기로 함.
key.remove('상표')
m1 = pd.merge(f_dict[2018],f_dict[2019],on=key,how='outer',suffixes=('_2018', '_2019'))
m2 = pd.merge(f_dict[2020],f_dict[2021],on=key,how='outer',suffixes=('_2020', '_2021'))
m = pd.merge(m1,m2,on=key,how='outer')
size = m.groupby('번호').size().sort_values(ascending=False)
size.head() -> 결과, 여전히 4개의 번호가 행의 개수가 2개임.
target = size[size>1].index
m.loc[m['번호'].isin(target)].sort_values('번호')
m = m.loc[~m['번호'].isin(target)] -> 그래서 이러한 번호들은 데이터에서 제외하기로 함.
m.groupby('번호').size().sort_values(ascending=False).head()
#이상치 발견되었던 주유소 필터링
m = m.loc[[x not in out_all for x in m['번호']]] -> [x not in out_all for x in m['번호']]가 list comprehension임. m['번호'] 안에 있는 각각의 값들을 x라고 하며, x가 out_all 안에 있는 값이면 false, 들어있지 않으면 true를 반환해서 들어있지 않은 값들만 m에 남김.

**연도별 개폐업 수치 분석**
id_dict=dict()
for year in range(2018,2022):
    id_dict[year] = set(m.loc[~m[f'상표_{year}'].isna()]['번호'].unique()) -> <~m[f'상표_{year}'].isna()> 로 nan이 없는 각 연도별 상표만을 가진 번호의 유니크값을 추출해서 집합으로 저장
    diff_dict=dict()
    for year in range(2018,2021):
    opened = len(id_dict[year+1].difference(id_dict[year])) -> 19년에는 존재하지만, 18년에는 존재하지 않는 주유소. 즉, 오픈한 주유소 추출
    closed = len(id_dict[year].difference(id_dict[year+1])) -> 20년에는 존재하지 않지만 19년에는 존재하는 주유소. 즉, 폐업한 주유소 추출
        diff_dict[f'{year}_{year+1}']=[opened,closed]
    diff_df = pd.DataFrame(diff_dict,index=['OPENED','CLOSED'])  
    diff_df.plot() -> 인덱스와 컬럼이 뒤집어져서 2018_2019, 등등이 인덱스로, opened, closed가 컬럼으로 인식될 수 있음. 이러한 경우, 
diff_df.T.plot(color=['r','b']) -> 전치행렬화해서 제대로 된 그래프를 출력 가능

**2020년에 신규개업한 셀프 주유소의 개수**
id_dict=dict()
for year in range(2018,2022):
    id_dict[year] = set(m.loc[(~m[f'상표_{year}'].isna())&(m['셀프여부']=='셀프')]['번호'].unique()) -> 상표 컬럼 값이 nan이 아니면서 셀프여부 컬럼 값이 셀프인 데이터의 번호의 유니크값을 추출해서 집합으로 저장
diff_dict=dict()
for year in range(2018,2021):
    opened = len(id_dict[year+1].difference(id_dict[year]))
    closed = len(id_dict[year].difference(id_dict[year+1]))
    diff_dict[f'{year}_{year+1}']=[opened,closed]
diff_df = pd.DataFrame(diff_dict,index=['OPENED','CLOSED'])    

데이터프레임에서 행 인덱스로 접근할 때는 iloc나 loc를 사용하고, 열 인덱스로 접근하거나 boolean 인덱스는 바로 df[''] 식으로 접근해줄 수 있음. 
특정 행과 열을 모두 접근할 때는 iloc나 loc로 loc['행','열']과 같이 접근할 수 있음.

**브랜드 분석 : 브랜드별 가격경쟁력 및 시장점유율 분석**
**주요 브랜드별 가격 Line Plot 분석**
brand_price_dict=dict()
for year in range(2018,2022):
brand_price_dict[str(year)]=m.groupby(f'상표_{year}')[f'휘발유_{year}'].mean() -> 상표_년도별로 그룹바이를 해주고, 휘발유_년도를 평균값으로 나타냄. 예를 들어, gs칼텍스_2018을 가진 데이터를 묶어주면서,  묶어주는 데이터들의 휘발유_2018의 평균값을 내서 묶음.
brand_price_df = pd.DataFrame(brand_price_dict)
brand_price_df = brand_price_df.drop('SK가스') -> nan값 제거
brand_price_df.T.plot(figsize=(10,5))

**주요 브랜드별 지난 4년간 시장 점유율 stacked bar plot 및 heatmap**
brand_share_dict=dict()
for year in range(2018,2022):
    brand_share_dict[str(year)]=m.groupby(f'상표_{year}').size() -> 상표_년도 별로 묶어서 그 값을 가진 행들의 개수 즉, size를 딕셔너리에 저장
    brand_share_df = brand_share_df.drop('SK가스')
    brand_ratio_df = brand_share_df.apply(lambda x:x/brand_share_df.sum(),axis=1) -> 열 기준으로 값들을 다 더해서 x값 나눠서 점유율 구함.
    brand_ratio_df = brand_ratio_df.sort_values('2018',ascending=False)
    brand_ratio_df.T.plot(kind='bar',**stacked=True**,rot=0,figsize=(10,5))
    plt.legend(bbox_to_anchor=(1, 1))  -> rot는 rotation, bbox_to_anchor는 그래프 밖에 사이즈를 설정해서 범례를 표시할 수 있음.
    plt.figure(figsize=(10,5))
    sns.**heatmap**(brand_ratio_df, cmap= 'RdBu_r', linewidths=1, linecolor='black',annot=True) -> 각 heat들 사이에 라인의 굵기와 컬러 지정가능, annot로 각 값을 히트 위에 표기 또는 미표기
    
**2019년 주유소를 셀프 및 일반 주유소로 구분하고 일반 주유소가 차지하는 비율을 구하시오.**
self_share_dict = m.loc[~m['상표_2019'].isna()].groupby('셀프여부').size() -> 상표_2019의 값이 nan이 아닌 데이터들을 셀프여부로 그룹바이하고, 각 행의 개수를 저장
self_ratio_dict = self_share_dict/self_share_dict.sum()



