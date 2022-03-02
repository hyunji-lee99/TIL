### 문제 1)
수많은 마라톤 선수들이 마라톤에 참여하였습니다. 단 한 명의 선수를 제외하고는 모든 선수가 마라톤을 완주하였습니다.

마라톤에 참여한 선수들의 이름이 담긴 배열 participant와 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때, 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.

제한사항)
- 마라톤 경기에 참여한 선수의 수는 1명 이상 100,000명 이하입니다.
- completion의 길이는 participant의 길이보다 1 작습니다. -> 한 명만 완주를 못했다는 뜻
- 참가자의 이름은 1개 이상 20개 이하의 알파벳 소문자로 이루어져 있습니다.
- 참가자 중에는 동명이인이 있을 수 있습니다.

입출력 예시
participant &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; completion	      &nbsp; &nbsp; &nbsp;return
["leo", "kiki", "eden"]	["eden", "kiki"]	"leo"
["marina", "josipa", "nikola", "vinko", "filipa"]	["josipa", "filipa", "marina", "nikola"]	"vinko"
["mislav", "stanko", "mislav", "ana"]	["stanko", "ana", "mislav"]	"mislav"

```
#counter(participant)에서 counter(completion)를 빼서 value가 0이 아닌 것은 완주못한 것.

from collections import Counter
# 배열 요소 개 수를 세서 dictionary로 리턴해줌

def solution(participant, completion):
    count_part=Counter(participant)
    count_comp=Counter(completion)
    #완주 실패 명단
    not_comp=''

    for key in count_comp:
    #count_comp의 key을 기준으로 순회
        temp=count_part[key]-count_comp[key]
        count_part[key]=temp

    for key in count_part:
        if count_part[key]!=0:
            not_comp=key
    answer=not_comp
    return answer
```

다른 사람의 효율적인 풀이)
for문을 써서 돌릴 필요없이 Counter 결과를 빼서 keys()로 명단 리턴해주면 됨.
```
import collections

def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]
```

### 문제 2)
전화번호부에 적힌 전화번호 중, 한 번호가 다른 번호의 접두어인 경우가 있는지 확인하려 합니다.
전화번호가 다음과 같을 경우, 구조대 전화번호는 영석이의 전화번호의 접두사입니다.

구조대 : 119
박준영 : 97 674 223
지영석 : 11 9552 4421
전화번호부에 적힌 전화번호를 담은 배열 phone_book 이 solution 함수의 매개변수로 주어질 때, 어떤 번호가 다른 번호의 접두어인 경우가 있으면 false를 그렇지 않으면 true를 return 하도록 solution 함수를 작성해주세요.

제한 사항
- phone_book의 길이는 1 이상 1,000,000 이하입니다.
- 각 전화번호의 길이는 1 이상 20 이하입니다.
- 같은 전화번호가 중복해서 들어있지 않습니다.

입출력 예제
phone_book	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; return
["119", "97674223", "1195524421"]	false
["123","456","789"]	true
["12","123","1235","567","88"]	false

```
#phone_book[i]가 다음 요소에 포함되는지 확인하고, 그 원소가 인덱스 0부터 phone_book[i]의 길이만큼을 차지하는지 확인.
def solution(phone_book):
    ans=True
    phone_book.sort()
    for i in range(len(phone_book)-1):
        if phone_book[i] in phone_book[i+1][0:len(phone_book[i])]:
            ans=False
            break
    answer=ans
    return answer
```

다른 사람의 효율적인 풀이)
여러 개의 순회 가능한 객체를 인자로 받고, 각 개체가 담고있는 원소를 튜플의 형태로 차례로 접근할 수 있는 반복자를 반환하는 zip 함수,
https://ooyoung.tistory.com/60
문자열 인스턴스가 지정한 문자로 시작하는지 확인하는 startswitch 함수를 사용하면 훨씬 빠르고 간결하게 해결할 수 있음.
https://security-nanglam.tistory.com/429
```
def solution(phoneBook):
    phoneBook = sorted(phoneBook)

    for p1, p2 in zip(phoneBook, phoneBook[1:]):
        if p2.startswith(p1):
            return False
    return True
```

### 문제 3)
스파이들은 매일 다른 옷을 조합하여 입어 자신을 위장합니다.

예를 들어 스파이가 가진 옷이 아래와 같고 오늘 스파이가 동그란 안경, 긴 코트, 파란색 티셔츠를 입었다면 다음날은 청바지를 추가로 입거나 동그란 안경 대신 검정 선글라스를 착용하거나 해야 합니다.

종류	이름
얼굴	동그란 안경, 검정 선글라스
상의	파란색 티셔츠
하의	청바지
겉옷	긴 코트

스파이가 가진 의상들이 담긴 2차원 배열 clothes가 주어질 때 서로 다른 옷의 조합의 수를 return 하도록 solution 함수를 작성해주세요.

제한사항
- clothes의 각 행은 [의상의 이름, 의상의 종류]로 이루어져 있습니다.
- 스파이가 가진 의상의 수는 1개 이상 30개 이하입니다.
- 같은 이름을 가진 의상은 존재하지 않습니다.
- clothes의 모든 원소는 문자열로 이루어져 있습니다.
- 모든 문자열의 길이는 1 이상 20 이하인 자연수이고 알파벳 소문자 또는 '_' 로만 이루어져 있습니다.
- 스파이는 하루에 최소 한 개의 의상은 입습니다.

입출력 예
clothes	return
```
[["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]]	5
[["crowmask", "face"], ["bluesunglasses", "face"], ["smoky_makeup", "face"]]	 3
```

입출력 예 설명
예제 #1
headgear에 해당하는 의상이 yellow_hat, green_turban이고 eyewear에 해당하는 의상이 blue_sunglasses이므로 아래와 같이 5개의 조합이 가능합니다.
1. yellow_hat
2. blue_sunglasses
3. green_turban
4. yellow_hat + blue_sunglasses
5. green_turban + blue_sunglasses

예제 #2
face에 해당하는 의상이 crow_mask, blue_sunglasses, smoky_makeup이므로 아래와 같이 3개의 조합이 가능합니다.
1. crow_mask
2. blue_sunglasses
3. smoky_makeup

```
def solution(clothes):
    dict={}
    mul=1
    #종류별로 분류해서 딕셔너리에 저장
    for cloth in clothes:
        if cloth[1] in dict:
            dict[cloth[1]].append(cloth[0])
        else: dict[cloth[1]]=[cloth[0]]

    for c in dict:
        #(선택할 수 있는 종류 + 선택안하는 경우)*(기존 조합가능한 개수)
        mul=mul*(len(dict[c])+1)
    #아무것도 선택안하는 경우 제외
    answer = mul-1
    return answer
```

### 문제 4)
스트리밍 사이트에서 장르 별로 가장 많이 재생된 노래를 두 개씩 모아 베스트 앨범을 출시하려 합니다. 노래는 고유 번호로 구분하며, 노래를 수록하는 기준은 다음과 같습니다.
1. 속한 노래가 많이 재생된 장르를 먼저 수록합니다.
2. 장르 내에서 많이 재생된 노래를 먼저 수록합니다.
3. 장르 내에서 재생 횟수가 같은 노래 중에서는 고유 번호가 낮은 노래를 먼저 수록합니다.

노래의 장르를 나타내는 문자열 배열 genres와 노래별 재생 횟수를 나타내는 정수 배열 plays가 주어질 때, 베스트 앨범에 들어갈 노래의 고유 번호를 순서대로 return 하도록 solution 함수를 완성하세요.

제한사항
- genres[i]는 고유번호가 i인 노래의 장르입니다.
- plays[i]는 고유번호가 i인 노래가 재생된 횟수입니다.
- genres와 plays의 길이는 같으며, 이는 1 이상 10,000 이하입니다.
- 장르 종류는 100개 미만입니다.
- 장르에 속한 곡이 하나라면, 하나의 곡만 선택합니다.
- 모든 장르는 재생된 횟수가 다릅니다.

입출력 예
genres	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;plays	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return
["classic", "pop", "classic", "classic", "pop"]	[500, 600, 150, 800, 2500]	[4, 1, 3, 0]

입출력 예 설명
classic 장르는 1,450회 재생되었으며, classic 노래는 다음과 같습니다.

고유 번호 3: 800회 재생
고유 번호 0: 500회 재생
고유 번호 2: 150회 재생
pop 장르는 3,100회 재생되었으며, pop 노래는 다음과 같습니다.

고유 번호 4: 2,500회 재생
고유 번호 1: 600회 재생
따라서 pop 장르의 [4, 1]번 노래를 먼저, classic 장르의 [3, 0]번 노래를 그다음에 수록합니다.

```
def solution(genres, plays):
    #genres, plays, index 결합한 collection 배열 생성
    collection=[]
    #genre별 재생횟수 order dictionary
    order={}
    i=0
    for g,p in zip(genres,plays):
        collection.append((g,p,i))
        if g in order:
            order[g]=order[g]+p
        else: order[g]=p
        i=i+1
    order=dict(sorted(order.items(),key=lambda x:x[1],reverse=True))

    orderedSong={}
    for temp in collection:
        if temp[0] in orderedSong:
            orderedSong[temp[0]].append([temp[1],temp[2]])
        else:
            orderedSong[temp[0]]=([[temp[1],temp[2]]])

    #orderedSong을 temp[1] 기준으로 내림차순으로 정렬하면, temp[1]이 같은 애들끼리는 temp[2] 오름차순으로 정렬이 되기 때문에 파이썬 다중정렬 방식을 이용해서 temp[1]->내림착순, temp[2]->오름차순으로 정렬해줘야 함.
    for key in orderedSong.keys():
        orderedSong[key]=sorted(orderedSong[key],key=lambda x:(-x[0],x[1]))

    orderedArray=[]
    for genre in order.keys():
        if len(orderedSong[genre])==1:
            orderedArray.append(orderedSong[genre][0][1])
        else:
            orderedArray.append(orderedSong[genre][0][1])
            orderedArray.append(orderedSong[genre][1][1])
    answer = orderedArray
    return answer
```
파이썬 다중정렬 방법
https://dailyheumsi.tistory.com/67
