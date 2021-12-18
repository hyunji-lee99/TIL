# 의류 판매 상품 리뷰 분석을 통한 상품 추천 여부 예측
## 프로젝트 목표 - Kaggle에서 제공하는 여성 의류 이커머스 데이터 바탕
1. 상품 리뷰 데이터 분석을 통해 상품 추천 여부를 예측하는 분류 모델 수행
2. 상품 추천 여부에 영향을 미치는 특성 데이터들에 대한 데이터 분석 수행

**데이터 읽기**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_origin = pd.read_csv("./data/Womens Clothing E-Commerce Reviews(수정).csv")
df_origin.head()
df_origin.info()
#수치형 변수의 데이터 정보를 요약하여 출력합니다.
df_origin.describe()

**데이터 정제**
#결측값을 처리하기 전에 우선 의미 없는 변수인 'Unnamed: 0, Unnamed: 0.1'를 drop을 사용하여 삭제합니다.
df_clean = df_origin.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])
#결측값 정보를 출력합니다.
df_clean.isnull().sum()

=> Clothing ID                   0
Age                           0
Title                      3810
Review Text                 845
Rating                        0
Recommended IND               0
Positive Feedback Count       0
Division Name                14
Department Name              14
Class Name                   14
dtype: int64
#아래 3개의 변수들의 결측값 정보를 알아보고 싶어서 그 데이터들을 출력합니다.
df_clean[df_clean['Division Name'].isnull()]
#이번 실습에선 Revie Text에 있는 데이터만을 머신러닝 입력으로 사용할 것이기 때문에 Review Text의 결측값이 있는 샘플을 삭제합니다.
df_clean = df_clean[~df_clean['Review Text'].isnull()]
df_clean.isnull().sum()

**데이터 시각화**
#일반적으로 막대그래프를 그리는 방법으로 시각화를 수행하지만, 문자열로 이루어진 Title과 Review Text를 word cloud 방식을 사용해서 시각화를 수행하겠습니다.
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
**Title 워드클라우드**
#'Title'의 결측값을 삭제합니다.
df_clean_title = df_clean[~df_clean['Title'].isnull()]
#findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean_title['Title'].str.lower().str.cat(sep=' '))
#nltk에서 지원하는 'stopwords'를 다운받습니다.
nltk.download('stopwords')
#영어 'stopwords'를 가져옵니다.
en_stops = set(stopwords.words('english'))
#tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence
#출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)
#wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate
(' '.join(filtered_sentence))
#wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
**Review Text 워드클라우드**
#findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean['Review Text'].str.lower().str.cat(sep=' ')) 
#tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence
#출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)
#wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(filtered_sentence))
#wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()

#분포를 막대 그래프를 사용하여 출력합니다.
df_clean['Recommended IND'].value_counts().plot(kind='bar')
#분포를 도수분포표로 확인합니다.
df_clean['Recommended IND'].value_counts()

**데이터 전 처리**
#상품 추천 여부 예측을 수행하기 위해서 주어진 이커머스 데이터에 대해서 분류 모델을 사용할 것입니다.
#이번 실습에서는 sklearn에서 제공하는 TfidfVectorizer를 사용하여 문자열 데이터를 수치 자료형 벡터로 변환해 보겠습니다.
from sklearn.feature_extraction.text import TfidfVectorizer
#TfidfVectorizer을 불러옵니다. (stop_words 는 영어로 설정)
vectorizer = TfidfVectorizer(stop_words = 'english')
#소문자화 'Review Text'데이터를 Tfidf로 변환합니다.
X = vectorizer.fit_transform(df_clean['Review Text'].str.lower())
#변환된 X의 크기를 살펴봅니다.
X.shape
#예측해야 할 변수 'Recommended IND' 만을 선택하여 numpy 형태로 y에 저장합니다.
y = df_clean['Recommended IND']
y = y.to_numpy().ravel() # 1 차원 벡터 형태로 출력하기 위해 ravel 사용
vectorizer.get_feature_names()

from sklearn.model_selection import train_test_split
#sklearn에서 제공하는 train_test_split을 사용하여 손 쉽게 분리 할 수 있습니다.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

**기본 분류 모델 학습- 의사결정나무**
from sklearn.tree import DecisionTreeClassifier
#의사결정나무 DecisionTreeClassifier class를 가져 옵니다.
model = DecisionTreeClassifier()
#fit 함수를 사용하여 데이터를 학습합니다.
model.fit(x_train, y_train)
#score 함수를 사용하여 모델의 성능을 출력합니다.
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
**다양한 분류 모델 학습**
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

models = []
models.append(('KNN', KNeighborsClassifier()))  # KNN 모델
models.append(('NB-M', MultinomialNB()))  # 멀티노미얼 나이브 베이즈
models.append(('NB-B', BernoulliNB()))  # 베르누이 나이브 베이즈 모델
models.append(('RF', RandomForestClassifier()))  # 랜덤포레스트 모델
models.append(('SVM', SVC(gamma='auto')))  # SVM 모델
models.append(('XGB', XGBClassifier()))  # XGB 모델

for name, model in models:
    model.fit(x_train, y_train)
    msg = "%s - train_score : %f, test score : %f" % (name, model.score(x_train, y_train), model.score(x_test, y_test))
    print(msg)
    # xgb 모델에서 변수 중요도를 출력합니다.
    max_num_features = 20
    ax = xgb.plot_importance(models[-1][1], height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = max_num_features)
    ytick = ax.get_yticklabels()
    word_importance = []
    for i in range(max_num_features):
        word_importance.append(vectorizer.get_feature_names()[int(ytick[i].get_text().split('f')[1])])
    ax.set_yticklabels(word_importance)
    plt.rcParams['figure.figsize'] = (10, 15)
    plt.xlabel('The F-Score for each features')
    plt.ylabel('Importances')
    plt.show()
**평가 및 예측**
𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦=(𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠) / (𝑇𝑜𝑡𝑎𝑙 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠)
현재 데이터는 추천을 한다(0) 데이터가 추천 하지 않는다(1) 데이터에 비해 월등히 많은 상황임.
이런 경우, 추천한다(0)만을 정확히 예측해도 높은  정확도를 가질 수 있음.
때문에, 이번 실습에서는 또 다른 성능 지표인 recall값 또한 살펴봐야 함. 
recall 방식은 추천을 하지 않는다(1) 대비 추천한다(0)의 비율을 나타내기 때문에 정확도에서 놓칠 수 있는 결과 해석을 보충함. 
**Coufusion matrix**
from sklearn.metrics import confusion_matrix
#의사결정나무 모델에 confusion matrix를 사용하기 위하여 테스트 데이터의 예측값을 저장합니다.
model_predition = model.predict(x_test)
#sklearn에서 제공하는 confusion_matrix를 사용합니다.
cm = confusion_matrix(y_test, model_predition)
#출력 파트 - seaborn의 heatmap을 사용
plt.rcParams['figure.figsize'] = (5, 5)
sns.set(style = 'dark', font_scale = 1.4)
ax = sns.heatmap(cm, annot=True)
plt.xlabel('Real Data')
plt.ylabel('Prediction')
plt.show()
cm
=> array([[3522,  130],
              [ 448,  429]])
위 confusion matrix에서 x 축은 실제 데이터의 label을 의미하고 y 축은 예측한 데이터의 label을 의미합니다.
0,0 의 값: 추천함(Pass) 이라고 예측했을 때, 실제 데이터가 추천함(Pass)인 경우의 개수
0,1 의 값: 추천 하지 않음(Fail) 이라고 예측했을 때, 실제 데이터가 추천함(Pass)인 경우의 개수
1,0 의 값: 추천함(Pass) 이라고 예측했을 때, 실제 데이터가 추천 하지 않음(Fail)인 경우의 개수
1,1 의 값: 추천 하지 않음(Fail) 이라고 에측했을 때, 실제 데이터가 추천 하지 않음(Fail)인 경우의 개수

#분류 모델의 또 다른 성능 지표로 Precsion과 Recall를 구하여 봅시다.
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#sklearn에서 제공하는 recall_score, precision_score를 사용하여 recall과 precision 결과물을 출력합니다.
print("Recall score: {}".format(recall_score(y_test, model_predition)))
print("Precision score: {}".format(precision_score(y_test, model_predition)))

#0번부터 4번까지 5개를 출력해보겠습니다.
for i in range(5): 
    # 의사결정나무 모델을 사용하였습니다.
    prediction = model.predict(x_test[i])
    print("{} 번째 테스트 데이터 문장: \n{}".format(i, df_clean['Review Text'][i]))
    print("{} 번째 테스트 데이터의 예측 결과: {}, 실제 데이터: {}\n".format(i, prediction[0], y_test[i]))

# 교통 표지판 이미지 분류
1. 교통 표지판 이미지 데이터를 분석하고 딥러닝 모델을 통하여 표지판 종류를 예측하는 분류 모델 수행
2. 대량의 이미지 데이터를 전 처리하는 과정과 이에 따른 CNN모델의 성능 변화를 학습

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential


%matplotlib inline

## 데이터 분석
### 이미지 데이터 정보 파악하기 - Meta
이미지 데이터를 읽어오기 위해서 ./data에 어떠한 파일들이 존재 하는지 확인해 봄.
file_list = os.listdir('./data')
file_list
['Meta', 'Meta.csv', 'Test', 'Test.csv', 'Train', 'Train.csv']

3개의 폴더와 3개의 csv 파일이 있음. 일반적으로 이미지 데이터의 csv 파일이 제공되는 경우에는, 해당 이미지의 디렉토리 정보가 저장되어 있음. 
그렇기에 먼저, csv 파일을 데이터프레임으로 읽어 보겠습니다. 

import pandas as pd
df_Meta = pd.read_csv('./data/Meta.csv')
df_Meta

meta.csv는 meta 폴더 내의 이미지에 대한 정보를 담고 있음. 위 정보를 바탕으로 이미지를 출력해봄.
Meta_images = []
Meta_labels = []

plt.figure(figsize=(16,16))
for i in range(len(df_Meta)):
    img = load_img('./data/'+df_Meta['Path'][i])
    plt.subplot(1, 3, i+1) 
    plt.imshow(img)
    Meta_images.append(img)
    Meta_labels.append(df_Meta['ClassId'][i])

### 이미지 데이터 정보 파악하기 -Train
df_Train = pd.read_csv('./data/Train.csv')
df_Train

2670개의 학습용 이미지 데이터에 대한 정보가 저장되어 있음. 
    Width    Height    Roi.X1    Roi.Y1    Roi.X2    Roi.Y2    ClassId    Path
0
1
..
2669

이러한 이미지 중에 width와 height 정보는 이미지의 폭과 높이에 대한 정보로, 간단히 샘플만 봐도 다양한 크기를 갖는 것을 알 수 있음. 
이미지 크기가 모두 다르다면 이미지마다 서로다른 feature의 개수가 있는 것이기 때문에 이를 통일해주는 전 처리가 필요함.
그렇다면, 어떤 크기로 통일을 하는 것이 좋을지는 이미지의 크기와 분포를 보고 판단해봐야 함. 

import seaborn as sns

plt.figure(figsize=(20,10))
ax = sns.countplot(x="Width", data=df_Train)

df_cutWidth = pd.cut(df_Train['Width'], np.arange(0,200,10)).value_counts(sort=False)

fig, ax = plt.subplots(figsize=(20,10))
ax.bar(range(len(df_cutWidth)),df_cutWidth.values)
ax.set_xticks(range(len(df_cutWidth)))
ax.set_xticklabels(df_cutWidth.index)
fig.show()

분포를 통해 30-35의 폭 또는 높이를 가진 이미지가 가장 많음을 확인했음. 
이미지의 크기를 통일할 때, 너무 작은 이미지는 큰 이미지의 정보 손실을 발생시키고, 너무 큰 이미지는 작은 이미지의 부족한 정보량을 부각할 것임.
따라서 적절한 이미지 크기를 잡는 것은 하나의 파라미터 조정이 되며, 이번 프로젝트에서는 이미지 분포 기반으로 대다수를 차지하는 크기인 33*33 크기로 통일.

image_height = 33
image_width = 33
image_channel = 3 # 컬러 이미지이기에 3채널

이미지 데이터에서 Roi는 Range of interest의 약자로 지금 데이터에서는 표지판이 있는 부분을 의미함. 
Train, Test.csv 파일에 있는 Roi 데이터는 아래 실행된 이미지에서의 좌측 상단 좌표와 우측 상단 좌표를 의미함. 
from PIL import Image
from PIL import ImageDraw

img_sample = Image.open('./data/'+df_Train['Path'][0])

draw = ImageDraw.Draw(img_sample) -> img_sample에 drawing 
draw.rectangle([df_Train['Roi.X1'][0], df_Train['Roi.Y1'][0], df_Train['Roi.X2'][0], df_Train['Roi.Y2'][0]], outline="red")
img_sample_resized = img_sample.resize((300,300))
img_sample_resized

Roi 데이터를 사용하면 보다 명확하게 표지판 부분을 crop할 수 있으며, 이러한 데이터 전처리를 통해 분류의 성능을 높일 수 있음. (이번 프로젝트에서는 생략)
img_sample_crop = img_sample.crop((df_Train['Roi.X1'][0], df_Train['Roi.Y1'][0], df_Train['Roi.X2'][0], df_Train['Roi.Y2'][0]))
 
#Shows the image in image viewer
img_sample_crop_resized = img_sample_crop.resize((300,300)) -> PIL 이미지 객체 저장
img_sample_crop_resized

## 데이터 전 처리
### 이미지 데이터 읽기
학습용 이미지를 불러와서 Train_images에 array 형태로 저장함.
image_height = 33
image_width = 33
image_channel = 3

Train_images = []
Train_labels = []

for i in tqdm(range(len(df_Train))): -> tqdm은 파이썬 진행률 프로세스바. 어떤 이터러블이든 tqdm()으로 감싸면 이터러블이 증가하는 것에 따라 진행률 증가
    img = load_img('./data/'+df_Train['Path'][i], target_size = (image_height, image_width)) -> 이미지를 불러올 때부터 사이즈를 변경해서 가져옴. 
    img = img_to_array(img)
    Train_images.append(img)

같은 방식으로 평가용 이미지를 불러와서 Test_images에 저장함. 
Test_images = []
Test_labels = []

for i in tqdm(range(len(df_Test))):
    img = load_img('./data/'+df_Test['Path'][i], target_size = (image_height, image_width))
    img = img_to_array(img)
    Test_images.append(img)

### 라벨 데이터 읽기
학습용, 평가용 데이터에 대한 label은 csv파일에 ClassId 열로 저장되어 있기 때문에 이를 불러와서 array로 저장함. 
Train_labels = df_Train['ClassId'].values
Test_labels = df_Test['ClassId'].values

### 데이터 분리하기
딥러닝 학습 시, 과적합을 막기 위해서 validation 데이터를 학습용 데이터에서 분리함. 
모든 데이터는 numpy array로 저장
x_train, x_val, y_train, y_val = train_test_split(np.array(Train_images), np.array(Train_labels), test_size=0.4)
평가용 데이터도 적용함
x_test = np.array(Test_images)
y_test = np.array(Test_labels) -> 테스트 데이터에 대해선 validation데이터가 필요없기 때문에 그냥 np.array로 변행해주면 됨. 
numpy  array가 가장 오류가 적기 때문에, numpy array로 변경해서 학습시키는 것이 가장 좋음. 

## 딥러닝 모델 
### cnn 모델 설정
cnn을 사용하여 간단하게 모델을 구현해보겠음. filters, kernel 등의 사이즈는 하이퍼 파라미터로 자신만의 모델로 튜닝이 가능함. 
model = Sequential([    
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(image_height, image_width, image_channel)),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25), -> 학습을 할 때 25%만큼 노드를 줄여준다는 뜻. 과적합을 피하기 위해서 사용함. 학습을 많이 수행할수록 과적합이 생길 수 있기 때문에 이를 방지하기 위해서!
    
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(rate=0.25),
    Dense(3, activation='softmax') -> 3개 종류로 분류하니까!
])

model.summary()

### 학습 수행
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#처음 만든 모델이라면 EPOCHS를 1~5개로 하여 잘 돌아가는지 성능을 확인해보고 값을 증가 시켜 봅시다. 
EPOCHS = 30

#EPOCHS에 따른 성능을 보기 위하여 history 사용
history = model.fit(x_train, 
                    y_train,
                    validation_data = (x_val, y_val), # validation 데이터 사용
                    epochs=EPOCHS, 
                   )

학습을 수행하면서 accuracy와 loss의 변화를 그래프로 출력.
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

### 모델 성능 평가 및 예측
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('test set accuracy: ', test_accuracy)
-> test set accuracy:  0.9571428298950195

테스트 데이터를 입력하여 예측된 결과를 비교해보겠음.
25개의 테스트 데이터를 불러와 실제 class와 예측 class를 출력하면 다음과 같음. 
test_prediction = np.argmax(model.predict(x_test), axis=-1) -> argmax를 통해서 predict로 불러온 각 클래스의 확률들 중 가장 큰값을 뽑아냄. 
plt.figure(figsize = (13, 13))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False) -> grid 해제
    plt.xticks([]) -> x축, y축에 아무것도 출력하지 않도록 설정
    plt.yticks([])
    prediction = test_prediction[start_index + i]
    actual = y_test[start_index + i]
    col = 'g' ->그린
    if prediction != actual:
        col = 'r' -> 레드
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
    plt.imshow(array_to_img(x_test[start_index + i]))
plt.show()

coufusion matrix를 통해서 시각화하여 분류 학습 결과 확인
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_prediction)
plt.figure(figsize = (20, 20))
sns.heatmap(cm, annot = True)

# 반도체 공정 데이터를 확용한 공정 이상 예측
1. 반도체 공정 데이터 분석을 통해 공정 이상을 예측하는 분류 모델 수행
2. 공정 이상에 영향을 미치는 요소들에 대한 데이터 분석

## 데이터 읽기
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#uci-secom.csv 데이터를 pandas를 사용하여 dataframe 형태로 불러옵니다.
data = pd.read_csv('data/uci-secom.csv')

#5개의 행을 확인합니다. head()를 사용합니다. head() 안에 숫자를 넣을 수 있습니다. 
data.head()

#dataframe의 정보를 요약해서 출력합니다.
#info()로 정보를 알 수 있습니다.
#shape로 몇 행과 몇 열로 되어있는지 알 수 있습니다. 처음이 행이고 두번째 열입니다.
data.info()
data.shape

#수치형 변수의 데이터 정보를 요약하여 출력합니다.
#mean은 평균, std는 표준편차를 나타냅니다. 
data.describe()

## 데이터 정제
데이터 정제에서는 일반적으로 결측값, 이상치를 처리함
결측값은 값이 없는 것. 즉, NaN, null이 결측값임.
이상치는 일반적인 범주에서 벗어난 값임. 평균 연령을 구할 때 200살과 같이 일반적인 범주에 있지 않은 값을 이상치라고 함. 
이번 데이터에서는 수많은 변수(feature)가 존재하기 때문에 각 데이터를 보며 이상치를 처리하기엔 한계가 있음. 
따라서 이번 프로젝트에선 간단하게 결측값에 대해서만 처리를 수행.

#결측값 정보를 출력합니다.
#isnull()은 결측값이 있는지 True, False로 반환합니다. 
#data.isnull().sum()로 각 컬럼에서 결측값의 수를 구합니다.
#data.isnull().sum().sum()로 전체 결측값의 수를 구할 수 있습니다.
data.isnull().sum()

모든 데이터를 사용하기 위해선 결측값을 0으로 대체함.
결측값이 많지 않다면 fillna(값, inplace=True)를 사용하여 삭제하는 방법도 있음. 

#결측값을 0으로 대체합니다.
#np.NaN이 결측값입니다. 이것을 replace을 사용해서 0으로 바꿉니다.
data = data.replace(np.NaN, 0)

#결측값 정보를 출력합니다.
data.isnull().sum()

#'Time'변수의 데이터는 pass/fail을 예측하는데 큰 영향이 없다 생각하여 삭제합니다.
#axis=0은 행방향으로 동작합니다. 
#axis=1은 열 방향으로 동작합니다. 
#drop() 안에 삭제할 컬럼 이름을 적고 axis =1 로 정합니다.
data = data.drop(columns = ['Time'], axis = 1)

data.shape

## 데이터 시각화
머신러닝을 할 때 숫자만으로는 데이터가 어떤 의미를 갖는지 이해하기 어려움.
데이터를 시각화해서 파악하는 것이 중요함.
센서에 관련된 590개의 변수들은 시각화하기엔 너무 양이 많기 때문에 영향력이 크다고 판단되는 59 센서에 대해서만 시각화를 진행해볼 예정. 
59번 데이터는 머신러닝 모델을 사용했을 때, 높은 중요도로 뽑힌 변수이기 때문에 대표로 출력.

### pass/fail 시각화
#분포를 막대 그래프를 사용하여 출력합니다.
#pandas 모듈을 plot()를 사용해서 막대그래프를 그릴 수 있습니다.
#value_counts()로 합계를 구합니다.  
data['Pass/Fail'].value_counts().plot(kind='bar')
#분포를 도수분포표로 확인합니다.
data['Pass/Fail'].value_counts()

### 센서 데이터 시각화 하기
다수의 feature 데이터에 대해서 한 눈에 볼 수 있도록 시각화를 수행할 때는 seaborn의 pairplot을 활용해서 해결할 수 있음.
590개 센서에 대한 출력을 pairplot으로 수행하기엔 출력 결과도 보기 힘들뿐더러 출력 시간도 매우 오래 걸림.
따라서 아래 코드와 같이 3,4,5,pass/fail 데이터에 대해서만 출력해보겠음. 
#3,4,5,Pass/Fail 컬럼으로 새로운 DataFrame을 만듭니다. 리스트 안에 컬럼 이름을 적습니다. 
data_test= data[['3','4','5','Pass/Fail']]
data_test
#seaborn의 pairplot()을 사용해서 컬럼끼리 비교할 수 있습니다. 
sns.pairplot(data_test)
#vars를 사용해서 특정한 컬럼끼리 비교할 수도 있습니다. 
sns.pairplot(data_test,height=5, vars=['3','4'])

### 59번 센서 시각화
#그래프의 사이즈를 설정합니다.
#subplots는 한 번에 여러 그래프를 보여주기 위해서 사용합니다. 
#subplots()에선 두개의 값을 받을 수 있는데 figure와 axes 값을 받을 수 있습니다. 여기서 변수명은 상관없습니다. 순서가 중요합니다.
#fig란 figure로써  전체 subplot을 말합니다. 몇개의 그래프가 있던지 상관없이 그것을 담는 그릇이라고 생각하면 됩니다. 전체 사이즈를 말합니다.
#ax는 axe로써 각각의 그래프를 말합니다. 
#figsize(가로, 세로)로 크기를 정합니다. 
fig, ax = plt.subplots(figsize=(8, 6))

#seborn 그래프의 스타일을 설정합니다.
#style에 white, whitegrid, dark 등을 넣어서 스타일을 바꿀 수 있습니다.
sns.set(style='darkgrid')

#59번 데이터의 분포를 출력합니다.
#displot로 분포도를 그립니다. 
#yellow, green와 같은 색깔을 넣습니다. 
sns.distplot(data['59'], color = 'darkblue')

#그래프의 제목을 설정합니다. 
plt.title('59 Sensor Measurements', fontsize = 20)

#그래프의 사이즈를 설정합니다. 첫번째는 가로, 두번째는 세로의 크기입니다. 
plt.rcParams['figure.figsize'] = (10, 16)

#3x1 형태로 그래프를 출력하기 위하여 subplot을 설정합니다. 
#subplot(행, 열, 인덱스)로 그래프의 위치를 정합니다. 
plt.subplot(3, 1, 1)
sns.distplot(data['59'], color = 'darkblue')
plt.title('59 Sensor Measurements', fontsize = 20)

#'Pass/Fail' 값이 1인 데이터를 출력합니다.
#data[data['Pass/Fail']==1]를 하면 'Pass/Fail' 값이 1인 행만 사용할 수 있습니다.
plt.subplot(3, 1, 2)
sns.distplot(data[data['Pass/Fail']==1]['59'], color = 'darkgreen')
plt.title('59 Sensor Measurements', fontsize = 20)

#'Pass/Fail' 값이 -1인 데이터를 출력합니다.
plt.subplot(3, 1, 3)
sns.distplot(data[data['Pass/Fail']==-1]['59'], color = 'red')
plt.title('59 Sensor Measurements', fontsize = 20)

#그래프의 사이즈를 설정합니다. 첫번째는 가로, 두번째는 세로의 크기입니다.
plt.rcParams['figure.figsize'] = (15, 10)

#위 나누어 출력 했던 그래프를 한번에 출력합니다.
sns.distplot(data['59'], color = 'darkblue')
sns.distplot(data[data['Pass/Fail']==1]['59'], color = 'darkgreen')
sns.distplot(data[data['Pass/Fail']==-1]['59'], color = 'red')

#제목과 폰트크기를 정합니다.
plt.title('59 Sensor Measurements', fontsize = 20)

**subplot으로 그래프의 행,열,인덱스를 지정해줘야 분리됨**

## 데이터 전 처리
공정 이상 예측을 수행하기 위해서 주어진 센서 데이터에 대해서 분류 모델을 사용할 것.
분류 모델의 필요한 입력 데이터를 준비 하기 위해서 다음과 같은 전 처리를 수행하겠음.
1. 전체 데이터를 feature 데이터인 x와 label 데이터인 y로 분리하기
2. StandardScalar를 통한 데이터 표준화하기

### x와 y로 분리
머신러닝의 feature 데이터는 x, label 데이터는 y에 저장함. 
#예측해야 할 변수인 `Pass/Fail`를 제거하여 머신러닝 입력값인 x에 저장합니다.
#data에는 'Pass/Fail'의 없어집니다. 
x = data.drop(columns = ['Pass/Fail'], axis = 1)

#예측해야 할 변수 `Pass/Fail`만을 선택하여 numpy 형태로 y에 저장합니다.
y = data['Pass/Fail']

#ravel은 "풀다"로 다차원을 1차원으로 푸는 것을 의미합니다.
#1차원 벡터 형태로 출력하기 위해 ravel 사용합니다. 
y = y.to_numpy().ravel() 
y

#타입을 확인합니다. 
type(y)

원본 데이터의 수가 많지 않기 때문에 원본 데이터에서 샘플 데이터를 추출하고 노이즈를 추가하여 테스트 데이터를 생성. 
data 폴더 내의 uci-secom-test.csv에 590개의 센서 데이터와 pass/fail 저장되어 있기 때문에 해당 데이터를 읽어와 x_test, y_test 데이터로 분리함. 
#data 폴더 내의 uci-secom-test.csv를 DataFrame으로 읽고 x_test, y_test로 분리합니다. 
data_test = pd.read_csv("data/uci-secom-test.csv") #기존의 데이터를 가져온 csv파일은 uci-secom.csv이고, 테스트데이터는 uni-secom-test.csv에서 가져온 것. 
x_test = data_test.drop(columns = ['Pass/Fail'], axis = 1)
y_test = data_test['Pass/Fail'].to_numpy().ravel() 

### 데이터 표준화
각 변수마다 스케일 차이를 맞추기 위해 표준화를 수행함. 
표준화는 서로 다른 피처의 크기를 통일하기 위해서 크기를 변환해주는 개념. 
데이터의 피처 각각이 평균이 0이고 분산이 1인 가우시안 정규 분포 형태와 가까워지도록 변환함.
from sklearn.preprocessing import StandardScaler
#정규화를 위해서 StandardScaler 불러옵니다.
sc = StandardScaler()
#x_train에 있는 데이터에 맞춰 정규화를 진행합니다. 
x_train = sc.fit_transform(x)
x_test = sc.transform(x_test)
y_train = y

#mean()으로 평균을 구하고 var()로 분산을 구합니다. 
#e는 소수부의 크기를 알려주는 자리입니다. 여기서는 엄청 작은 값으로 0으로 생각하면 됩니다. 
x_train_sc = pd.DataFrame(data=x_train)
print("평균")
print(x_train_sc.mean())
print("분산")
print(x_train_sc.var())

## 머신러닝 모델 학습
전 처리된 데이터를 바탕으로 분류 모델 학습을 수행하고 학습 결과를 출력.
먼저 기본적인 분류 모델인 로지스틱 분류기를 사용하여 학습을 수행하고, 다양한 모델들을 살펴봄.
**로지스틱 회귀**
로지스틱 회귀는 선형 회귀 방식을 분류에 적용한 알고리즘입니다.
로지스틱 회귀는 회귀라는 말이 들어갔지만 분류에 사용됩니다.
로지스틱 회귀가 선형 회귀와 다른 점은 학습을 통해 선형 함수의 회귀 최적선을 찾는 것이 아닙니다.
시그모이드 함수 최적선을 찾고 이 시그모이드 함수의 반환 값을 확률로 간주해 확률에 따라 분류를 결정한다는 점입니다.
확률에 따라서 분류를 결정합니다.
로지스틱 회귀는 주로 이진(0과 1) 분류에 사용됩니다. 로지스틱 회귀에서 예측 값은 예측 확률의 의미합니다.
예측 값 즉, 예측 확률이 0.5이상이면 1로, 그렇지 않으면 0으로 예측합니다.

from sklearn.linear_model import LogisticRegression

#로지스틱 분류기 모델 class를 가져 옵니다.
#max_iter는 로지스틱 알고리즘의 반복 횟수를 정하는 파라미터로 본 실습에서는 default 값으로는 모자르기에 아래와 같이 설정합니다.
model = LogisticRegression(max_iter=5000)

#데이터를 학습시킬 때는 fit 함수를 사용합니다. 
model.fit(x_train, y_train)

#score 함수를 사용하여 모델의 성능을 확인합니다. 
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

#Logistic Regression의 중요도를 계산합니다.
#가중치 값들의 크기로 판단하기에 .coef_로 해당 값들을 불러옵니다.
abs_coef = np.abs(model.coef_).ravel()
abs_coef

#bar 형태 그래프로 Logistic Regression의 feature 별 중요도를 상위 20개 출력합니다.
#상위 20개의 feature 정보를 출력하기 위하여 sorting을 수행하고 해당 feature 번호를 LR_imort_x에 저장합니다.
**LR_import_x = [str(i[0]) for i in sorted(enumerate(abs_coef), key=lambda x:x[1], reverse=True)]**

**plt.bar(LR_import_x[:20], sorted(abs_coef, reverse=True)[:20])**

plt.rcParams['figure.figsize'] = (15, 10)
plt.xlabel('Features')
plt.ylabel('Weight absolute values')
plt.show()
