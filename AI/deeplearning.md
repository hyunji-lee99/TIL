## 딥러닝
딥러닝<머신러닝<인공지능
딥러닝이란 머신러닝의 여러 방법론 중 하나, 인공신경망에 기반하여 컴퓨터에게 사람의 사고방식을 가르치는 방법임.
생물학의 신경망에서 영감을 얻은 학습 알고리즘. 사람의 신경 시스템을 모방함.

## 퍼셉트론
인공 신경망의 가장 기본 단위.
신경 세포와 비슷한 형태로 구성
여러 개의 input 신호 ======> 여러 개의 output
입력값x, 가중치w, bias(w0), 출력값y으로 구성되어 있음.
y=activation function(w0+w1x1+w2x2) (활성화 함수)
x1과 x2라는 신호가 들어오면 각각 가중치 w1,w2가 곱해져서 신호를 증폭함. bias w0는 입력값에 상관없이 무조건 입력되는 값. 이러한 값들이 모두 sumation된 것이 activation function을 거쳐서 출력값 y가 나옴.
활성화 함수는 y=activation(x)=1(x>=0) or 0(x<0)
퍼셉트론은 **선형 분류기**로써, 데이터 분류가 가능함.
하나의 선으로 분류할 수 없는 문제를 해결하는 것에 한계가 있음.

### 학습 여부를 예측하는 퍼셉트론 함수
```
def Perceptron(x_1,x_2):
    # 설정한 가중치값을 적용
    w_0 = -5
    w_1 = -1
    w_2 = 5
    # 활성화 함수에 들어갈 값을 계산
    output = w_0+w_1*x_1+w_2*x_2
    # 활성화 함수 결과를 계산
    if output < 0:
        y = 0
    else:
        y = 1
    return y, output
```
**DIY 퍼셉트론 만들기**
```
def perceptron(w, x):
  output = w[0]
  for i in range(len(x)):
    output=output+(x[i]*w[i+1])

  if(output>=0):
    y=1
  elif(output<0):
    y=0

  return y, output
```
## 다층 퍼셉트론(Multi Layer Perceptron, MLP)
선 하나만으로 분류가 불가능한 비선형적인 문제를 해결할 수 있음. 단층 퍼셉트론은 입력층과 출력층만 존재함. 단층 퍼셉트론은 입력층과 출력층만이 존재함.
다층 퍼셉트론은 단층 퍼셉트론을 여러 개 쌓은 것이라고 할 수 있음.
입력층과 출력층 사이에 모든 layer를 hidden layer라고 하는데, 히든층이 많아진다면 깊은 신경망이라는 의미의 **Deep Learning**이라는 단어를 사용함.
장점으론 분류할 수 있는 방법이 많아지면서 성능이 좋아지지만, 단점으론 퍼셉트론이 쌓이면서 하나의 모델에 필요한 가중치를 많이 구해야 하기 때문에 어려움.


# 텐서플로우와 신경망
텐서플로우는 가장 많이 사용되고 있는 딥러닝 프레임워크.
## 딥러닝 모델의 학습방법
딥러닝 모델은 노드/유닛(각 층을 구성하는 요소), 가중치(노드 간의 연결강도), layer(모델을 구성하는 층)으로 구성됨.
예측값과 실제값 간의 오차값을 최소화하기 위해 오차값을 최소화하는 모델의 인자를 찾는 알고리즘을 적용함.
즉, loss 함수를 최소화하는 가중치를 찾기 위해 최적화 알고리즘을 적용함.
<딥러닝 모델이 예측값을 구하는 방식>
순전파(forward propagation)은 입력값을 바탕으로 출력값을 계산하는 과정임. 즉, 입력값부터 퍼셉트론을 반복적으로 거치며 출력값을 계산하는 방식.
순전파를 사용하면 예측값과 실제값 간의 오차값을 구하여 loss 함수를 구할 수 있음. 그렇다면 어떻게 최적화를 해야할까? -> 경사하강법 사용.
경사하강법은 loss함수 값이 작아지게 업데이트하는 방법. 가중치는 gradient값을 사용하여 업데이트를 수행함. gradient값은 각 가중치마다 정해지며, 역전파(backpropogation)을 통하여 구할 수 있음.
각각의 가중치 w1,w2,w3,..에 대하여 각각의 gradient 값인 grw1,grw2,grw3,..이 존재함. 특정 단계의 gradient 값을 구하려면 그 전단의 계산값이 필요함.
예를 들어, w3의 gradient 값을 구하려면 다음 노드에 있는 w6에 대한 값이 필요함. 최종적으론 제일 끝단에 있는 gradient값을 계산하고, 그 다음 노드의 값을 계산하고,..와 같은 형식으로 진행되기 때문에 backpropagation이라 함.

정리 : 입력단에 값들이 도착하면 순전파를 통해서 다음 단계로 전달을 하면서 최종적인 예측값을 구할 수 있었고, 예측값과 실제값의 오차를 통해서 loss를 구할 수 있었고, loss를 바탕으로 gradient값을 역전파를 통해 구하면서 각 w값에 대한 gradient 값을 구하면서 가중치 w값을 업데이트할 수 있음. 이러한 과정들을 반복해서 loss를 작게 하는 가중치를 구할 수 있음.
딥러닝 모델의 학습 순서
1. 학습용 feature 데이터를 입력하여 예측값을 구함.(순전파)
2. 예측값과 실제값 사이의 오차 구함.(loss 계산)
3. loss를 줄일 수 있는 가중치 업데이트.(역전파)
4. 1-3번 반복으로 loss를 최소로 하는 가중치를 얻음.

### 텐서플로우로 딥러닝 구현하기
텐서플로우는 유연하고, 효율적이며, 확장성 있는 딥러닝 프레임워크임. 대형 클러스터 컴퓨터부터 스마트폰까지 다양한 디바이스에서 동작 가능.
1. 데이터 전처리하기
tensorflow 딥러닝 모델은 tensor 형태의 데이터를 입력받음. tensor란 다차원 배열로서 tensorflow에서 사용하는 객체.
#1차원 배열 : vector, 2차원 배열: matrix, 3,4,5,..차원 배열 : tensor
**epoch와 batch**
딥러닝에 사용하는 추가적인 전 처리 작업으로, 한번의 epoch는 전체 데이터 셋에 대해 한 번 학습을 완료한 상태. batch는 나눠진 데이터 셋(보통 mini-batch라고 표현)
iteration은 batch를 수행하는 횟수. epoch를 나누어서 실행하는 횟수를 의미함.
딥러닝을 학습할 때 가중치 w들을 여러 번 업데이트해야 하는데, 딥러닝 모델이 무거울 수록 업데이트해야 하는 w의 양이 늘어나고 많은 계산이 필요하게 됨. 이러한 계산 양을 줄이기 위해서 한번에 전체 epoch 만큼의 데이터를 넣는 것이 아니고 쪼개서 데이터를 넣어보자고 하는 것이 batch임. 1번 배치, 2번 배치,...등으로 나누어서 데이터를 넣어보는 것임. 이렇게 나누면 확률적으로 성능이 안좋아질 수 있으나, 계산 측면에서 한 배치 안에서 가중치를 업데이트하는 것이 훨씬 빠르기 때문에 더 좋다고 할 수 있음.
예를 들어, 총 데이터가 1000개, 배치 사이즈를 100으로 하면, 10개의 배치가 나오며 1 iteration은 100개 데이터에 대해서 학습한 것임. 1 epoch=1000/batch size=10 iteraion
```
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(100)
tf.random.set_seed(100)
#데이터를 DataFrame 형태로 불러 옵니다.
df = pd.read_csv("data/Advertising.csv")
#의미없는 변수는 삭제합니다.
df = df.drop(columns=['Unnamed: 0'])
"""
1. Sales 변수는 label 데이터로 Y에 저장하고 나머진 X에 저장합니다.
"""
X = df.drop(columns=['Sales'])
Y = df['Sales']
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)
"""
2. 학습용 데이터를 tf.data.Dataset 형태로 변환합니다.
   from_tensor_slices 함수를 사용하여 변환하고 batch를 수행하게 합니다.
"""
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5) #shuffle은 데이터를 한 번 섞어주는 개념으로 볼 수 있음. 인자로는 데이터 크기를 입력하면 됨.
#하나의 batch를 뽑아서 feature와 label로 분리합니다.
[(train_features_batch, label_batch)] = train_ds.take(1)
#batch 데이터를 출력합니다.
print('\nFB, TV, Newspaper batch 데이터:\n',train_features_batch)
print('Sales batch 데이터:',label_batch) </code>
```

2. 딥러닝 모델 구축하기
텐서플로우의 패키지로 제공되는 고수준 API keras를 사용함. 딥러닝 모델을 간단하고 빠르게 구현 가능함.
케라스의 메소드
1. 모델 클래스 객체 생성 tf.keras.models.Sequential() -> **딥러닝 모델을 만들 것이라 선언**하는 것과 같은 의미. 하나의 딥러닝을 위한 도화지를 펼치는 것이라 할 수 있음. 이 도화지에 각 레이어를 쌓으면서 모델을 완성함.
2.  모델의 각 레이어 구성
tf.keras.layers.Dense(units,activation)
units: 레이어 안의 node의 수
activation: 적용할 activation 함수 설정
e.i) sig,tanh, le 등..
첫 번째 즉, input layer는 입력 형태에 대한 정보를 필요로 함. -> input_shape / input_dim 인자 설정.
모델 구축코드 예시
```
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(10,**input_dim=3**),
tf.keras.layers.Dense(1), -> 회귀 분석
])
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(10, **input_shape=(3,)**),
tf.keras.layers.Dense(1)
])
```

모델에 레이어 추가하기
[model].add(tf.keras.layers.Dense(units,activation))
sequential로 모델 구성해놓고, model.add로 레이어를 추가할 수 있음.
모델 구축코드 예시
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10,input_dim=3))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(1))

3. 모델 학습시키기
모델 학습 방식을 설정하기 위한 함수 [model].compile(optimizer,loss)
optimizer : 모델 학습 최적화 방법 e.i) gd,sgd,momentum,adam 등..
loss: 손실 함수 설정
loss는 회귀에서는 일반적으로 MSE인 ‘mean_squared_error’, 분류에서는 ‘sparse_categorical_crossentropy’ 를 주로 사용함
모델을 학습시키기 위한 함수 [model].fit(x,y)
x:학습 데이터(feature)
y:학습 데이터의 label
또는 tensor형태의 데이터셋과 epochs를 넣어줘도 됨.
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_ds, epochs=100, verbose=2)
verbose 인자는 학습 시, 화면에 출력되는 형태를 설정함. (0: 표기 없음, 1: 진행 바, 2: 에포크당 한 줄 출력)

4. 평가 및 예측하기
모델을 평가하기 위한 메소드 [model].evaluate(x,y)
x: 테스트 데이터
y:테스트 데이터의 label

모델로 예측을 수행하기 위한 함수 [model].predict(x)
x: 예측하고자 하는 데이터(feature)
```
#evaluate 메서드를 사용하여 테스트용 데이터의 loss 값을 계산합니다.
loss = model.evaluate(test_X, test_Y, verbose=0)
#predict 메서드를 사용하여 테스트용 데이터의 예측값을 계산합니다.
predictions = model.predict(test_X)
#결과를 출력합니다.
print("테스트 데이터의 Loss 값: ", loss)
for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %f" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %f" % (i, predictions[i][0]))
```
$ evaluate() 메서드는 학습된 모델을 바탕으로 입력한 feature 데이터 X와 label Y의 loss 값과 metrics 값을 출력합니다. 이번 실습에서는 metrics 를 compile에서 설정하지 않았지만, 분류에서는 일반적으로 accuracy를 사용하여 evaluate 사용 시, 2개의 아웃풋을 리턴합니다.

**신경망 모델로 분류하기**
```
#sklearn에 저장된 데이터를 불러 옵니다.
X, Y = load_iris(return_X_y = True)

#DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']

#학습용 평가용 데이터로 분리합니다
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

#Dataset 형태로 변환합니다.
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

"""
1. keras를 활용하여 신경망 모델을 생성합니다.
   3가지 범주를 갖는 label 데이터를 분류하기 위해서 마지막 레이어 노드를 아래와 같이 설정합니다.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=4),
    tf.keras.layers.Dense(**3**, activation='softmax')
    ])

#학습용 데이터를 바탕으로 모델의 학습을 수행합니다.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#분류에서는 일반적으로 loss를 ‘sparse_categorical_crossentropy’으로 사용합니다. metrics 인자는 에포크마다 계산되는 평가 지표를 의미합니다. 정확도를 의미하는 ‘accuracy’ 를 입력하면 에포크마다 accuracy를 계산하여 출력합니다.
history = model.fit(train_ds, epochs=100, verbose=2)

#테스트용 데이터를 바탕으로 학습된 모델을 평가합니다.
loss, acc = model.evaluate(test_X, test_Y)

#테스트용 데이터의 예측값을 구합니다.
predictions = model.predict(test_X)

#결과를 출력합니다.
print("테스트 데이터의 Accuracy 값: ", acc)
for i in range(5):
    print("%d 번째 테스트 데이터의 실제값: %d" % (i, test_Y.iloc[i]))
    print("%d 번째 테스트 데이터의 예측값: %d" % (i, np.argmax(predictions[i])))
```
### 다양한 신경망
이미지와 자연처 처리에서 사용하는 CNN, RNN 모델 등이 있음.
### 이미지 처리 e.i) 얼굴 인식, 화질 개선, 이미지 자동 태깅
이미지 -> 컴퓨터가 각 픽셀값을 가진 숫자 배열로 인식
이미지 전처리 -> 모두 같은 크기를 갖는 이미지로 통일
1. 가로 세로 픽셀 사이즈를 표현하는 **해상도 통일** 해상도=(가로 픽셀 수)*(세로 픽셀 수)
2. **색을 표현하는 방식** 통일(RGB, HSV, Gray-scale, Binary,...)
$ mnist 데이터 : 사람의 손글씨를 이미지 데이터로 표현한 것.

기존의 다층 퍼셉트론 기반 신경망의 이미지 처리 방식은 픽셀 수 만큼의 입력값 노드부터 신경망을 여러번 거쳐야 하기 때문에 극도로 많은 파라미터가 필요하고, 만약 이미지가 변화한다면 너무 많은 데이터의 변화가 일어나기 때문에 문제가 있음. -> 분류 성장 매우 하락

이를 해결하기 위한 방식이 합성공 신경망(Convolution Neural Network, CNN)이 있음.
cnn은 작은 필터를 순환시키는 방식임. 이미지의 패턴이 아닌 특징을 중점으로 인식. 예를 들어, 고양이의 경우 귀, 코, 수염, 꼬리 => 고양이 식으로 고양이임을 인식함. 고양이가 뒤짚혀 있는 경우에도 특징을 파악하기 때문에 고양이임을 알 수 있음.

합성곱 신경망의 구조
입력 이미지 => (Convolution Layer =>  Pooling Layer) => Fully-Connected Laye
( ) : CNN
cnn모델로 이미지의 특징들을 추출하고, 이를 기반으로 fc가 분류를 수행함. fc는 우리가 기존에 사용했던 인공지능 모델이라고 할 수 있음. 노드가 모두 다 연결되어 있기 때문에 fully-connected라고 이름 붙여짐.
이미지에서 어떤 특징이 있는지 구하는 과정 -> **convolution** 레이어. 필터가 이미지를 이동하며 새로운 이미지(피쳐맵)를 생성.
tf.keras.layers.Conv2D(filters, kernel_size, activation, padding)
-filters : 필터(커널) 개수
-kernel_size : 필터(커널)의 크기
-activation : 활성화 함수
-padding : 이미지가 필터를 거칠 때 그 크기가 줄어드는 것을 방지하기 위해서 가장자리에 0의 값을 가지는 픽셀을 넣을 것인지 말 것인지를 결정하는 변수. ‘SAME(적용)’ 또는 ‘VALID(적용x)’

입력 이미지 * 필터(커널) = 피처맵
예를 들어, 고양이 이미지에 귀 필터를 통과한다면 피처맵에 귀에 해당하는 부분에 대해서 높은 값, 귀에 해당하지 않는 부분에 대해선 낮은 값을 매핑하면서 피처맵에서 귀의 존재 유무를 판단하고, 귀의 위치를 알 수 있음.

**피처맵의 크기 변형**
필터를 적용할 때 고려해야 할 사항, 원본 사이즈와 다른 사이즈의 피처맵이 나오는 것이 싫어서 사용하는 패딩 방식.
padding (원본 이미지의 상하좌우에 0 값을 가진 줄을 한 줄씩 추가)
striding 은 필터를 이동시키는 거리 설정.
**pooling layer**
이미지의 왜곡의 영향을 축소하고 정보를 압축하는 과정으로, 예를 들어, 4*4 사이즈의 피처맵을 max pooling(사용 많음)이나 average pooling으로 2*2 사이즈로 바꿀 수 있음.
tf.keras.layers.MaxPool2D(padding)
padding : ‘SAME’ 또는 ‘VALID’

이러한 과정을 거쳐 추출된 특징들로 fc layer에서 이미지를 분류함.
귀 필터로 추출한 결과, 입 필터로 추출한 결과, 수염 필터로 추출한 결과 등을 노드 입력값으로 시작.

tf.keras.layers.Flatten()
convolution, maxpooling layer의 결과는 n차원의 텐서 형태기 때문에 이를 1차원으로 평평하게 만들어줌.

분류를 할 때, 우리가 단순히 이것이 a가 맞는지 아닌지 형식으로 문제를 푼다면 마지막 레이어에 노드를 하나만 두고, step activation 함수를 통해서 0인지 1인지만 파악하면 됨.
하지만, 여러 개의 label을 두고 각 레이블들의 확률을 구해야 한다면 softmax 함수를 사용해야 함. 예를 들어, 고양이일 확률 60%, 호랑이일 확률 20%, 물고기일 확률 20% 등..
**분류를 위한 softmax 활성화 함수**
마지막 계층에 softmax 활성화 함수 사용. 마지막 레이어에 존재하는 노드의 개수는 우리가 예측해야 하는 label 개수만큼 두면 됨. softmax의 결과물은 각 레이블의 확률 a,b,c,d,..등으로 여러 값이 나오고 모든 값을 더하면 1이 됨.
tf.keras.layers.Dense(node, activation)

정리 : **convolution layer는 특징을 찾아**내고, **pooling layer는 처리할 맵(이미지)의 크기를 줄여주고, 노이즈 감소 효과**를 냄. 이를 n회 반복. 반복할 때마다 줄어든 영역에서의 특징을 찾게 되고, 영역의 크기가 작아졌기 때문에 빠른 학습이 가능해짐. 마지막으로 fully-connected layer에서 분류함.

**mnist 분류 cnn 모델-데이터 전처리**
cnn 모델은 채널(rgb 혹은 흑백)까지 고려한 3차원 데이터를 입력으로 받기에 채널 차원을 추가해서 데이터의 모양을 바꿔줘야 함.
[데이터 수, 가로 길이, 세로 길이] -> [데이터 수, 가로 길이, 세로 길이, 채널 수]
차원 추가 함수 tf.expand_dims(data, axis) axis에 0을 넣으면 데이터 안에 0번째 인덱스에 새로운 차원 추가되면서 뒤로 밀림. 마지막 인덱스에 차원을 추가하려면 -1을 넣으면 됨.
#MNIST 데이터 세트를 불러옵니다.
mnist = tf.keras.datasets.mnist
#MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() -> 이미 load_data를 통해서 train,test_images 안에 있는 데이터들이 텐서 형태로 변경되어 있음.
#Train 데이터 5000개와 Test 데이터 1000개를 사용합니다.
train_images, train_labels = train_images[:5000], train_labels[:5000]
test_images, test_labels = test_images[:1000], test_labels[:1000]
"""
1. CNN 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀, 세로픽셀, 1) 형태로 변환합니다.
"""
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)

**mnist 분류 cnn 모델-모델 구현**
model = tf.keras.Sequential([
tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = '**relu**', padding = 'SAME', input_shape = (28,28,1)),-> expand_dim으로 추가된 차원 1. input_dim은 사용불가
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME'),
    tf.keras.layers.MaxPool2D(padding = 'SAME'),
    tf.keras.layers.Flatten(), -> 텐서 형태를 1차원 벡터형태로 변경
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
model.compile(loss = 'sparse_categorical_crossentropy',
optimizer = 'adam',
metrics = ['accuracy'])
history = model.fit(train_images, train_labels, epochs = 20, batch_size = 512) -> 배치 사이즈를 여기서 같이 설정해주면, 데이터가 들어오면 배치 사이즈까지 처리한 후에 학습을 하게 됨.
def Visulaize(histories, key='loss'):
for name, history in histories:
    plt.plot(history.epoch, history.history[key],
         label=name.title()+' Train')

plt.xlabel('Epochs')
plt.ylabel(key.replace('_',' ').title())
plt.legend()
plt.xlim([0,max(history.epoch)])    
plt.savefig("plot.png")

**mnist 분류 cnn 모델-평가 및 예측**
"""
1. 평가용 데이터를 활용하여 모델을 평가합니다.
   loss와 accuracy를 계산하고 loss, test_acc에 저장합니다.
"""
loss, test_acc = model.evaluate(test_images, test_labels, verbose = 0)

"""
2. 평가용 데이터에 대한 예측 결과를 predictions에 저장합니다.
"""
predictions = model.predict_classes(test_images) -> softmax로 뽑아낸 각 이미지의 여러 레이블의 확률값 중 가장 큰 값을 뽑아서 출력함.

## 자연어 처리
기계 번역 모델, 음성 인식 등에서 자연어 처리를 사용함.
**자연어 전 처리(preprocessing)**
원 상태 그대로의 자연어는 전처리 방법이 필요함.
-noise canceling(오류 교정)
예를 들어, "안녕하 세요. 반갑 스니다."=>"안녕하세요. 반갑습니다."
자연어 문장의 스펠링 체크 및 띄어쓰기 오류 교정
-tokenizing
예를 들어, "딥러닝 기초 과목을 수강하고 있습니다."=> '딥', '러닝', '기초', '과목', '을', '수강', '하고', '있습니다', '.'
문장을 토큰으로 나움. 토큰은 어절, 단어 등으로 목적에 따라 다르게 정의. 숫자가 아닌 문장 데이터를 딥러닝의 입력으로 사용하기 힘들기 때문에 수치 변환을 해줘야 하는데, 한 문장 전체를 수치로 변환하기엔 어려움이 크기 때문에 단어, 어절 등으로 쪼갠 토큰을 숫자로 변환해줘야 함.  
-stopword removal(불용어 제거)
불필요한 단어를 의미하는 불용어 제거. 예를 들어, 아 휴, 아이쿠, 아이고 등.. 불용어는 활용 목적에 따라 정의할 수 있음.
**bag of words**: 자연어 데이터에 속해있는 단어들의 가방
자연어 데이터 ['안녕','만나서','반가워','나도'] => bag of words ['안녕': 0, '만나서': 1, '반가워': 2, '나도': 3] 이와 같이 인덱스를 매핑하는 방식은 큰 의미가 없고, 입력되는 순으로 인덱싱한다고 생각하면 됨.
**토큰 시퀀스**
안녕 만나서 반가워 -> 0 1 2
나도 만나서 반가워 -> 3 1 2
안녕 반가워 -> 0 2 4
4는 패딩의 의미를 가지며, 모든 문장의 길이를 맞추기 위해 기준보다 짧은 문장에는 패딩을 수행. 유독 긴 문장은 해당 문장을 빼주는 것이 좋음.

**영화 리뷰 긍정/부정 분류 RNN 모델 - 데이터 전 처리**
 자연어 자료는 곧 단어의 연속적인 배열로써, 시계열 자료
 sequence.pad_sequences(data, maxlen=300, padding='post') -> data 시퀀스의 크기가 maxlen 인자보다 작으면 그 크기에 맞게 패딩을 추가함. post 인자값은 패딩을 뒤에 붙이겠다는 의미.
 """
 1. 인덱스로 변환된 X_train, X_test 시퀀스에 패딩을 수행하고 각각 X_train, X_test에 저장합니다.
    시퀀스 최대 길이는 300으로 설정합니다.
 """
 X_train = sequence.pad_sequences(X_train, maxlen=300, padding='post')
 X_test = sequence.pad_sequences(X_test, maxlen=300, padding='post')

**단어 표현(word embedding)**
bag of words는 어떤 의미를 가진 인덱스를 가지는 것이 아니라 순서를 표현하는 인덱스임. bag of words의 인덱스로 정의된 토큰들에게 의미를 부여하는 방식.
embedding table을 참조해서 특정 인덱스에 해당하는 테이블을 가져옴. 예를 들어, 아버지:0 이면 [1,3,0,-2,0,0].. 어머니:1이면 [2,2,0,-1,0,0]
토큰의 특징을 벡터로 가져오는 것으로, 벡터를 사용하면 두 단어의 유사도를 측정할 수 있고, 연산이 가능함.
tf.keras.layers.Embedding(input_dim, output_dim, input_length)
input_dim: 들어올 단어의 개수
output_dim: 결과로 나올 임베딩 벡터의 크기(차원)
input_length: 들어오는 단어 벡터의 크기

자연어 문장을 기존 mlp 모델에 적용시키기에는 한계가 있음. 문장 벡터를 각 단어별로 순서대로 묶어주어야 문장의 의미를 제대로 가지고 있을 수 있는데, 이러한 문장 벡터를 해체시키고 입력값으로 설정하면 문장의 의미가 희미해질 수 있기 때문에.
이러한 문제를 해결하려면
순환 신경망(Recurrent Neural Network)을 사용함. 기존 퍼셉트론 계산과 비슷하게 x입력 데이터를 받아 y를 출력.
출력값을 두 갈래로 나뉘어 신경망에게 '기억'하는 기능을 부여함. 예를 들어, '안녕'이라는 데이터에 대한 임베딩 데이터 벡터를 입력하면 rnn을 거쳐서 결과물 y외에도 안녕이라는 데이터를 기억하고 있는 h값이 출력됨. 이러한 h 값이 두 번째 토큰이 '만나서'가 rnn으로 들어갈 때 같이 입력됨. 즉, 전에 사용했던 토큰에 대한 기억을 받아와서 다음 토큰의 계산 시 사용됨.
tf.keras.layers.SimpleRNN(units)

이 과정을 거친 각 레이어의 결과값들을  유닛이 1개인 fc레이어에 넣어서 결과값을 출력함
정리 : 임베딩(전처리, 특징을 뽑아냄) -> rnn(기억으로 인해 이 전 토큰의 영향을 받으면서 학습, 딥러닝) -> 활성함수로 분류(fc) $ 멀티 레이블의 경우 softmax, 바이너리일 경우, sigmoid, 회귀의 경우엔 액티브함수 사용할 필요 없음.
e.i) image captioning : cnn특징 -> rnn., chat bot 등..
rnn의 단점을 보완하는 LSTM, GRU, Transformer,...등 기술 발전 중.

**영화 리뷰 긍정/부정 분류 RNN 모델 - 모델 학습**
"""
1. 모델을 구현합니다.
   임베딩 레이어 다음으로 `SimpleRNN`을 사용하여 RNN 레이어를 쌓고 노드의 개수는 5개로 설정합니다.
   Dense 레이어는 0, 1 분류이기에 노드를 1개로 하고 activation을 'sigmoid'로 설정되어 있습니다.
"""
max_review_length = 300
embedding_vector_length = 32
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(1000, embedding_vector_length, input_length = max_review_length),
    tf.keras.layers.SimpleRNN(5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
#모델을 확인합니다.
print(model.summary())
#학습 방법을 설정합니다.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#학습을 수행합니다.
model_history = model.fit(X_train, y_train, epochs = 3, verbose = 2)

**영화 리뷰 긍정/부정 분류 RNN 모델 - 평가 및 예측하기**
"""
1. 평가용 데이터를 활용하여 모델을 평가합니다.
   loss와 accuracy를 계산하고 loss, test_acc에 저장합니다.
"""
loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)

"""
2. 평가용 데이터에 대한 예측 결과를 predictions에 저장합니다.
"""
predictions = model.predict(X_test)

#모델 평가 및 예측 결과를 출력합니다.
print('\nTest Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))
print('예측한 Test Data 클래스 : ',1 if predictions[0]>=0.5 else 0)
