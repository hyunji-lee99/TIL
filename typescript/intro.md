#TypeScript
타입스크립트는 자바스크립트에 타입을 부여한 언어입니다. 자바스크립트의 superset으로, 자바스크립트처럼 생겼고 컴파일하면 자바스크립트로 컴파일되는 프로그래밍 언어입니다.
타입스크립트로 작성하는 것은 모두 자바스크립트로 변하며, 타입스크립트는 자바스크립트가 가지고 있지 않은 다양한 규칙들을 가지고 있기 때문에 사용하는 것입니다.
자바스크립트와 반대로 엄격한 규칙을 적용하고 훌륭한 자동완성 기능때문에 팀 프로젝트나 대형 프로젝트에서 버그를 줄이기 위해서 많이들 사용합니다.

우선 타입스크립트의 대략적인 느낌을 알아보기 위해서 간단하게 타입스크립트의 예시를 살펴보겠습니다.
typescript를 설치합니다.
```
yarn global add typescript
```
그리고 생선된 tsconfig.json파일에 다음을 작성해줍니다.
```{
  "compilerOptions": {
    "module": "commonjs", //nodejs를 normal하게 사용할 수 있도록 함.
    "target": "ES2015", //어떤 버전의 자바스크립트로 컴파일할지
    "sourceMap": true, //소스맵을 사용할지
    "outDir":"dist" //컴파일한 결과는 모두 dist 디렉토리에 저장
  },
  "include": ["src/**/*"], //어떤 파일이 컴파일과정에 포함될지 -> 타입스크립트는 모두 src안으로 들어감.
  "exclude": ["node_modules"] //어떤 파일이 컴파일 과정에 포함이 되지 않을지
}
```
index.ts로 가서
```
alert('hello')
```
를 작성하고 자바스크립트로 컴파일해봅니다.

터미널로 가서 ```tsc```를 입력하면 ts 파일을 컴파일해서 js 파일과 js.map파일을 만들어줍니다.
yarn으로 ts파일을 컴파일하고 실행하고 싶다면? package.json파일에
```
"scripts": {
    "start": "node index.js"
    "prestart":"tsc"
  },
```
를 작성해주면 yarn start 실행 시 ts를 컴파일하고, index.js를 실행합니다.

타입스크립트는 어떤 종류의 변수와 데이터를 사용할지 설정해줘야 합니다.
또, 어떤 변수나 데이터를 선언해주는 파일 가장 하단엔 반드시
<code>export {};</code> 를 작성해줘야 합니다.

index.ts에 다음을 작성해주면 에러가 발생합니다.
```
const name='hyunji', age=24, gender="male";
const sayhi= (name, age, gender) => {
  console.log(`hello ${name} you are ${age}, you are a ${gender}`);
}

sayhi(name,age); => 자바스크립트와 달리 에러 발생!
```
자바스크립트와 달리 타입에 대한 엄격한 규칙이 적용되기 때문입니다.
또는 sayhi 함수를 const sayhi= (name, age, gender?)로 바꿔주면 gender는 선택적인 변수로 사용합니다. 즉, 위와 같이 gender를 파라미터로 넘겨주지 않아도 에러가 발생하지 않는 것입니다.

```
const sayhi= (name:string, age:number, gender:string) => {
  console.log(`hello ${name} you are ${age}, you are a ${gender}`);
}

sayhi("hyunji","345","female"); => 에러 발생!
```
위와 같이 함수 파라미터의 변수를 설정해줄 수 있고, 함수 호출 시 그 변수 규칙을 따르지 않으면 에러가 발생합니다.

또, 함수의 리턴값의 자료형을 설정해줄 수 있습니다.
```
const sayhi= (name:string, age:number, gender:string):void => {
  return `hello ${name} you are ${age}, you are a ${gender}`;
}

sayhi("hyunji","345","female"); => void 타입의 함수에서 string 타입을 반환할 수 없다는 에러가 발생
```
그러면 void형 함수를 string형 함수로 바꿔주면 에러는 해결됩니다.

이와 같이 타입스크립트는 변수의 자료형이 자유로운 자바스크립트에서 각 함수와 변수 등 명확한 타입을 정의해주면서 자바스크립트를 사용하는 프로그래밍 언어라고 할 수 있습니다.

그리고, 기존의 package.json 파일에서
```
"scripts": {
    "start": "tsc-watch --onSuccess \" node dist/index.js\"
  },
```
를 변경해줍니다. 이 코드는 yarn start를 실행해서 ts 컴파일에 성공하면 dist/index.js 파일을 실행합니다. 여기서 watch 모드를 이용해서 src에서 무언가 바뀌면 dist도 바뀌게 됩니다.

## interface
우선, 간단한 인터페이스의 예시를 살펴봅시다.
```
interface Human {
  name: string;
  age: number;
  gender: string;
}

const person={
  name: 'hyunji';
  age: 24;
  gender:'male';
}

const sayhi=(person:Human)=>{
  return `hello ${person.name} you are ${person.age}, you are a ${person.gender}`;
};

console.log(sayhi(person));
```
인터페이스는 자바스크립트에선 작동하지 않습니다. 즉, 어떤 오브젝트를 파라미터로 넘겨줄 때 자료형을 선언해주는 것처럼 사용할 수 있습니다. 또, 인터페이스를 인자로 받아 사용할 때 항상 인터페이스의 속성 갯수와 인자로 받는 객체의 속성 갯수를 일치시키지 않아도 됩니다. 다시 말해, 인터페이스에 정의된 속성, 타입의 조건만 만족한다면 객체의 속성 갯수가 더 많아도 상관없다는 뜻입니다. 예시로 들자면, 아래 코드도 가능합니다. 
```
interface personAge{
  age: number;
}

function logAge(obj:personAge) {
  console.log(obj.age)
}

let person = {name:'hyunji', age:24}
logAge(person)
```


#class
```
class Human {
  public name: string;
  public age:number;
  public gender:string;
  constructor(name:string, age:number, gender:string){
    this.name=name;
    this.age=age;
    this.gender=gender;
  }
}

const lynn=new Human('lynn',18,'female');

const sayhi=(person)=>{
  return `hello ${person.name} you are ${person.age}, you are a ${person.gender}`;
};

console.log(sayhi(lynn));
```
다음과 같은 코드를 컴파일하면 index.js 파일엔 자동으로 클래스가 컴파일 되어 나타납니다. 상황에 따라 인터페이스를 사용할지, 클래스를 사용할 지 다르겠지만 리액트나 익스프레스 등의 프레임워크에선 클래스를 사용하게 될 확률이 높습니다.
