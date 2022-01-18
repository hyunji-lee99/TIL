# React에서 Django로 데이터 보내기(post)

우선, django에서의 설정은 다음 문서와 똑같이 해줍니다.
[Django api로 React에서 데이터받기](https://github.com/hyunji-lee99/TIL/blob/main/Django/project1.md)

frontend 디렉토리에 src/App.js의 코드를
다음과 같이 바꿔줍니다.
```
import React, {useEffect} from 'react';
import styled from 'styled-components';

function App() {


    useEffect(()=>{
    (async () => {
            const response =
                await fetch('http://127.0.0.1:8000/api/', {
                    method: 'POST',
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: (
                        JSON.stringify({
                            title:data.title,
                            content:data.content
                        })
                    ),
                })
                    .then(((result) => console.log(result)));
            }
    )();
    },[])

  return (
    <div>
    </div>
  );
}

const data=
        {
            "title":"test2",
            "content":"content2"
        }
export default App;
```

fetch api를 이용해서 위와 같이 데이터를 django로 전송해줍니다.
