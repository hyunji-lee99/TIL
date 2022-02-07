import React,{useEffect, useState} from 'react';
import {dbService} from '../fbase';
import {collection, addDoc, serverTimestamp, getDocs, onSnapshot, query, orderBy, doc} from 'firebase/firestore';
import Nweet from '../components/Nweet';

function Home({userObj}) {
  const [tweet,setTweet]=useState("")
  const [tweets,setTweets]=useState([])
  const [attachment, setAttachment]=useState()
  
  const getTweets=async()=>{
    const dbTweets=await getDocs(collection(dbService,"nweets")) //documnet들을 전부 가져옴.
    dbTweets.forEach(document=>{
      const tweetObject={
        ...document.data(),
        id:document.id,
      }
      setTweets(prev => [tweetObject,...prev]); 
      //dbTweet 안에 있는 document들에 하나씩 모두 접근하면서 이전의 tweet값인 prev를 documennt.data()에 붙여서 setTweets해줌. 
    })
  }
  useEffect(() => {
    const q = query(
      collection(dbService, "nweets"),
      orderBy("createdAt", "desc")
    );
    onSnapshot(q, (snapshot) => {
      const newArr = snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
      }));
      setTweets(newArr)
      console.log(tweets)
    }); 
  },[]);

  
  const onSubmit=async(e)=>{
    e.preventDefault()
    await addDoc(collection(dbService,"nweets"),{
      text: tweet,
      createdAt:serverTimestamp(),
      createid:userObj.uid,
    })
    setTweet("")
  }
  const onChange=(e)=>{
    const {target:{value},}=e; 
    // e 안에 target 값은 value로 달라고 요청 
    setTweet(value)
  }
  const onFileChange=(e)=>{
      const {target:{files},}=e;
      const theFile=files[0];
      const reader=new FileReader();
      reader.onloadend=(finishEvent)=>{
        const {
          currentTarget: {result},
        }=finishEvent;
        setAttachment(result)
      }
      reader.readAsDataURL(theFile);
      console.log(theFile)
  }  
  const ClearPhoto=()=>{
    setAttachment(null)
  }
  return (
    <div>
      <form>
        <input value={tweet} onChange={onChange} type="text" placeholder="What's on your mind?" maxLength={120}></input>
        <input type='file' accept='image/*' onChange={onFileChange}></input>
        <input onClick={onSubmit} type="submit"  value="Tweet"></input>
        {attachment?
        <div>
        <img src={attachment} width='50px' height='50px'></img>
        <button onClick={ClearPhoto}>Clear</button>
        </div>:
        null}
      </form>
      <div>
        {tweets.map((data)=>(
        <Nweet key={data.id} nweetObj={data} isOwner={data.createid === userObj.uid}/>
        ))}
      </div>
    </div>
  );
}

export default Home;