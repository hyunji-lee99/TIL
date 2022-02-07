import { deleteDoc, doc, updateDoc } from '@firebase/firestore';
import React,{useState} from 'react';
import { dbService } from '../fbase';

function Nweet({nweetObj, isOwner}) {
  const [editing,setEditing]=useState(false)
  const [newNweet,setNewNweet]=useState(nweetObj.text)
  async function onDeleteClick(){
    const ok=window.confirm("Are you sure you want to delete this nweet?")
    if (ok){
      const NweetTextRef=doc(dbService,"nweets",`${nweetObj.id}`)
      await deleteDoc(NweetTextRef)
    }
  }
  const onChange=(e)=>{
    const {target:{value},}=e
    setNewNweet(value)
  };

  const onSubmit=async(e)=>{
    e.preventDefault();
    const NweetTextRef=doc(dbService,"nweets",`${nweetObj.id}`);
    await updateDoc(NweetTextRef,{text:newNweet,});
    setEditing(false);
  };
  const toggleEditting=()=>{setEditing(prev=>!prev)}
  return (
    <div>
      {editing? (<><form onSubmit={onSubmit}>
        <input type='text' placeholder='Edit your nweet' value={newNweet} required onChange={onChange}/>
        <input type='submit' value='Update Nweet'></input>
        </form>
        <button onClick={toggleEditting}>Cancel</button>
        </>
        ):
        (<>
        <h4>{nweetObj.text}</h4>
        {isOwner &&<><button onClick={onDeleteClick}>Delete Nweet</button>
        <button onClick={toggleEditting}>Edit Nweet</button></>}
        </>)}
    </div>
  );
}

export default Nweet;