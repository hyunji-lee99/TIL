import { createUserWithEmailAndPassword, signInWithEmailAndPassword, GoogleAuthProvider,signInWithPopup } from '@firebase/auth';
import React, { useState } from 'react';
import { authService } from '../fbase';

function Auth() {
  const [email,setEmail]=useState('');
  const [password,setPassword]=useState('');
  const [newAccount,setNewAccount]=useState(true);
  const [error,setError]=useState('');
    const onChange=(e)=>{
    const {target: {name,value}}=e;
    if(name==='email'){
      setEmail(value)
    }
    else if(name==='password'){
      setPassword(value)
    }
  }

  const onSubmit=async(e)=>{
    e.preventDefault();
    try{
      if(newAccount){
        //create account
        const data=await createUserWithEmailAndPassword(authService,email,password)
      }
      else{
        //log-in
        const data=await signInWithEmailAndPassword(authService,email,password)
      }
    }catch(error){ 
        setError(error.message)
    }  
  }

  function toggleAccount(){
    setNewAccount((prev)=>!prev)
  }
  const onSocialClick= async(e)=>{
    //console.log(e.target.name)  
    const {
      target: {name},
    }=e;
    let provider;
    if(name==="google"){
      provider=new GoogleAuthProvider();
    }
    await signInWithPopup(authService, provider);
  }
  return (
    <div>
      <form onSubmit={onSubmit}>
        <input name='email' type='text' placeholder='Email' required value={email} onChange={onChange}></input>
        <input name='password' type='password' placeholder='password' required value={password} onChange={onChange}></input>
        <input type='submit' value={newAccount? 'create Account':'log in'}></input>
      </form>
      {error}
      <span onClick={toggleAccount}>{newAccount? "log-in":"create account"}</span>
      <div>
          <button name="google" onClick={onSocialClick}>Continue with Google</button>
      </div>
      </div>
  );
}

export default Auth;