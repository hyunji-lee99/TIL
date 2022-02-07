import React from 'react';
import { useHistory } from 'react-router';
import {authService} from '../fbase';
function Profile(props) {
  const history=useHistory(); 
  // 특정 url에서만 location 변경을 해야 할때 사용하는 hook, 일괄 location 변경은 redirect 사용
  const onLogoutClick=()=>{
    authService.signOut();
    history.push("/");
  }
  return (
    <>
      <button onClick={onLogoutClick}>Log Out</button>
    </>
  );
}

export default Profile;