import React from 'react'
import {HashRouter as Router,Redirect,Route,Switch} from 'react-router-dom'
import Auth from '../routes/Auth';
import Home from '../routes/Home';
import Profile from '../routes/Profile';
import Navigation from './Navigation';

function AppRouter({isLoggedIn, userObj}){
  console.log(isLoggedIn)
  return(
    <Router>
      {isLoggedIn && <Navigation/>}
      {/* Navigation이 존재하려면 isLoggedIn이 true여야 한다는 의미. */}
      <Switch>
        {isLoggedIn?
        (
          <>
        <Route exact path='/'>
          <Home userObj={userObj}/>
        </Route>
        <Route exact path='/profile'>
          <Profile/>
        </Route>
        {/* <Redirect from="*" to="/"></Redirect>  */}
        {/* 경로가 "/"일 땐 상관없지만, 이 외에 경로는 새로고침하면 모두 "/"로 redirect시킴.  */}
        </>
        ):
        (
        <>
        <Route exact path='/'>
          <Auth/>
        </Route>
        {/* <Redirect from="*" to="/"></Redirect>  */}
        </>
        )}
      </Switch>
    </Router>
  )
}

export default AppRouter;