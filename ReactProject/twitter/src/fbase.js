// Import the functions you need from the SDKs you need
import {initializeApp} from 'firebase/app'
import {getAuth} from 'firebase/auth'
import {getFirestore} from 'firebase/firestore'


// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAOu5COoM4ehmNwXm9e9rF6zcBxSWFW6W4",
  authDomain: "nwitter-10779.firebaseapp.com",
  projectId: "nwitter-10779",
  storageBucket: "nwitter-10779.appspot.com",
  messagingSenderId: "886762334994",
  appId: "1:886762334994:web:dec87ed818590dcc1c61cb"
};

// Initialize Firebase
const app=initializeApp(firebaseConfig)
export const authService = getAuth()
export const dbService = getFirestore()