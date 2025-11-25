import { Navigate, Route, Routes } from 'react-router-dom'
import './App.css'
import Home from './pages/home/Home'
import Login from './pages/login/Login'
import SignUp from './pages/signup/SignUp'
import { Toaster } from 'react-hot-toast'
import { useAuthContext } from './context/AuthContext'
import { ThemeProvider } from './providers/theme-provider'

function App () {
  const { authUser } = useAuthContext()
  return (
    <div className='h-screen w-screen flex items-center justify-center bg-slate-50 '>
      <ThemeProvider
        attribute='class'
        defaultTheme='system'
        enableSystem
        disableTransitionOnChange
      >
        <Routes>
          <Route
            path='/'
            element={authUser ? <Home /> : <Navigate to={'/login'} />}
          />
          <Route
            path='/login'
            element={authUser ? <Navigate to='/' /> : <Login />}
          />
          <Route
            path='/signup'
            element={authUser ? <Navigate to='/' /> : <SignUp />}
          />
        </Routes>
        <Toaster />
      </ThemeProvider>
    </div>
  )
}

export default App
