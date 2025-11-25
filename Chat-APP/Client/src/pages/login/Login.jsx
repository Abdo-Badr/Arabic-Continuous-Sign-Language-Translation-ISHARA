import { useState } from "react";
import { Link } from "react-router-dom";
import useLogin from "../../hooks/useLogin";
//import Videolog from "../../assets/Video/login.mp4";
import gifolog from "../../assets/Images/login.gif";
import "../../pages/login/login.css";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { loading, login } = useLogin();
  const handleSubmit = async (e) => {
    e.preventDefault();
    await login(username, password);
  };
  return (
    <div className="container  items-center justify-center ">
      <div className="form p-8 bg-blue-500	">
        <h1 className="text-3xl font-semibold text-center "style={{color:'white'}}>
          Login
          <span className="text" style={{color:'white'}}> IShara App</span>
        </h1>
        <form onSubmit={handleSubmit}>
          <div>
            <label className="label p-2">
              <span className="text-base label-text " style={{color:'white'}}>Email</span>
            </label>
            <input
              type="text"
              placeholder="Enter Email"
              className="l_input input-bordered h-10"
              value={username}
              onChange={(e) => setUsername(e.target.value)}

            />
          </div>
          <div>
        <label className="label">
              <span className="text-base label-text"style={{color:'white'}}>Password</span>
            </label>
            <input
              type="password"
              placeholder="Enter Password"
              className="l_input input-bordered h-10"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <br></br>
          <div className="btt">
          <Link
            to="/signup"
            className=" text-sm  hover:underline hover:text-blue-50 mt-2 inline-block justify-center"
            style={{ color: 'white' }}
          >
            {"Don't"} have an account? 
          </Link>
          </div>
          <div className="btt">
            <button className="btn btn-neutral btn-sm mt-2 border border-slate-700 w-80 justify-center" 
             disabled={loading}
             style={{ color: 'white' }}>
              {loading ? (
                <span className="loading loading-spinner " ></span>
              ) : (
                "Login"
              )}
            </button>
          </div>
        </form>
      </div>
      <div className="video">
        <img width='340' height='180' src={gifolog}>
		</img>
        </div>
    </div>
  );
};
export default Login;

