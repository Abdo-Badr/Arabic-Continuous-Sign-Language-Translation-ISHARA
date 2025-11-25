import { Link } from "react-router-dom";
import GenderCheckbox from "./GenderCheckbox";
import { useState } from "react";
import useSignup from "../../hooks/useSignup";
import gifosig from "../../assets/Images/sign.gif";
import "../../pages/signup/sign.css";
const SignUp = () => {
  const [inputs, setInputs] = useState({
    fullName: "",
    username: "",
    password: "",
    confirmPassword: "",
    gender: "",
  });
  const { loading, signup } = useSignup();
  const handleCheckboxChange = (gender) => {
    setInputs({ ...inputs, gender });
  };
    const handleSubmit = async (e) => {
    e.preventDefault();
    await signup(inputs);
  };
  return (
    <div className="container  items-center justify-center">
      <div className="form  p-8 bg-blue-500">
        <h1 className="text-3xl font-semibold text-center text-gray-300">
          <span className="text" style={{ color: 'white' }}>Sign Up </span>
        </h1>
        <form onSubmit={handleSubmit}>
          <label className="label p-2" htmlFor="fullName">
            Full Name
          </label>
          <input
            className="l_input input-bordered  h-8  "
            placeholder="John Doe"
            type="text"
            id="fullName"
            value={inputs.fullName}
            onChange={(e) => setInputs({ ...inputs, fullName: e.target.value })}
          />
          <br />
          <label className="label p-2 " htmlFor="username" style={{ color: 'white' }}>
            Email
          </label>
          <input
            type="text"
            placeholder="johndoe"
            id="username"
            className="l_input input-bordered h-8  "
            value={inputs.username}
            onChange={(e) => setInputs({ ...inputs, username: e.target.value })}
          />
          <br />
          <label className="label" htmlFor="password" style={{ color: 'white' }}>
            Password
          </label>
          <input
            type="password"
            id="password"
            placeholder="Enter Password"
            className="l_input input-bordered h-8  "
            value={inputs.password}
            onChange={(e) => setInputs({ ...inputs, password: e.target.value })}
          />
          <br />
          <label className="label" htmlFor="confirmPassword" style={{ color: 'white' }}>
            Confirm Password
          </label>
          <input
            type="password"
            id="confirmPassword"
            placeholder="Confirm Password"
            className="l_input input-bordered h-8  "
            value={inputs.confirmPassword}
            onChange={(e) =>
              setInputs({ ...inputs, confirmPassword: e.target.value })
            }
          />
          <br />
          <br></br>
          <GenderCheckbox
            onCheckboxChange={handleCheckboxChange}
            selectedGender={inputs.gender}
          />
          <br></br>
          <div className="btt">
            <Link
              to={"/login"}
              className="text-sm hover:underline hover:text-blue-50 mt-2 inline-block"
              href="#"
              style={{ color: 'white' }}
            >
              Already have an account?
            </Link>
          </div>
          <div className="btt">
            {" "}
            <button
              className="btn btn-neutral btn-sm mt-2 border border-slate-700 w-80 justify-center"
              disabled={loading}
              style={{ color: 'white' }}
            >
              {" "}
              {loading ? (
                <span className="loading loading-spinner"></span>
              ) : (
                "Sign Up"
              )}{" "}
            </button>{" "}
          </div>
        </form>
      </div>
      <div className="video">
        <img src={gifosig}></img>
      </div>
    </div>
  );
};
export default SignUp;
