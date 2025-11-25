import { useState } from 'react'
import Logo from '../../assets/Images/logo.ico'
import ThemeSwitch from './theme-switch'
import LogoutButton from './LogoutButton'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '../ui/dropdown-menu'
import { useAuthContext } from '../../context/AuthContext'
import { useLanguage } from '../../context/languageContext' 
import Flag from 'react-svg-country-flags'

function NavSide () {
  const { authUser } = useAuthContext()
  const profilePic = authUser?.profilePic
  const { selectedLanguage, setSelectedLanguage } = useLanguage()
  const [isModalOpen, setIsModalOpen] = useState(false)

  const handleSelect = language => {
    setSelectedLanguage(language)
  }

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen)
  }

  console.log('Selected language:', selectedLanguage)

  return (
    <div className='width: 282px; height: 1080px;  border-gray-600 border-r p-4 flex flex-col items-center bg-blue-500 dark:bg-gray-900'>
      <div className='grid w-32 h-20 rounded place-content-center'>
        <img src={Logo} alt='Logo' />
      </div>
      <div className='m-1'>
        <button>
          <svg
            xmlns='http://www.w3.org/2000/svg'
            className='size-7'
            fill='none'
            viewBox='0 0 24 24'
            stroke='white'
          >
            <path
              strokeLinecap='round'
              strokeLinejoin='round'
              strokeWidth='2'
              d='M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z'
            />
          </svg>
        </button>
      </div>
      <div className='m-9'>
        <button onClick={toggleModal}>
          <svg
            xmlns='http://www.w3.org/2000/svg'
            className='size-7'
            fill='none'
            viewBox='0 0 24 24'
            stroke='white'
          >
            <path
              strokeLinecap='round'
              strokeLinejoin='round'
              strokeWidth='2'
              d='M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z'
            />
          </svg>
        </button>
        {isModalOpen && (
          <div
            className=' fixed  inset-0 z-50 flex items-center justify-center bg-gray-900 bg-opacity-50'
            onClick={toggleModal}
          >
            <div
              className='sett rounded-lg p-6 text-center artboard phone-3 border-radius:10px; bg-blue-500 dark:bg-gray-900 '
              style={{ borderRadius: '30px', height: '600px' }}
              onClick={e => e.stopPropagation()}
            >
              <form className='card-body'>
                <div className='w-10 rounded-full'>
                  <button>
                    <img src={profilePic} alt='user avatar' />
                  </button>
                  <p style={{ color: 'white' }}>Change_profilePic</p>
                </div>

                <div className='form-control'>
                  <label className='label'>
                    <span className='label-text' style={{ color: 'white' }}>
                      Edit your UserName
                    </span>
                  </label>
                  <input
                    type='text'
                    placeholder='New UserName'
                    className='input input-bordered bg-indigo-100'
                    required
                  />
                </div>
                <div className='form-control'>
                  <label className='label'>
                    <span className='label-text' style={{ color: 'white' }}>
                      Edit Password
                    </span>
                  </label>
                  <input
                    type='password'
                    placeholder='New password'
                    className='input input-bordered bg-indigo-100'
                    required
                  />
                </div>
                <div className='form-control'>
                  <label className='label'>
                    <span className='label-text' style={{ color: 'white' }}>
                      Confirm Password
                    </span>
                  </label>
                  <input
                    type='password'
                    placeholder='New password'
                    className='input input-bordered bg-indigo-100'
                    required
                  />
                </div>
                <br></br>
                <div className='form-control mt-6'>
                  <button className='btn btn-neutral'>Save</button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
      <ThemeSwitch />
      <div className='m-9'>
        <DropdownMenu>
          <DropdownMenuTrigger asChild className='relative'>
            <label className='swap swap-rotate'>
              {selectedLanguage === 'Egypt' ? (
                <Flag
                  country='EG'
                  width='100px'
                  height='auto'
                  className='h-[2rem] w-[2rem]'
                />
              ) : (
                <Flag
                  country='QA'
                  width='100px'
                  height='auto'
                  className='h-[2rem] w-[2rem] '
                />
              )}
            </label>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => handleSelect('Egypt')}>
              <Flag country='EG' width='100px' height='auto' />
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleSelect('Qatar')}>
              <Flag country='QA' width='100px' height='auto' />
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <LogoutButton />
      <br />
      <div className='flex gap-3 items-center'>
        <div className='avatar'>
          <div className='w-10 rounded-full'>
            <img src={profilePic} alt='user avatar' />
          </div>
        </div>
        <div className='flex flex-col'>
          <p style={{ color: 'white' }}>{authUser.fullName}</p>
        </div>
      </div>
    </div>
  )
}

export default NavSide
