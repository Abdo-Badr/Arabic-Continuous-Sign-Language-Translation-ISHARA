import { useRef, useState, useEffect } from 'react'
import Webcam from 'react-webcam'
import { FaRedo, FaVideo, FaStop, FaCamera, FaPause, FaPlay } from 'react-icons/fa'
import { MdOutlineSignLanguage } from "react-icons/md";
import { IoMdSend } from "react-icons/io";
import axios from 'axios'
import toast from 'react-hot-toast'
import useConversation from '../../zustand/useConversation'
import { useLanguage } from '../../context/languageContext';

const Capture = ({ toggleModal }) => {
  const webcamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const [isLoading, setIsLoading] = useState(false);
  const [Loading, setLoading] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([])
  const [recordedVideoURL, setRecordedVideoURL] = useState(null)
  const [capturedImageURL, setCapturedImageURL] = useState(null)
  const { messages, setMessages, selectedConversation } = useConversation()
  const [isRecording, setIsRecording] = useState(false)
  const [isTakingPhoto, setIsTakingPhoto] = useState(true)
  const [isPaused, setIsPaused] = useState(false)
  const [timer, setTimer] = useState(0)
  const { selectedLanguage } = useLanguage();
  
  useEffect(() => {
    let timerInterval;
    if (isRecording && !isPaused) {
      timerInterval = setInterval(() => {
        setTimer(prevTimer => prevTimer + 1)
      }, 1000)
    }
    return () => clearInterval(timerInterval)
  }, [isRecording, isPaused])

  useEffect(() => {
    if (recordedChunks.length > 0) {
      const blob = new Blob(recordedChunks, { type: 'video/webm' })
      const url = URL.createObjectURL(blob)
      setRecordedVideoURL(url)
    }
  }, [recordedChunks])

  const startRecording = () => {
    const stream = webcamRef.current.stream
    mediaRecorderRef.current = new MediaRecorder(stream)
    mediaRecorderRef.current.ondataavailable = handleDataAvailable
    mediaRecorderRef.current.start()
    setIsRecording(true)
    setTimer(0)
    
    setTimeout(() => {
      stopRecording()
    }, 20000) // Stop recording after 20 seconds
  }

  const stopRecording = () => {
    mediaRecorderRef.current.stop()
    setIsRecording(false)
    setIsPaused(false)
  }

  const handleDataAvailable = ({ data }) => {
    if (data.size > 0) {
      setRecordedChunks(prevChunks => [...prevChunks, data])
    }
  }

  const retakeVideo = () => {
    setRecordedChunks([])
    setRecordedVideoURL(null)
  }

  const sendVideo = async () => {
    setIsLoading(true);
    try {
      const formData = new FormData()
      formData.append('file', new Blob(recordedChunks, { type: 'video/webm' }), 'recorded-video.webm')
      const result = await axios.post(
        `/api/messages/uploadFile/${selectedConversation._id}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      if (result.data.error) throw new Error(result.data.error)
      setMessages([...messages, result.data])
      toggleModal(); // Close modal after sending video
    } catch (error) {
      toast.error(error.message)
    } finally {
      setIsLoading(false);
    }
  }

  const sendTranslate = async () => {
    setLoading(true);
    try {
      const formData = new FormData()
      formData.append('file', new Blob(recordedChunks, { type: 'video/webm' }), 'recorded-video.webm')
      formData.append('model', selectedLanguage);
      const response = await axios.post(
        'http://localhost:5001/flask-api',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      const translate = response.data.result;
      
      const newFormData = new FormData();
      newFormData.append('file', new Blob(recordedChunks, { type: 'video/webm' }), 'recorded-video.webm');
      newFormData.append('translate', translate);
      const result = await axios.post(
        `/api/messages/uploadFile/${selectedConversation._id}`,
        newFormData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      if (result.data.error) throw new Error(result.data.error)
      setMessages([...messages, result.data])
      toggleModal(); // Close modal after sending translated video
    } catch (error) {
      toast.error(error.message)
    } finally {
      setLoading(false);
    }
  }

  

  const capturePhoto = () => {
    const imageSrc = webcamRef.current.getScreenshot()
    setCapturedImageURL(imageSrc)
  }

  const retakePhoto = () => {
    setCapturedImageURL(null)
  }

  const sendPhoto = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(capturedImageURL)
      const blob = await response.blob()
      const formData = new FormData()
      formData.append('file', blob, 'captured-photo.png')
      const result = await axios.post(
        `/api/messages/uploadFile/${selectedConversation._id}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      if (result.data.error) throw new Error(result.data.error)
      setMessages([...messages, result.data])
      toggleModal(); // Close modal after sending photo
    } catch (error) {
      toast.error(error.message)
    } finally {
      setIsLoading(false);
    }
  }

  const toggleMode = () => {
    setIsTakingPhoto(!isTakingPhoto)
  }

  const togglePause = () => {
    if (isPaused) {
      mediaRecorderRef.current.resume()
    } else {
      mediaRecorderRef.current.pause()
    }
    setIsPaused(!isPaused)
  }

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60)
    const seconds = time % 60
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  }

  return (
    <div>
      {recordedVideoURL ? (
        <div>
          <h1><b>Recorded Video</b></h1>
          <br />
          <video controls src={recordedVideoURL} />
          <br />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={retakeVideo}>
              <FaRedo className="size-5"/>
            </button>
                {Loading ? (
                  <div className='loading loading-spinner'></div>
                ) : (
                  <MdOutlineSignLanguage onClick={sendTranslate} className="size-8" />
                )}
             
            {isLoading ? (
              <div className='loading loading-spinner'></div>
            ) : (
              <button onClick={sendVideo}>
                <IoMdSend className="size-8"/>
              </button>
            )} 
          </div>
        </div>
      ) : capturedImageURL ? (
        <div>
          <h1><b>Captured Photo</b></h1>
          <br />
          <img src={capturedImageURL} alt="Captured" />
          <br />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={retakePhoto}>
              <FaRedo className="size-5"/>
            </button>
            {isLoading ? (
              <div className='loading loading-spinner'></div>
            ) : (
              <button onClick={sendPhoto}>
                <IoMdSend className="size-8"/>
              </button>
            )} 
          </div>
        </div>
      ) : (
        <>
          <div>
            <h1><b>Take a Video or Photo</b></h1>
          </div>
          <br />
          <Webcam audio={false} ref={webcamRef} width={640} height={480} screenshotFormat="image/png" />
          <br />
          <div style={{ display: 'flex', justifyContent: 'center', gap: '20px' }}>
            {isRecording ? (
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{ fontSize: '20px', marginRight: '20px' }}>
                  {formatTime(timer)}
                </div>
                <button onClick={stopRecording} style={circleStyle('#F44336')}>
                  <FaStop />
                </button>
                <button onClick={togglePause} className='m-6'>
                  {isPaused ? <FaPlay /> : <FaPause />}
                </button>
              </div>
            ) : (
              <>
                <button onClick={isTakingPhoto ? capturePhoto : startRecording} style={circleStyle('#4CAF50')}>
                  {isTakingPhoto ? <FaCamera /> : <FaVideo />}
                </button>
                <button onClick={toggleMode}>
                  {isTakingPhoto ? <FaVideo /> : <FaCamera />}
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}

const circleStyle = (color) => ({
  backgroundColor: color,
  border: 'none',
  borderRadius: '50%',
  width: '50px',
  height: '50px',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  fontSize: '24px',
  color: 'white',
  cursor: 'pointer'
})

export default Capture
