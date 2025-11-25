import { useEffect, useRef, useState } from 'react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '../ui/dropdown-menu'
import { ImageIcon, Plus, Video } from 'lucide-react'
import { Dialog, DialogContent, DialogDescription } from '../ui/dialog'
import ReactPlayer from 'react-player'
import toast from 'react-hot-toast'
import useConversation from '../../zustand/useConversation'
import axios from 'axios'
import FormData from 'form-data'
import { MdOutlineSignLanguage } from 'react-icons/md'
import { IoMdSend } from 'react-icons/io'
import { useLanguage } from '../../context/languageContext'

const MediaDropdown = () => {
  const imageInput = useRef(null)
  const videoInput = useRef(null)
  const [selectedImage, setSelectedImage] = useState(null)
  const [selectedVideo, setSelectedVideo] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [showOptions, setShowOptions] = useState(false)
  const { messages, setMessages, selectedConversation } = useConversation()
  const { selectedLanguage } = useLanguage()

  const handleSendImage = async () => {
    setIsLoading(true)
    const formData = new FormData()
    if (selectedImage) {
      formData.append('file', selectedImage)
    }

    try {
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
      setSelectedImage(null)
    } catch (error) {
      toast.error(error.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendVideo = async () => {
    setIsLoading(true)
    const formData = new FormData()
    if (selectedVideo) {
      formData.append('file', selectedVideo)
    }

    try {
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
      setSelectedVideo(null)
    } catch (error) {
      toast.error(error.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendTranslate = async () => {
    setLoading(true)
    const formData = new FormData()
    formData.append('model', selectedLanguage)
    if (selectedVideo) {
      formData.append('file', selectedVideo)
    } else {
      console.error('No video selected')
      return
    }

    try {
      const response = await axios.post(
        'http://localhost:5001/flask-api',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )

      const translate = response.data.result

      const newFormData = new FormData()
      newFormData.append('file', selectedVideo)
      newFormData.append('translate', translate)

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
      setSelectedVideo(null)
    } catch (error) {
      console.error('Error sending video:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <input
        type='file'
        ref={imageInput}
        accept='image/*'
        onChange={e => setSelectedImage(e.target?.files[0])}
        hidden
      />

      <input
        type='file'
        ref={videoInput}
        accept='video/mp4'
        onChange={e => setSelectedVideo(e.target?.files[0])}
        hidden
      />

      {selectedImage && (
        <MediaImageDialog
          isOpen={selectedImage !== null}
          onClose={() => setSelectedImage(null)}
          selectedImage={selectedImage}
          isLoading={isLoading}
          handleSendImage={handleSendImage}
        />
      )}

      {selectedVideo && (
        <MediaVideoDialog
          isOpen={selectedVideo !== null}
          onClose={() => setSelectedVideo(null)}
          selectedVideo={selectedVideo}
          isLoading={isLoading}
          loading={loading}
          handleSendVideo={handleSendVideo}
          handleSendTranslate={handleSendTranslate}
          showOptions={showOptions}
          setShowOptions={setShowOptions}
        />
      )}

      <DropdownMenu>
        <DropdownMenuTrigger>
          <Plus className='text-gray-600 dark:text-gray-400' />
        </DropdownMenuTrigger>

        <DropdownMenuContent>
          <DropdownMenuItem onClick={() => imageInput.current.click()}>
            <ImageIcon size={18} className='mr-1' /> Photo
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => videoInput.current.click()}>
            <Video size={20} className='mr-1' />
            Video
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  )
}

export default MediaDropdown

const MediaImageDialog = ({
  isOpen,
  onClose,
  selectedImage,
  isLoading,
  handleSendImage
}) => {
  const [renderedImage, setRenderedImage] = useState(null)

  useEffect(() => {
    if (!selectedImage) return
    const reader = new FileReader()
    reader.onload = e => setRenderedImage(e.target?.result)
    reader.readAsDataURL(selectedImage)
  }, [selectedImage])

  return (
    <Dialog
      open={isOpen}
      onOpenChange={isOpen => {
        if (!isOpen) onClose()
      }}
    >
      <DialogContent>
        <DialogDescription className='flex flex-col gap-10 justify-center items-center'>
          {renderedImage && (
            <img
              src={renderedImage}
              width={300}
              height={300}
              alt='selected image'
            />
          )}
          {isLoading ? (
            <div className='loading loading-spinner'></div>
          ) : (
            <IoMdSend
              className='size-8 text-black dark:text-white'
              onClick={handleSendImage}
            />
          )}
        </DialogDescription>
      </DialogContent>
    </Dialog>
  )
}

const MediaVideoDialog = ({
  isOpen,
  onClose,
  selectedVideo,
  isLoading,
  handleSendVideo,
  loading,
  handleSendTranslate,
}) => {
  const renderedVideo = URL.createObjectURL(
    new Blob([selectedVideo], { type: 'video/mp4' })
  )

  return (
    <Dialog
      open={isOpen}
      onOpenChange={isOpen => {
        if (!isOpen) onClose()
      }}
    >
      <DialogContent>
        <DialogDescription>Video</DialogDescription>
        <div className='w-full'>
          {renderedVideo && (
            <ReactPlayer url={renderedVideo} controls width='100%' />
          )}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          {loading ? (
            <div className='loading loading-spinner'></div>
          ) : (
            <MdOutlineSignLanguage onClick={handleSendTranslate} className='size-8' />
          )}
          {isLoading ? (
            <div className='loading loading-spinner'></div>
          ) : (
            <IoMdSend className='size-8' onClick={handleSendVideo} />
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
