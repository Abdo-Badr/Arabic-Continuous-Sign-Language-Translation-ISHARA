import { useState } from 'react'
import useSendMessage from '../../hooks/useSendMessage'
import { Smile, Camera } from 'lucide-react'
import { IoMdSend } from 'react-icons/io'
import EmojiPicker, { Theme } from 'emoji-picker-react'
import useComponentVisible from '../../hooks/useComponentVisible'
import { Button } from '../ui/button'
import MediaDropdown from './media-dropdown'
import Capture from './Capture'

const MessageInput = () => {
  const [message, setMessage] = useState('')
  const { loading, sendMessage } = useSendMessage()
  const { ref, isComponentVisible, setIsComponentVisible } =
    useComponentVisible(false)

  const handleSubmit = async e => {
    e.preventDefault()
    if (!message) return
    await sendMessage(message)
    setMessage('')
  }

  const [isModalOpen, setIsModalOpen] = useState(false)

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen)
  }

  return (
    <div className='bg-gray-primary p-2 flex gap-4 items-center'>
      <div className='relative flex gap-2 ml-2'>
        {/* EMOJI PICKER WILL GO HERE */}
        <div ref={ref} onClick={() => setIsComponentVisible(true)}>
          {isComponentVisible && (
            <EmojiPicker
              theme={Theme.DARK}
              onEmojiClick={emojiObject => {
                setMessage(prev => prev + emojiObject.emoji)
              }}
              style={{
                position: 'absolute',
                bottom: '1.5rem',
                left: '1rem',
                zIndex: 50
              }}
            />
          )}
          <Smile className='text-gray-600 dark:text-gray-400' />
        </div>
        <MediaDropdown />
        <Camera onClick={toggleModal} />
        {isModalOpen && (
          <div
            className='fixed inset-0 z-50 flex items-center justify-center bg-gray-900 bg-opacity-50'
            onClick={toggleModal}
          >
            <div
              className=' rounded-lg p-6 text-center w-1/3 border-radius:10px; bg-left-panel'
              onClick={e => e.stopPropagation()}
            >
              <Capture toggleModal={toggleModal} />
            </div>
          </div>
        )}
      </div>
      <form className='w-full flex gap-3' onSubmit={handleSubmit}>
        <div className='flex-1'>
          <input
            type='text'
            className='py-2 text-sm w-full h-full outline-none bg-gray-primary focus-visible:ring-transparent'
            placeholder='Type a message'
            value={message}
            onChange={e => setMessage(e.target.value)}
          />
        </div>
        <div className='mr-4 flex items-center gap-3'>
          <Button
            type='submit'
            size={'sm'}
            className='bg-transparent text-foreground hover:bg-transparent'
          >
            {loading ? (
              <div className='loading loading-spinner'></div>
            ) : (
              <IoMdSend className='size-8' />
            )}
          </Button>
        </div>
      </form>
    </div>
  )
}
export default MessageInput
