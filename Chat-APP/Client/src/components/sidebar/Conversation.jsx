import { useSocketContext } from '../../context/SocketContext'
import useConversation from '../../zustand/useConversation'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
//import { formatDate } from '../../lib/utils'
import { MessageSeenSvg } from '../../lib/svgs'
import { ImageIcon, VideoIcon } from "lucide-react";

const Conversation = ({ conversation, lastIdx }) => {
  const { selectedConversation, setSelectedConversation } = useConversation()

  const isSelected = selectedConversation?._id === conversation._id
  const { onlineUsers } = useSocketContext()
  const isOnline = onlineUsers.includes(conversation._id)
  const lastMessage = conversation.lastMessage
  const lastMessageType = lastMessage?.messageType

  return (
    <>
      <div
        className={`flex gap-2 items-center hover:bg-chat-hover rounded p-2 py-1 cursor-pointer
				${isSelected ? 'bg-gray-primary' : ''}
			`}
        onClick={() => setSelectedConversation(conversation)}
      >
        <Avatar className='border overflow-visible relative'>
          {isOnline && (
            <div className='absolute top-0 right-0 w-2.5 h-2.5 bg-green-500 rounded-full border-2 border-foreground' />
          )}
          <AvatarImage
            src={conversation.profilePic || '/placeholder.png'}
            className='object-cover rounded-full'
          />
          <AvatarFallback>
            <div className='animate-pulse bg-gray-tertiary w-full h-full rounded-full'></div>
          </AvatarFallback>
        </Avatar>
        <div className='w-full'>
          <div className='flex items-center'>
            <h3 className='text-sm font-medium'>{conversation.fullName}</h3>
            <span className='text-xs text-gray-500 ml-auto'>
              {/* {formatDate(
                lastMessage?._createdAt || conversation._createdAt
              )} */}
            </span>
          </div>
          <p className='text-[12px] mt-1 text-gray-500 flex items-center gap-1 '>
            {lastMessage?.senderId?._id ? <MessageSeenSvg /> : ''}
            {lastMessageType === 'text' ? (
              lastMessage?.message.length > 30 ? (
                <span>{lastMessage?.message.slice(0, 30)}...</span>
              ) : (
                <span>{lastMessage?.message}</span>
              )
            ) : null}
            {lastMessageType === 'image' && <ImageIcon size={16} />}
            {lastMessageType === 'video' && <VideoIcon size={16} />}
          </p>
        </div>
      </div>
      {!lastIdx && <div className='my-0 py-2 h-1 ' />}
    </>
  )
}
export default Conversation
