import { useEffect } from "react";
import useConversation from "../../zustand/useConversation";
import MessageInput from "./MessageInput";
import Messages from "./Messages";
import { TiMessages } from "react-icons/ti";
import { useAuthContext } from "../../context/AuthContext";
import { useSocketContext } from '../../context/SocketContext';
import { X } from "lucide-react";


const MessageContainer = () => {
	const { selectedConversation, setSelectedConversation } = useConversation();
	const { onlineUsers } = useSocketContext();
	const isOnline = selectedConversation && onlineUsers.includes(selectedConversation._id);
	const profilePic = selectedConversation?.profilePic;

	useEffect(() => {
		// Cleanup function (unmounts)
		return () => setSelectedConversation(null);
	}, [setSelectedConversation]);

	return (
		<div className='w-3/4 flex flex-col'>
			{!selectedConversation ? (
				<NoChatSelected />
			) : (
				<>
					<div className='w-full sticky top-0 z-40'>
						{/* Header */}
						<div className='flex justify-between bg-gray-primary p-2'>
							<div className='flex gap-3 items-center'>
								<div className="avatar">
									<div className='w-12 rounded-full'>
										<img
											src={profilePic}
											alt='user avatar'
										/>
									</div>
								</div>
								<div className='flex flex-col'>
									<h3 className='font-semibold text-lg my-0 text-ellipsis line-clamp-1'>{selectedConversation.fullName}</h3>
									<p className='-my-1 text-sm'>{isOnline ? <span className='text-sky-500'>online</span> : <span className='text-slate-600'>offline</span>}</p>
								</div>
							</div>
							<div className='flex items-center gap-7 mr-5'>
								<X size={16} className='cursor-pointer' onClick={() => setSelectedConversation(null)} />
							</div>
						</div>
					</div>
					<Messages />
					<MessageInput />
				</>
			)}
		</div>
	);
};

export default MessageContainer;

const NoChatSelected = () => {
	const { authUser } = useAuthContext();
	return (
		<div className='flex items-center justify-center w-full h-full bg-gray-secondary flex-col py-10'>
			<div className='px-4 text-center sm:text-lg md:text-xl font-semibold flex flex-col items-center gap-2'>
				<p>Welcome 👋 {authUser.fullName}</p>
				<p>Select a chat to start messaging</p>
				<TiMessages className='text-3xl md:text-6xl text-center' />
			</div>
		</div>
	);
};
