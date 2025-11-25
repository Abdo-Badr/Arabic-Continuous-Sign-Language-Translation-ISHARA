import { useAuthContext } from "../../context/AuthContext";
import { extractTime } from "../../utils/extractTime";

const Message = ({ message }) => {
    const { authUser } = useAuthContext();
    const fromMe = message.senderId === authUser._id;
    const formattedTime = extractTime(message.createdAt);
    const chatClassName = fromMe ? "chat-end" : "chat-start";
    const bubbleBgColor = fromMe ? "bg-blue-500" : "bg-gray-700";
    const shakeClass = message.shouldShake ? "shake" : "";
    const profilePic = authUser?.profilePic;

    let content = null;

    switch (message.type) {
        case "text":
            if (/\b(?:https?|ftp):\/\/\S+/gi.test(message.message)) {
                // If the message contains a link
                content = (
                    <a href={message.message} target="_blank" rel="noopener noreferrer" className={`chat-bubble text-white ${bubbleBgColor} ${shakeClass} pb-2 mr-2 text-sm font-light underline link link-success`}>
                        {message.message}
                    </a>
                );
            } else {
                // If the message is regular text
                content = <div className={`chat-bubble text-white ${bubbleBgColor} ${shakeClass} pb-2`}>{message.message}</div>;
            }
            break;
        case "video":
            content = (
                <div className={`chat-bubble ${bubbleBgColor} ${shakeClass}`} style={{ maxWidth: '400px' }}>
                    <video src={`http://localhost:5000/${message.message}`} type="video/mp4" controls className="mb-2" alt="video" />
                    {message.translate && <div className="text-white mt-2 text-right">{message.translate}</div>}
                </div>
            );
            break;
        case "image":
            content = <img src={`http://localhost:5000/${message.message}`} alt="image" className={`chat-bubble ${bubbleBgColor} ${shakeClass}`} style={{ maxWidth: '400px' }} />;
            break;
        default:
            content = <div>Unsupported message type</div>;
    }

    return (
        <div className={`chat ${chatClassName}`}>
            <div className='chat-image avatar'>
                <div className='w-10 rounded-full'>
                    <img alt='Tailwind CSS chat bubble component' src={profilePic} />
                </div>
            </div>

            {content}
            <div className='chat-footer opacity-50 text-xs flex gap-1 items-center font-bold'>{formattedTime}</div>
        </div>
    );
};

export default Message;
