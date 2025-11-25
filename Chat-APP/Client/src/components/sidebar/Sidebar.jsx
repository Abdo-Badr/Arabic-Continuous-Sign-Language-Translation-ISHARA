import Conversations from './Conversations';
import SearchInput from './SearchInput';

const Sidebar = () => {
  return (
    <div className='w-1/4 border-gray-600  p-4 flex flex-col sticky top-0 bg-left-panel z-10'>
      <h1 className="text-1xl font-bold leading-7 sm:truncate sm:text-3xl sm:tracking-tight p-4">Chats</h1>
      <SearchInput />
      <Conversations />
    </div>
  )
}
export default Sidebar;
