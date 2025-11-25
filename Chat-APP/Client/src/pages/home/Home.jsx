import MessageContainer from "../../components/messages/MessageContainer";
import Sidebar from "../../components/sidebar/Sidebar";
import NavSide from "../../components/sidebar/NavSide";
import { LanguageProvider } from '../../context/languageContext';
const Home = () => {
	return (
		<div className='flex  h-full w-full lg:flex-row'>
			<LanguageProvider>
				<NavSide />
				<Sidebar />
				<MessageContainer />
			</LanguageProvider>
		</div>
	);
};
export default Home;
