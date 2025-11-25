"use client";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { useTheme } from "next-themes";
import { Sun, Moon } from 'lucide-react'

const ThemeSwitch = () => {
	const { setTheme } = useTheme();

	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild className='relative'>
				<label className="swap swap-rotate">
					<Sun className='h-[2rem] w-[2rem] text-white rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0' />
					<Moon className='absolute h-[2rem] w-[2rem] text-white rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100' />
				</label>
			</DropdownMenuTrigger>
			<DropdownMenuContent >
				<DropdownMenuItem onClick={() => setTheme("light")}>Light</DropdownMenuItem>
				<DropdownMenuItem onClick={() => setTheme("dark")}>Dark</DropdownMenuItem>
				<DropdownMenuItem onClick={() => setTheme("system")}>System</DropdownMenuItem>
			</DropdownMenuContent>
		</DropdownMenu>
	);
};
export default ThemeSwitch;
