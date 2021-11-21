def option_1(): #test function, delete when importing the useful ones
	print("Predict stocks with current data")
	
	
	
def more_options(): #don't delete
	print("More options:\n \t 1: current day\n\t 2: last week\n\t 3: last month\n\t 4: YTD \n\t 5: last 6 months\n\t 6: last year")
	choice = input()
	choice = str(choice)
	if choice == "1":
		print("current day")
	elif choice == "2":
		print("last week")
	elif choice == "3":
		print("last month")
	elif choice == "4":
		print("YTD")
	elif choice == "5":
		print("last 6 months")
	elif choice == "6":
		print("last year")
	else:
		print("why")
		more_options()
def ask(): #don't delete
	print("1: Predict stocks with current data")
	print("2: More options")
	choice = input()
	choice = str(choice)
	if choice == "1":
		option_1()
	elif choice = "2":
		more_options()
	else:
		print("sorry, you have abused me for too long, I'm handing in my two weeks notice")
		ask()
		
		
if __name__ = '__main__':
	ask()
