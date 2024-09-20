setup:
	pip install -r requirements.txt

populate:
	./populate_database.py

question1:
	./query_data.py "how many players can play monopoly?"

question2:
	./query_data.py "when does the game ticket to ride end?"

question3:
	./query_data.py "in rust, how do you read a file?"
