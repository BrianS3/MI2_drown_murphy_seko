from gps_695 import database as d
from gps_695 import credentials as c
import os

# Get information about functions
help(c.create_env_variables)

# set up new environmental variables
db_user = ''
db_pass = ''
api_bearer = ''
db_host = ''
database = ''

c.create_env_variables(db_user=db_user, db_pass=db_pass, api_bearer=api_bearer, db_host=db_host, database=database)

#loading credentials to current environment
c.load_env_credentials()

print(os.getenv('mysql_username'))
print(os.getenv('mysql_pass'))
print(os.getenv('db_host'))
print(os.getenv('database'))
print(os.getenv('twitter_bearer'))

#connect to your database
d.connect_to_database()

#set up your database to recieve twitter data
d.create_mysql_database()

#reset your database
d.reset_mysql_database()

#make request to twitter api and return a json object
json_object = d.call_tweets(keyword='', start_date='', end_date='', results=)
