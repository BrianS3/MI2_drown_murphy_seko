import gps_695
from gps_695 import database
from gps_695 import credentials
# import os

# credentials.create_env_variables()

help(credentials.create_env_variables)

credentials.load_env_credentials()
mysql_username, mysql_pass = credentials.get_mysql_user_pass()

print(mysql_username)
print(mysql_pass)

