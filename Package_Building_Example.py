import gps_695
from gps_695 import database
from gps_695 import credentials
# import os

# help(credentials.create_env_variables)
# credentials.create_env_variables()

credentials.load_env_credentials()
credential_list = credentials.get_mysql_credentials()

print(credential_list[0])
print(credential_list[2])

