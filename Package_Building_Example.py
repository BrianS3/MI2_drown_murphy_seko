import gps_695
from gps_695 import database
from gps_695 import package
# import os

# package_helpers.create_env_variables()

help(package_helpers.create_env_variables)

package_helpers.load_env_credentials()
mysql_username, mysql_pass = package_helpers.get_mysql_user_pass()

print(mysql_username)
print(mysql_pass)

