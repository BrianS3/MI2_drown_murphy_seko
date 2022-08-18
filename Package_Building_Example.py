import gps_695
from gps_695 import database_helpers
from gps_695 import package_helpers
# import os

# package_helpers.another_test()

#database_helpers.test()

package_helpers.create_env_variables()

package_helpers.load_env_credentials()
mysql_username, mysql_pass = package_helpers.get_mysql_user_pass()

print(mysql_username)
print(mysql_pass)