def create_env_variables(db_user=None, db_pass=None, api_bearer=None, db_host=None, database=None):
    """
    Creates .env variables for database and API credentials.
    This saves an .env file to your current working directory.
    Leaving a variable as "None" will skip loading into env file. Use this to load
    variables after creating an initial .env file.
    :param db_user: username for mysql database
    :param db_pass: password for mysql database
    :param api_bearer: API bearer token
    :param db_host: server information for database
    :param database: name of mysql database, example "databse1" or "master"
    :return: None
    """
    if db_user != None:
        with open(".env", "a+") as f:
            f.write(f"mysql_username={db_user}")
            f.write("\n")
    if db_pass != None:
        with open(".env", "a+") as f:
            f.write(f"mysql_pass={db_pass}")
            f.write("\n")
    if api_bearer != None:
        with open(".env", "a+") as f:
            f.write(f"twitter_bearer={api_bearer}")
            f.write("\n")
    if db_host != None:
        with open(".env", "a+") as f:
            f.write(f"db_host={db_host}")
            f.write("\n")
    if database != None:
        with open(".env", "a+") as f:
            f.write(f"database={database}")
            f.write("\n")

def load_env_credentials():
    """
    Sets variables in .env file for current session.
    :return: None
    """
    import os
    with open(".env", "r") as f:
        for line in f.readlines():
            try:
                key, value = line.split('=')
                os.environ[key] = value.strip()
            except ValueError:
                # syntax error
                pass