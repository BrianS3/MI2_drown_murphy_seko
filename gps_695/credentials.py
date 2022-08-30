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
                os.environ[key] = value
            except ValueError:
                # syntax error
                pass

def get_mysql_credentials():
    """
    Pull database credentails from current env variables for mysql database.
    :return: List of database credentials
    """
    import os
    credentials = [os.getenv('mysql_username'),
                   os.getenv('mysql_pass'),
                   os.getenv('db_host'),
                   os.getenv('database')]
    return credentials

def get_twitter_api_bearer():
    """
    Pull API bearer token from current env variables for Twitter API.
    :return: string, twitter bearer token.
    """
    import os
    api_b = os.getenv('twitter_bearer')
    return api_b