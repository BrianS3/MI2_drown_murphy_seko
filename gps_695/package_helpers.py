def create_env_variables(db_user, db_pass, api_bearer):
    """
    Creates .env variables for database and API credentials.
    This saves an .env file to your current working directory.
    :param db_user: username for mysql database
    :param db_pass: password for mysql database
    :param api_bearer: API bearer token
    :return: None
    """
    with open(".env", "w") as f:
        f.write(f"mysql_username={db_user}")
        f.write("\n")
        f.write(f"mysql_pass={db_pass}")
        f.write("\n")
        f.write(f"twitter_bearer={api_bearer}")

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

def get_mysql_user_pass():
    """
    Pull username and password from current env variables for mysql database.
    :return: Tuple of Username, Password
    """
    import os
    (k,v) = (os.getenv('mysql_username'), os.getenv('mysql_pass'))
    return (k,v)

def get_twitter_api_bearer():
    """
    Pull API bearer token from current env variables for Twitter API.
    :return: string, twitter bearer token.
    """
    import os
    api_b = os.getenv('twitter_bearer')
    return api_b