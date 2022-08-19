def create_env_variables(db_user, db_pass, api_bearer):
    with open(".env", "w") as f:
        f.write(f"mysql_username={db_user}")
        f.write("\n")
        f.write(f"mysql_pass={db_pass}")
        f.write("\n")
        f.write(f"twitter_bearer={api_bearer}")

def load_env_credentials():
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
    import os
    (k,v) = (os.getenv('mysql_username'), os.getenv('mysql_pass'))
    return (k,v)

def get_twitter_api_bearer():
    import os
    api_b = os.getenv('twitter_bearer')
    return api_b