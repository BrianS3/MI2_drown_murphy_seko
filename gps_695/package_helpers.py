def create_env_variables():
    with open(".env", "w") as f:
        username = input("Enter DB Username: ")
        f.write(f"mysql_username={username}")
        f.write("\n")
        passw = input("Enter DB Password: ")
        f.write(f"mysql_pass={passw}")

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

def another_test():
    print('test')