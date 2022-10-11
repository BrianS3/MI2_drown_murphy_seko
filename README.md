# [Package Name]

[Package name] was designed to simplify how twitter data is analyzed. This package will create a mysql database and load twitter data to it. It will also perform a sentiment analysis on the tweets, encouraging users to run analyze new data frequently. Simply enter your keyword or phrase and let the package do the rest.

This package uses unsupervised machine learning to understand what words are associated with your keyword or phrase and add to your content search by pulling tweets by these related words. Tweets are pulled from the day before a run is executed up to 25 days prior.

Supervised learning is used to expedite the package runtime. Only a subset of words are analyzed using [text2emotion](https://pypi.org/project/text2emotion/), the rest are predicted with [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) via scikit learn

# Getting Started

## Setting up MySQL
To get started you need to first setup a MySQL database. If you have never done this, check out a [this tutorial](https://www.youtube.com/watch?v=u6vJEpRB_zc&t=1s) for more information.

Once you have MySQL server installed, use the following commands to setup a blank database and user for access.

In mysql command line create user:

```create user 'your_user' identified by 'password';```

Then setup the database:

```Create database your_db;```

Grant privilege's to user, these are the minimum required to run this package:

```grant alter,create,delete,drop,index,insert,select,update, references on your_db.* to 'your_user';```

## Initialize user credentials
All credentials for this package are stored in an .env file for convenience. This file will be created in your current working directory. If that directory has an existing .env file, <b><u>this package will overwrite it.</b></u>

### Loading credentials:

```
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

#you can load
c.create_env_variables(db_user=db_user, db_pass=db_pass, api_bearer=api_bearer, db_host=db_host, database=database)
```

Once loaded you can change the name of any variable as needed.

```
c.create_env_variables(database='new_db_name')
```

Once the .env file is created, use the following code to load your credentials per session.

```
c.load_env_credentials()
```

# Example code 

## Do a full database load

Want to dive right in and just start analyzing tweets? Run the following:

```
from gps_695 import credentials as c
from gps_695 import database as d

c.load_env_credentials()
d.database_load("hello")
```

## Creating your database

This package will automatically create a database when you choose to execute ```database.database_load("keyword")```, however you can do this process on your own at any time.

```
from gps_695 import credentials as c
from gps_695 import database as d

c.load_env_credentials()
d.reset_mysql_database()
```

## Partial database loads

If you want to load a single database load, you can. Be advised this will not perform any sentiment analysis.

```
from gps_695 import credentials as c
from gps_695 import database as d

c.load_env_credentials()
d.load_tweets("hello", '2020-01-01', '2020-02-01', 50)
```

## Database connections

If you wish to manually connect to your database and extract data you can do so as followed:

```
from gps_695 import database as d
from gps_695 import credentials as c
import pandas as pd

c.load_env_credentials()

query = "SELECT * FROM TWEET_TEXT;"

df = pd.read_sql_query(query, cnx)
```

# Analyzing your data

# FAQs
1) How long does the package take to run?

    Package run times are usually between 15-30 minutes, depending on your internet connection speed. Run times are subject to the complexity of tweets, where higher volume and more complex speech will increase runtime.

    If you are using a local MySQL database, runtime will be lower (generally).

2) I use "X" database (not MySQL), will this package work?

    No.

3) Why did you choose MySQL? "X" is so much better...

    Because I wrote this package and you didn't. Jokes aside, MySQL was chosen for simplicity and ease of setup. Future iterations of this package may include more connectors, but for now MySQL was the simplest choice in our opinion to get you moving quickly. MySQL had no obvious benefits of Maria DB, Postgres, or any other open source database software.

# Errors

Common errors that you may encounter:

1) Users may experience issues with the NLKT library. Corpora needed to process text may not be properly installed.

    See https://www.nltk.org/data.html for more information to find instruction on how to download the correct corpus. The traceback message will contain the missing file, such as:
    "Resource omw-1.4 not found."

    ![](assets\nltk_error_pic_1.png)

2) Freeze Support

    ```
    freeze_support()

    An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
    The "freeze_support()" line can be omitted if the program is not going to be frozen to produce an executable.
    ```


    If you experience this error nest your code under

    ```if __name__ == "__main__":```