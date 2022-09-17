def connect_to_database():
    """
    Creates connection to mysql database.
    :return: connection objects
    """
    import mysql
    from mysql.connector import connect
    import os

    cnx = mysql.connector.connect(user=f"{os.getenv('mysql_username')}", password=f"{os.getenv('mysql_pass')}", host=f"{os.getenv('db_host')}", database=f"{os.getenv('database')}")
    return cnx

def create_mysql_database():
    """
    Function creates mysql database to store twitter data
    :return: None
    """
    from gps_695 import database as d
    import mysql
    # from mysql.connector import connect, Error

    file = open('gps_695/database_table_creation.sql', 'r')
    sql = file.read()
    file.close

    cnx = d.connect_to_database()
    cursor = cnx.cursor()

    try:
        cursor.execute(sql)
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))

def reset_mysql_database():
    """
    Function resets mysql database for new data loading.
    :return: None
    """
    from gps_695 import database as d
    import mysql
    # from mysql.connector import connect, Error

    file = open('gps_695/database_clear.sql', 'r')
    sql = file.read()
    file.close

    cnx = d.connect_to_database()
    cursor = cnx.cursor()

    try:
        cursor.execute(sql)
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))

    d.create_mysql_database()

def call_tweets(keyword, start_date, end_date, results):
    """
    Pulls tweets from research project API v2
    :param keyword: keyword of tweet for API query
    :param start_date: start date of query, YYYY-MM-DD format, string
    :param end_date: end date of query,  YYYY-MM-DD format, string
    :param results: number of results to return, max 500, int
    :return: json object
    """
    import requests
    import os

    search_api_bearer = os.getenv('twitter_bearer')
    url = f"https://api.twitter.com/2/tweets/search/all?query={keyword}&start_time={start_date}T00:00:00.000Z&end_time={end_date}T00:00:00.000Z&max_results={results}&tweet.fields=created_at,geo,text&expansions=attachments.poll_ids,attachments.media_keys,author_id,geo.place_id,in_reply_to_user_id,referenced_tweets.id,entities.mentions.username,referenced_tweets.id.author_id&place.fields=contained_within,country,country_code,full_name,geo,id,name&user.fields=created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,withheld"
    payload = {}
    headers = {
        'Authorization': f'Bearer {search_api_bearer}',
        'Cookie': 'guest_id=v1%3A166033111819561959; guest_id_ads=v1%3A166033111819561959; guest_id_marketing=v1%3A166033111819561959; personalization_id="v1_PKCeHmYzBzlDXBwYR96qjg=="'
    }
    response = requests.request("GET", url, headers=headers, data=payload).json()
    return response



def load_tweets(keyword, start_date, end_date, results = 500)
    """Pulls tweets from research project API v2
    :param keyword: keyword of tweet for API query
    :param start_date: start date of query, YYYY-MM-DD format, string
    :param end_date: end date of query,  YYYY-MM-DD format, string
    :param results: number of results to return, max 500, int
    :return: loads tweet data to DB"""
    from typing import List
    import pandas as pd
    from gps_695 import database as d
    from gps_695 import credentials as c
    import numpy as np
    from pandas import json_normalize
    import json
    import re
    import os
    import mysql
    from mysql.connector import connect, Error
    from datetime import date

    today = date.today()

    datafile = d.call_tweets(keyword, start_date, end_date, results)
    bbb = json.loads(datafile)

    df_data = pd.json_normalize(bbb['data'], max_level=5)
    retweet = df_data[df_data['referenced_tweets'].isna() == False]
    retweet['rt_id'] = retweet['referenced_tweets'].apply(lambda x: list(list(x)[0].values())[1])
    retweet = retweet[['id', 'rt_id']]

    df_text = df_data[['id', 'text']]
    df_text_merge = pd.merge(df_text, retweet, how='outer', left_on='id', right_on='id')
    df_retweet = pd.json_normalize(bbb['includes']['tweets'], max_level=1)
    df_retweet = df_retweet[['id', 'text']]
    df_retweet.rename(columns={'id': 'rt_id_includes', 'text': 'rt_text'}, inplace=True)
    df_text_final = pd.merge(df_text_merge, df_retweet, how='outer', left_on='rt_id', right_on='rt_id_includes')
    df_text_final = df_text_final[['id', 'text', 'rt_id', 'rt_text']]
    df_text_final.rename(
        columns={'id': 'TWEET_ID', 'text': 'TWEET', 'rt_id': 'RETWEET_ID', 'rt_text': 'RETWEET_TEXT'},
        inplace=True)

    column_list = list(df_text_final.columns)

    failed = pd.DataFrame(columns=column_list)


    cnx = d.connect_to_database()
    cursor = cnx.cursor()

    count_inserted = 0
    count_skipped = 0

    print('loading...')
    for ind, row in df_text_final.iterrows():
        try:
            try:
                query = (f"""
                INSERT INTO TWEET_TEXT (TWEET_ID, TWEET, RETWEET_ID, RETWEET_TEXT, SEARCH_TERM)
                VALUES (
                "{row[column_list[0]]}"
                ,"{row[column_list[1]]}"
                ,"{row[column_list[2]]}"
                ,"{row[column_list[3]]}"
                ,"{keyword}"
                );
                """)
                cursor.execute(query)
                count_inserted += 1
            except:
                query = (f"""
                INSERT INTO TWEET_TEXT (TWEET_ID, TWEET, RETWEET_ID, RETWEET_TEXT, SEARCH_TERM)
                VALUES (
                '{row[column_list[0]]}'
                ,'{row[column_list[1]]}'
                ,'{row[column_list[2]]}'
                ,'{row[column_list[3]]}'
                ,"{keyword}"
                );
                """)
                cursor.execute(query)
                count_inserted += 1
        except:
            count_skipped += 1
            # failed.loc[len(failed.index)] = list(row)
            continue
    cnx.commit()

    # print("length of dataframe: ", len(df_text_final))
    # print("inserted: ", count_inserted)
    # print("skipped: ", count_skipped)

    #failed.to_excel(pathway)