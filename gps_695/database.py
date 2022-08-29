def call_tweets(api_bearer, keyword, start_date, end_date, results):
    """
    Pulls tweets from research project API v2
    :param api_bearer: API bearer token
    :param keyword: keyword of tweet for API query
    :param start_date: start date of query, YYYY-MM-DD format, string
    :param end_date: end date of query,  YYYY-MM-DD format, string
    :param results: number of results to return, max 500, int
    :return: json object
    """
    import requests
    search_keyword = keyword
    search_start_date = start_date
    search_end_date = end_date
    search_results = results
    search_api_bearer = api_bearer
    url = f"https://api.twitter.com/2/tweets/search/all?query={search_keyword}&start_time={search_start_date}T00:00:00.000Z&end_time={search_end_date}T00:00:00.000Z&max_results={search_results}&tweet.fields=created_at,geo,text&expansions=attachments.poll_ids,attachments.media_keys,author_id,geo.place_id,in_reply_to_user_id,referenced_tweets.id,entities.mentions.username,referenced_tweets.id.author_id&place.fields=contained_within,country,country_code,full_name,geo,id,name&user.fields=created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,withheld"
    payload = {}
    headers = {
        'Authorization': f'Bearer {api_bearer}',
        'Cookie': 'guest_id=v1%3A166033111819561959; guest_id_ads=v1%3A166033111819561959; guest_id_marketing=v1%3A166033111819561959; personalization_id="v1_PKCeHmYzBzlDXBwYR96qjg=="'
    }
    response = requests.request("GET", url, headers=headers, data=payload).json()
    return response