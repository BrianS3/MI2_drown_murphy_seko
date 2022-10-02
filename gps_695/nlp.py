def clean_tweets(df):
    '''
    Removes tweets that are sent when a person posts a video or photo only;
    removes URLS, username mentions from tweet text;
    removes non-english tweets;
    isolates hashtags;
    :param df: Pandas DataFrame from database data.
    :return: original df with added columns TIDY_TWEET, TWEET_LANG, HASHTAGS
    '''
    import re
    from googletrans import Translator
    import numpy as np

    data = df
    data.dropna(inplace=True)

    # Remove photos and videos/usernames/digits/URLs
    # make lowercase and strip excess spaces
    data = data[~data.TWEET_TEXT.str.contains("Just posted a")]  # posts that are only photos/videos
    data['TIDY_TWEET'] = [re.sub("@[\w]*", "", item) for item in data['TWEET_TEXT']]  # usernames
    data['TIDY_TWEET'] = [re.sub("[0-9]", "", item) for item in data['TIDY_TWEET']]  # digits
    data['TIDY_TWEET'] = [re.sub(r"https?:\/\/.*[\r\n]*", "", item) for item in data['TIDY_TWEET']]  # URLs
    data['TIDY_TWEET'] = [re.sub(r"[^\w'\s]+", "", item) for item in data['TIDY_TWEET']]  # punctuation
    data['TIDY_TWEET'] = [re.sub(r"RT ", "", item) for item in data['TIDY_TWEET']]  # if initial tweet by user is labeled as a retweet
    data['TIDY_TWEET'] = data['TIDY_TWEET'].str.lower().str.strip() # extra spaces

    # Remove empty strings
    data['TIDY_TWEET'].replace('', np.nan, inplace=True)
    data = data.dropna()

    # Detect language and translate
    translator = Translator()
    for i, item in enumerate(data['TIDY_TWEET']):
        lang = translator.detect(item).lang
        data.loc[i, 'TWEET_LANG'] = lang

    data = data[data.TWEET_LANG == 'en']  # removing non-English tweets

    # Remove empty strings
    data['TIDY_TWEET'].replace('', np.nan, inplace=True)
    data = data.dropna()
    data.reset_index(inplace=True, drop=True)

    # isolate hashtags
    hashtags = []
    for item in data['TWEET_TEXT']:
        hashtags.append(re.findall(r"\B#\w*[a-zA-Z]+\w*", item.lower().strip()))
    data['HASHTAGS'] = hashtags

    return data

def lemmatize(df):
    '''INPUT: df with tidy_tweet column
    tokenizes;
    removes stopwords and lingering URLS;
    lemmatizes
    RETURNS: original df with added LEMM column
    '''
    from nltk.corpus import stopwords

    # tokenize
    tokens = []
    for i, item in enumerate(df['TIDY_TWEET']):
        tokens.append(item.split())

    # remove URLs
    for item in tokens:
        for thing in item:
            if 'http' in thing:
                item.remove(thing)

    # remove stopwords
    from nltk.corpus import stopwords
    
    stop_words = set(stopwords.words('english'))
    new_tokens = []
    for item in tokens:
        new = [i for i in item if not i in stop_words]
        new_tokens.append(new)

    # lemmatize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    lemms = []
    for item in new_tokens:
        l = []
        for word in item:
            lemm = lemmatizer.lemmatize(word.lower())
            l.append(lemm)
        lemms.append(l)

    df['LEMM'] = lemms


def analyze_tweets(df):
    '''
    Analyzes tweets for sentiment analysis
    Get emotions: Happy, Angry, Surprise, Sad, Fear, *Neutral, *Mixed
    *determined by top emotions(highest=0: neutral; highest>1: mixed
    :param df: Pandas DataFrame with TIDY_TWEET column
    :return: df with columns TWEET_ID, OVERALL_EMO
    '''
    import text2emotion as te

    # get emotion scores and predominant tweet emotion(s)
    emos = []
    for item in df['TIDY_TWEET']:
        emo = te.get_emotion(item)
        emos.append(emo)
    df['TWEET_EMO'] = emos

    predominant_emotion = []
    pred_emo_score = [] ##
    for item in df['TWEET_EMO']:
        sort_by_score_lambda = lambda score: score[1]
        sorted_value_key_pairs = sorted(item.items(), key=sort_by_score_lambda, reverse=True)

        emos = []
        emo_scores = [] ##
        if sorted_value_key_pairs[0][1] == 0:
            emos.append('Neutral')
            emo_scores.append(0) ##
        else:
            emos.append(sorted_value_key_pairs[0][0])
            emo_scores.append(sorted_value_key_pairs[0][1]) ##
        for i, item in enumerate(sorted_value_key_pairs):
            a = sorted_value_key_pairs[0][1]
            if sorted_value_key_pairs[i][1]==a and i!=0 and a!=0:
                emos.append(sorted_value_key_pairs[i][0])

        predominant_emotion.append(emos)
        pred_emo_score.append(emo_scores) ##
        
    for i, item in enumerate(predominant_emotion):
        if len(item)>1:
            predominant_emotion[i] = ['Mixed']

    predominant_emotion = [element for sublist in predominant_emotion for element in sublist]
    pred_emo_score = [element for sublist in pred_emo_score for element in sublist] ##
    df['OVERALL_EMO'] = predominant_emotion
    df['OVERALL_EMO_SCORE'] = pred_emo_score

def get_associated_keywords(df, search_term, returned_items=3, perc_in_words=0.1, **kwargs):
    '''
    Function finds the associated keywords from the initial data load
    :param df: df with LEMM column
    :param search_term: the search term associated with the news event/tweets
    :param returned_items: integer value to specify how many keywords max you want returned
    :param perc_in_words: the smallest threshold required for word frequency. If this is set to 10%, then 10% of all words must have the terms.
    :param **kwargs: keyword arguments from sklearn NMF model for grid search
    Lowering this value produces more variety.
    :return: list of strings representing keywords associated with search_term
    '''
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pandas as pd

    df2 = df.copy()
    for ind, row in df2.iterrows():
        lemm2 = "".join(row['LEMM'].replace("[", "").replace("'", "").replace("]", '').replace(",", ""))
        row['LEMM'] = lemm2

    # Perform NMF unsupervised learning to find topics
    vect = TfidfVectorizer(min_df=int(np.round(perc_in_words * len(df))), stop_words='english', ngram_range=(2, 2))

    # term must appear in 10% of tweets, looking for bigrams
    try:
        X = vect.fit_transform(df2.LEMM)
        model = NMF(**kwargs, max_iter=1000)  # , random_state=42)
        model.fit(X)
        components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names_out())
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            associated_keywords = list(tmp.nlargest().index)
            for word in associated_keywords:
                if search_term in word:
                    associated_keywords.remove(word)
        return associated_keywords[:returned_items]

    except ValueError:
        return "Could not find associated topics."

def evaluate_keywords(search_term, keyword_list):
    '''
    Original search_term, list of associated search terms found by get_associated_keywords()
    Uses distance metric to determine the closest 2 terms to original search_term based on Google Trends
    :param search_term: search term used in primary data loading
    :param keyword_list: keyword list returned from get_associated_keywords
    :return: OUTPUT list of 2 closest terms (strings)
    '''
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    kw = keyword_list
    kw.insert(0, search_term)

    def check_trend(kw_list):
        """
        Uses google trend to build a simple line chart of the current trend by keyword/phrase
        :param: keyword: or phrase, or many keywords/phrases separated by commas. Must be strings.
        :return: dataframe with google trend data per term
        """
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
        data = pytrends.interest_over_time()
        data = data.reset_index()

        return data

    trend = check_trend(kw)

    trend_check = trend[kw[0]]
    kw.pop(0)
    out_data = pd.DataFrame(columns=['term', 'mse'])

    for i in range(0,len(kw)):
        trend_compare = trend[kw[i]]
        temp = pd.DataFrame(list(zip([kw[i]],[mean_squared_error(trend_check,trend_compare)])), columns=['term', 'mse'])
        out_data = pd.concat([out_data,temp])

    return out_data

def gridsearch(search_term):
    """
    Function performs a grid search to optimize the model parameters for obtaining associated keywords.
    :param search_term: search term used in load_tweets function
    :return: top two keywords as a list with lowest MSE from google trends, to search term.
    """
    import pandas as pd
    from gps_695 import database as d
    from gps_695 import nlp as n
    from itertools import product

    cnx = d.connect_to_database()
    query = f"""select LEMM from TWEET_TEXT where lower(SEARCH_TERM) = '{search_term}';"""
    df = pd.read_sql_query(query, cnx)

    alphas = [0, 0.5, 1]
    l1_ratios = [0, 5, 10]
    percents = [0.1, 0.05, 0.01]

    grid_search_params = list(product(alphas, l1_ratios, percents))

    param_df = pd.DataFrame(
        grid_search_params,
        columns=['alpha', 'l1_ratio', 'percents'])

    grid_search_results = pd.DataFrame()
    file = open('output_data/grid_search.txt', 'w')

    for ind, row in param_df.iterrows():
        alpha_val = row['alpha']
        l1_val = row['l1_ratio']
        perc_val = row['percents']

        file = open('output_data/grid_search.txt', 'a')
        file.write(f"{ind} -- alpha:{alpha_val}  l1_ratio:{l1_val} perc in words:{perc_val} \n")

        kw_list = n.get_associated_keywords(df, search_term, perc_in_words=perc_val.astype(float),
                                            alpha_W=alpha_val.astype(float), l1_ratio=l1_val.astype(int))
        file.write(f"{kw_list}")
        file.write("\n")
        file.close()

        grid_search_results = pd.DataFrame(columns=['term', 'mse'])

        for key in kw_list:
            if key in list(grid_search_results['term']):
                continue
            else:
                if kw_list != "Could not find associated topics.":
                    top_terms = n.evaluate_keywords(search_term=search_term, keyword_list=kw_list, search_results = grid_search_results)
                    grid_search_results = pd.concat([grid_search_results, pd.DataFrame(top_terms, columns=['term', 'mse'])])
                else:
                    continue

    grid_search_results = grid_search_results.drop_duplicates()
    grid_search_results.to_csv('output_data/grid_search.csv')
    grid_search_results = grid_search_results.sort_values('mse')
    associated_words = list(grid_search_results['term'].iloc[:2])

    return associated_words
