
def clean_tweets(df):
    '''
    INPUT: Pandas DataFrame from database data.
    Removes tweets that are sent when a person posts a video or photo only;
    removes URLS, username mentions from tweet text;
    translates non-English Tweets;
    isolates hashtags;
    OUTPUT: original df with added columns TIDY_TWEET, TWEET_LANG, HASHTAGS
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
        lang = translator.detect(item)
        data.loc[i, 'TWEET_LANG'] = lang

        # translating non-English tweets
        if data['TWEET_LANG'][i] != 'en':
            translated = translator.translate(item)
            data.loc[i, 'TIDY_TWEET'] = translated.text

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

# create cleaned tweet dataframe

# function to get lemmatized tweets from clean tweets
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
    INPUT: Pandas DataFrame with TIDY_TWEET column
    Get emotions: Happy, Angry, Surprise, Sad, Fear, *Neutral, *Mixed
    *determined by top emotions(highest=0: neutral; highest>1: mixed
    OUTPUT: df with columns TWEET_ID, OVERALL_EMO
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


##### UNSUPERVISED LEARNING MODEL #####

def get_associated_keywords(df, search_term, returned_items=3,topic_nums_input = 5,perc_in_words=0.1, **kwargs):
    '''
    INPUT: df with LEMM column
    search_term: the search term associated with the news event/tweets
    returned_items: integer value to specify how many keywords max you want returned
    topic_nums_input: the number of topics to model from LDA
    perc_in_words: the smallest threshold required for word frequency. If this is set to 10%, then 10% of all words must have the terms.
    Lowering this value produces more variety.
    OUTPUT: list of strings representing keywords associated with search_term
    '''
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pandas as pd

    # Find best number of components to use

    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    from operator import itemgetter

    search_term = search_term.lower()

    #split to avoid data leakage
    part_80 = df.sample(frac = 0.8).reset_index(drop=True) # for NMF topic selection
    part_20 = df.drop(part_80.index).reset_index(drop=True) #for LDA n_components selection
    
    # Convert to bag of words
    lemms = part_20.LEMM
    texts = []
    for lemm in lemms:
        texts.append(lemm.replace('"', '').replace("'", '').split())
    dictionary = corpora.Dictionary(texts)
    topic_nums = list(np.arange(1, (topic_nums_input+1)))  # tested up to 100 topics, with same results as 5 topics (but significantly slower)

    corpus = [dictionary.doc2bow(text) for text in texts]

    # Get coherence scores with LDA
    coherence_scores = []
    for num in topic_nums:
        model = LdaModel(corpus, num, dictionary)  # , random_state=42)
        cm = CoherenceModel(model=model, texts=texts, corpus=corpus, coherence='c_v')

    coherence_scores.append(round(cm.get_coherence(), 5))

    # Get number of components with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

    # Turn list of lemmatized words into a string for analysis
    df2 = part_80.copy()
    for ind, row in df2.iterrows():
        lemm2 = "".join(row['LEMM'].replace("[", "").replace("'", "").replace("]", '').replace(",", ""))
        row['LEMM'] = lemm2

    #     df2 = df2.sample(500) # testing to see if 500 tweets is enough--results are the same top 3 as 1000

    # Perform NMF unsupervised learning to find topics
    vect = TfidfVectorizer(min_df=int(np.round(perc_in_words * len(df))), stop_words='english', ngram_range=(2, 2))

    # term must appear in 10% of tweets, looking for bigrams
    try:
        X = vect.fit_transform(df2.LEMM)
        model = NMF(n_components=best_num_topics, **kwargs)  # , random_state=42)
        # max_iter tested @ default(200), 500, 1000, 10k: all resulted in the same output
        model.fit(X)
        nmf_features = model.transform(X)

        components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names_out())
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            associated_keywords = list(tmp.nlargest().index)
            for word in associated_keywords:
                if search_term in word:
                    associated_keywords.remove(word)
        return associated_keywords[:]

    except ValueError:
        return "Could not find associated topics."

def evaluate_keywords(search_term, keyword_list):
    '''
    INPUT: original search_term, list of associated search terms found by get_associated_keywords()
    Uses distance metric to determine the closest 2 terms to original search_term based on Google Trends
    OUTPUT: list of 2 closest terms (strings)
    '''
    from sklearn.metrics import mean_squared_error
    search_term = search_term.lower()
    if len(keyword_list) < 3:
        return keyword_list
    
    kw = keyword_list

    def check_trend(kw_list):
        """
        Uses google trend to build a simple line chart of the current trend by keyword/phrase
        :param keyword: keyword or phrase, or many keywords/phrases separated by commas. Must be strings.
        :return: dataframe with google trend data per term
        """
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        try:
            pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
        except:
            print("OOF, looks like this is a heavy run, we need to wait for the API to rest")
            print("API Reset in:")
            countdown()
            print("Alright, let's continue")
            pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
        data = pytrends.interest_over_time()
        data = data.reset_index()

        return data[kw_list[0]]

    def countdown(t=100):
        """
        Timer for user to countdown for API reset
        :param t: input time, default is 100 seconds
        :return:
        """
        while t:
            mins, secs = divmod(t, 60)
            timer = '{:02d}:{:02d}'.format(mins, secs)
            print(timer, end="\r")
            time.sleep(1)
            t -= 1

    search_trend_list = [search_term]
    search_trend = check_trend(search_trend_list)

    trend_dict = {}
    for k in kw:
        temp_list = []
        temp_list.append(k)
        trend_dict[k] = check_trend(temp_list)

    final = []
    for k,v in trend_dict.items():
        final.append((k, round(mean_squared_error(search_trend,trend_dict[k]),4)))

    return final


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
    import os
    os.mkdir("output_data/")

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

    count = 0
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

        if kw_list != "Could not find associated topics.":
            top_terms = n.evaluate_keywords(search_term, kw_list)
            grid_search_results = pd.concat([grid_search_results, pd.DataFrame(top_terms, columns=['term', 'mse'])])
            count += 1
        else:
            count += 1
            continue

    grid_search_results = grid_search_results.drop_duplicates()
    grid_search_results = grid_search_results.sort_values('mse')
    associated_words = list(grid_search_results['term'].iloc[:2])

    return associated_words
