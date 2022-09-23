
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
        
    return df


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
    for item in df['TWEET_EMO']:
        sort_by_score_lambda = lambda subject_score_pair: subject_score_pair[1]
        sorted_value_key_pairs = sorted(item.items(), key=sort_by_score_lambda, reverse=True)

        emos = []
        if sorted_value_key_pairs[0][1] == 0:
            emos.append('Neutral')
        else:
            emos.append(sorted_value_key_pairs[0][0])
        for i, item in enumerate(sorted_value_key_pairs):
            a = sorted_value_key_pairs[0][1]
            if sorted_value_key_pairs[i][1] == a and i != 0 and a != 0:
                emos.append(sorted_value_key_pairs[i][0])
        predominant_emotion.append(emos)

    for i, item in enumerate(predominant_emotion):
        if len(item) > 1:
            predominant_emotion[i] = ['Mixed']
    predominant_emotion = [element for sublist in predominant_emotion for element in sublist]

    df['OVERALL_EMO'] = predominant_emotion

    return df[['TWEET_ID', 'OVERALL_EMO']]


##### UNSUPERVISED LEARNING MODEL #####

def get_associated_keywords(df, search_term, returned_items=3):
    '''INPUT: df with LEMM column
    search_term: the search term associated with the news event/tweets
    returned_items: integer value to specify how many keywords max you want returned
    OUTPUT: list of strings representing keywords associated with search_term
    '''
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    
   
    # Find best number of components to use

    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    from operator import itemgetter

    # Convert to bag of words
    texts = df['LEMM']
    dictionary = corpora.Dictionary(texts)
    topic_nums = list(np.arange(1, 6)) # tested up to 100 topics, with same results as 5 topics (but significantly slower)
    
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Get coherence scores with LDA
    coherence_scores = []
    for num in topic_nums:
        model = LdaModel(corpus, num, dictionary)#, random_state=42)
        cm = CoherenceModel(model=model, texts=texts, corpus=corpus, coherence='c_v')
    
    coherence_scores.append(round(cm.get_coherence(), 5))
    

    # Get number of components with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
   
    # Turn list of lemmatized words into a string for analysis
    df2=df.copy()
    for i, lemm in enumerate(df2['LEMM']):
        lemm2 = " ".join(lemm)
        df2.loc[i, 'LEMM'] = lemm2

#     df2 = df2.sample(500) # testing to see if 500 tweets is enough--results are the same top 3 as 1000

    # Perform NMF unsupervised learning to find topics
    vect = TfidfVectorizer(min_df=int(np.round(0.1*len(df))), stop_words='english', ngram_range=(2,2))
    
    # term must appear in 10% of tweets, looking for bigrams
    try:
        X = vect.fit_transform(df2.LEMM)
        model = NMF(n_components=best_num_topics)#, random_state=42)
        # max_iter tested @ default(200), 500, 1000, 10k: all resulted in the same output
        model.fit(X)
        nmf_features = model.transform(X)

        components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names_out())
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            associated_keywords = list(tmp.nlargest().index)
            for word in associated_keywords:
                if word == search_term:
                    associated_keywords.remove(word)
        return associated_keywords[:returned_items]

    except ValueError:
        return "Could not find associated topics."

    
    
    
def evaluate_keywords(search_term, keyword_list):
    '''
    INPUT: original search_term, list of associated search terms found by get_associated_keywords()
    Uses distance metric to determine the closest 2 terms to original search_term based on Google Trends
    OUTPUT: list of 2 closest terms (strings)
    '''
    
    if len(keyword_list) < 3:
        return keyword_list
    
    kw = keyword_list
    kw.insert(0,search_term)
    
    term_dict = {'AB':kw[1], 'AC':kw[2], 'AD':kw[3]}
    
    def check_trend(kw_list):
        """
        Uses google trend to build a simple line chart of the current trend by keyword/phrase
        :param keyword: keyword or phrase, or many keywords/phrases separated by commas. Must be strings.
        :return: dataframe with google trend data per term
        """
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
        data = pytrends.interest_over_time()
        data = data.reset_index()

        return data

    trend = check_trend(kw)

    a = trend[kw[0]]
    b = trend[kw[1]]
    c = trend[kw[2]]
    d = trend[kw[3]]
    
    from sklearn.metrics import mean_squared_error
    trend['AB'] = mean_squared_error(a,b)
    trend['AC'] = mean_squared_error(a,c)
    trend['AD'] = mean_squared_error(a,d)
    
    sum_dict = {'AB':trend['AB'].sum(), 'AC':trend['AC'].sum(), 'AD':trend['AD'].sum()}
    sort_sum_dict = sorted(sum_dict.items(), key=lambda x:x[1])
    
    top_scoring = [sort_sum_dict[:2][0][0], sort_sum_dict[:2][1][0]]
    top_terms = [term_dict[item] for item in top_scoring]
    
    return top_terms
