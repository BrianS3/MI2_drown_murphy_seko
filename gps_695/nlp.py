
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
    from string import punctuation

    data = df
    data.dropna(inplace=True)

    # Remove photos and videos/usernames/digits/hashtags/URLs
    # make lowercase and strip excess spaces
    data = data[~data.TWEET_TEXT.str.contains("Just posted a")]  # posts that are only photos/videos
    data['TIDY_TWEET'] = [re.sub("@[\w]*", " ", item) for item in data['TWEET_TEXT']]  # usernames
    data['TIDY_TWEET'] = [re.sub("[0-9]", " ", item) for item in data['TIDY_TWEET']]  # digits
    #         data['TIDY_TWEET'] = [re.sub(r"\B#\w*[a-zA-Z]+\w*", "", item) for item in data['TIDY_TWEET']] # hashtags
    data['TIDY_TWEET'] = [re.sub(r"https?:\/\/.*[\r\n]*", "", item) for item in data['TIDY_TWEET']]  # URLs
    data['TIDY_TWEET'] = [re.sub(r"[^\w'\s]+", " ", item) for item in data['TIDY_TWEET']]  # punctuation
    data['TIDY_TWEET'] = [item.lower().strip() for item in data['TIDY_TWEET']]

    # Remove empty strings
    data['TIDY_TWEET'].replace('', np.nan, inplace=True)
    data = data.dropna()

    # Detect language and translate
    translator = Translator()
    for i, item in enumerate(data['TIDY_TWEET']):
        lang = detect(item)
        data.loc[i, 'TWEET_LANG'] = lang

        #         data = data[data.TWEET_LANG == 'en'] # removing non-English tweets

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

    for i, lemm in enumerate(df['LEMM']):
        lemm2 = " ".join(lemm)
        df.loc[i, 'LEMM'] = lemm2
        
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


def get_associated_keywords(df, search_term, components=2, returned_items=2):
    '''INPUT: df with TIDY_TWEET column
    components: integer value to use with n_components in NMF algorithm
    returned_items: integer value to specify how many keywords you want returned
    OUTPUT: list of strings representing associated keywords
    '''
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(min_df=50, stop_words='english')

    X = vect.fit_transform(df.TIDY_TWEET)
    model = NMF(n_components=components, random_state=42)
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
