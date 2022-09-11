
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
    RETURNS: df with lemmatized tweet column
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
    Get emotions: Happy, Angry, Surprise, Sad, Fear
    Get sentiments: Positive, Negative, Neutral, Compound
    OUTPUT: original df with added columns TWEET_EMO, OVERALL_EMO, TWEET_SENT_SCORES, TWEET_SENT
    '''
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

        for item in sorted_value_key_pairs[:1]:
            a = sorted_value_key_pairs[0][1]
            b = sorted_value_key_pairs[1][1]
            if a == b:
                predominant_emotion.append([sorted_value_key_pairs[0][0], sorted_value_key_pairs[1][0]])
            else:
                predominant_emotion.append(sorted_value_key_pairs[0][0])

    df['OVERALL_EMO'] = predominant_emotion

    #     # get sentiment scores and predominant tweet sentiment
    #     analyzer = SentimentIntensityAnalyzer()
    #     scores = []
    #     for item in df['TIDY_TWEET']:
    #         vs = analyzer.polarity_scores(item)
    #         scores.append(vs)
    #     df['TWEET_SENT_SCORES'] = scores

    #     predominant_sentiment = []
    #     for i, item in enumerate(df['TWEET_SENT_SCORES']):
    #         sort = sorted(item, key=item.get, reverse=True)
    #         predominant_sentiment.append(sort[:1])
    #     predominant_sentiment = [element for sublist in predominant_sentiment for element in sublist]
    #     df['TWEET_SENT'] = predominant_sentiment

    return df

