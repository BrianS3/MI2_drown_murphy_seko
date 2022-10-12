def check_trend(*args):
    """
    Uses google trend to build a simple line chart of the current trend by keyword/phrase
    :param keyword: keyword or phrase, or many keywords/phrases separated by commas. Must be strings.
    :return: creates a plotly image which generates from .show()
    """
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [*args]
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
    data = pytrends.interest_over_time()
    data = data.reset_index()

    import plotly.express as px
    fig = px.line(data, x="date", y=kw_list, title='Keyword Web Search Interest Over Time')
    fig.show()
    
    
def streamgraph(df):
    '''
    Creates a streamgraph: counts of overall emotions by date
    :return: altair streamgraph visualization
    '''
    import altair as alt
    from altair_saver import save
    from gps_695 import database as d
    import pandas as pd

    try:
        cnx = d.connect_to_database()
    except:
        print('Credentials not loaded, use credentials.load_env_credentials()')

    query = 'SELECT * FROM TWEET_TEXT;'
    df = pd.read_sql_query(query, cnx)

    alt.data_transformers.disable_max_rows()

    chart = alt.Chart(df, title=f"Search Terms: {np.unique(df['SEARCH_TERM'])}").mark_area().encode(
        alt.X('CREATED:T',
            axis=alt.Axis(domain=False, grid=False, tickSize=0)
        ),
        alt.Y('count(OVERALL_EMO):N', stack='center',
             axis=alt.Axis(domain=False, grid=False, tickSize=0)),
        alt.Color('OVERALL_EMO:N',
            scale=alt.Scale(scheme='tableau10')
        )
    ).properties(width=500).configure_view(strokeOpacity=0)

    save(chart, "output_data/streamgraph.html")


def emo_choropleth():
    '''
    Creates a choropleth map of overall_emo by state
    :return: None, image saved to "output_data" directory.
    '''
    from gps_695 import database as d
    import pandas as pd
    import plotly.express as px
    import numpy as np

    try:
        cnx = d.connect_to_database()
    except:
        print('Credentials not loaded, use credentials.load_env_credentials()')

    query = """
    SELECT * FROM TWEET_TEXT
    JOIN AUTHOR_LOCATION
    USING (AUTHOR_ID)
    JOIN US_STATES
    USING (STATE_ID);"""
    df = pd.read_sql_query(query, cnx)

    fig = px.choropleth(df,
                        locations='STATE_ABBR', 
                        locationmode="USA-states", 
                        scope="usa",
                        color='OVERALL_EMO',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        )

    fig.update_layout(
          title_text = f"Overall Emotion by State (of users with location listed), Search Terms: {np.unique(df['SEARCH_TERM'])}",
          title_font_size = 14,
          title_font_color="black", 
          title_x=0.45, 
             )

    fig.write_image('output_data/choro_overall_emo.png')


def hashtag_chart():
    '''
    INPUT: df with specific columns
    Creates a bar chart with top 10(max) hashtags
    OUTPUT altair bar chart visualization
    '''
    import altair as alt
    import pandas as pd
    from altair_saver import save
    from collections import Counter
    from gps_695 import database as d
    
    try:
        cnx = d.connect_to_database()
    except:
        print('Credentials not loaded, use credentials.load_env_credentials()')

    query = 'SELECT * FROM TWEET_TEXT;'
    df = pd.read_sql_query(query, cnx)
    

    hash_counts = Counter(df['HASHTAGS'])
    hash_counts.pop('[]')
    
    hash_df = pd.DataFrame.from_dict(hash_counts, orient='index', columns=['count'])
    hash_df = hash_df.reset_index()
    hash_df = hash_df.sort_values('count', ascending=False)
    hash_df = hash_df[:10]
    hash_df.reset_index(inplace=True, drop=True)


    bars = alt.Chart(hash_df, title="Top Hashtags").mark_bar().encode(
        y = alt.Y('index:N', sort='-x', axis=alt.Axis(grid=False, title='hashtag')),
        x = alt.X('count:Q', axis=alt.Axis(grid=False)),
        color = alt.Color('count:Q',scale=alt.Scale(scheme="goldorange"), legend=None)
    ).properties(height=175, width=250)
    
    save(bars, "output_data/hashtag_bars.html")
    
