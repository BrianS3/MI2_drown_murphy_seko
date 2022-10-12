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
    
    
def streamgraph():
    '''
    Creates a streamgraph: counts of overall emotions by date
    :return: altair streamgraph visualization
    '''
    import altair as alt
    from altair_saver import save
    from gps_695 import database as d
    import pandas as pd
    import numpy as np

    cnx = d.connect_to_database()

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
    Creates a choropleth map of most common overall_emo by state, excluding 'Neutral' and 'Mixed'
    :return: None, image saved to "output_data" directory.
    '''
    from gps_695 import database as d
    import pandas as pd
    import plotly.express as px
    import numpy as np
    from collections import Counter

    cnx = d.connect_to_database()

    query = """
    SELECT * FROM TWEET_TEXT
    JOIN AUTHOR_LOCATION
    USING (AUTHOR_ID)
    JOIN US_STATES
    USING (STATE_ID);"""
    df = pd.read_sql_query(query, cnx)

    df = df.where(df.OVERALL_EMO != 'Mixed').dropna()
    df = df.where(df.OVERALL_EMO != 'Neutral').dropna()
    
    most_common_list = []
    for state in df['STATE_ABBR']:
        state_df = df.where(df.STATE_ABBR == state).dropna()
        counter = Counter(state_df['OVERALL_EMO'])
        most_common_list.append(counter.most_common()[0][0])

    df['MOST_COMMON_EMO'] = most_common_list
    df = df.sort_values('MOST_COMMON_EMO')

    fig = px.choropleth(df,
                        locations='STATE_ABBR', 
                        locationmode="USA-states", 
                        scope="usa",
                        color='OVERALL_EMO',
                        color_discrete_sequence=px.colors.qualitative.T10,
                        )

    fig.update_layout(
          title_text = f"Overall Emotion by State (of users with location listed), Search Terms: {np.unique(df['SEARCH_TERM'])}",
          title_font_size = 14,
          title_font_color="black", 
          title_x=0.45, 
             )

    fig.write_image('output_data/emo_choropleth.png')


def hashtag_chart():
    '''
    Creates a bar chart with top 10(max) hashtags
    :return: altair bar chart visualization
    '''
    import altair as alt
    import pandas as pd
    from altair_saver import save
    from collections import Counter
    from gps_695 import database as d
    
    cnx = d.connect_to_database()

    query = 'SELECT * FROM TWEET_TEXT;'
    df = pd.read_sql_query(query, cnx)
    

    hash_counts = Counter(df['HASHTAGS'])
    hash_counts.pop('[]')
    
    hash_df = pd.DataFrame.from_dict(hash_counts, orient='index', columns=['count'])
    hash_df = hash_df.reset_index()
    hash_df = hash_df.sort_values('count', ascending=False)
    hash_df = hash_df[:10]
    hash_df.reset_index(inplace=True, drop=True)


    bars = alt.Chart(hash_df, title=["Top Hashtags", f"Search Terms: {np.unique(df['SEARCH_TERM'])}"]).mark_bar().encode(
        y = alt.Y('index:N', sort='-x', axis=alt.Axis(grid=False, title='hashtag')),
        x = alt.X('count:Q', axis=alt.Axis(grid=False)),
        color = alt.Color('count:Q',scale=alt.Scale(scheme="goldorange"), legend=None)
    ).properties(height=175, width=250)
    
    save(bars, "output_data/hashtag_chart.html")
    
def forecast_chart():
    '''
    Creates a line chart with projected tweet volumes for next 10 days
    :return: altair bar chart visualization
    '''
    from gps_695 import database as d
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    import datetime
    import altair as alt
    from altair_saver import save

    cnx = d.connect_to_database()

    query = """
    SELECT COUNT(CREATED), CREATED FROM TWEET_TEXT
    GROUP BY CREATED"""
    df = pd.read_sql_query(query, cnx)
    df.drop(df.tail(1).index,inplace=True)
    x_dates = []
    df['CREATED']= pd.to_datetime(df['CREATED'])
    for i in range(10):
        x_dates.append(df['CREATED'].max() + datetime.timedelta(days=i+1))

    ARIMAmodel = ARIMA(df['COUNT(CREATED)'], order = (2, 0, 2))
    ARIMAmodel = ARIMAmodel.fit()
    y_pred = ARIMAmodel.get_forecast(10)
    y_pred_df = y_pred.conf_int(alpha = 0.05)
    y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df['CREATED'] = x_dates

    df = pd.concat([df,y_pred_df])
    lines = alt.Chart(df).mark_line().encode(
    x='CREATED',
    y = alt.Y('COUNT(CREATED)'),
    y2 = alt.Y('Predictions:Q')
    )
    plot_title = alt.TitleParams("Historical and Predicted Tweet Counts", subtitle=["Ten day ARIMA prediction of tweet volumes"])
    base = alt.Chart(df.reset_index(), title = plot_title).encode(alt.X('CREATED', title = "Tweet Date"))

    lines = alt.layer(
        base.mark_line(color='black').encode(alt.Y('COUNT(CREATED)')),
        base.mark_line(color='orange').encode(alt.Y('Predictions:Q', title = "Tweet Count"))
    )
    save(lines, "output_data/forecast_chart.html")

def interactive_tweet_trends():
    """
    Creates interactive chart of tweet trends and emotions over time.
    :return: None, chart is saved to output folder
    """
    import pandas as pd
    import altair as alt
    from gps_695 import database as d

    cnx = d.connect_to_database()

    df_volume = pd.read_sql_query("""
    SELECT
    created
    ,round(count(tweet_id)/count(distinct search_term)) as avg_vol
    FROM TWEET_TEXT
    group by created;
    """, cnx)

    df_sent = pd.read_sql_query("""
    select 
    created
    ,overall_emo
    ,count(overall_emo)/count(distinct search_term) as emo_count
    from tweet_text
    group by created, overall_emo;
    """, cnx)

    brush = alt.selection_interval(encodings=['x'])
    colorConditionDC = alt.condition(brush, alt.value('#2182bd'), alt.value('gray'))

    volume = alt.Chart(df_volume).mark_bar(color='grey').encode(
        x=alt.X('created', title="Date Created"),
        y=alt.Y('avg_vol:Q', title="Average Tweet Volume")
    ).properties(height=200, width=700)

    i_volume = volume.add_selection(brush).encode(color=colorConditionDC).resolve_scale(y='shared'
                                                                                        )
    sent_line = alt.Chart(df_sent).mark_line(size=2).encode(
        x=alt.X('created', title="Date Created"),
        y=alt.Y('emo_count:Q', title="Count of Emotions"),
        color=alt.Color('overall_emo', legend=None),
        tooltip='overall_emo'
    ).properties(height=200, width=700)

    selection = alt.selection_multi(fields=['overall_emo'])
    make_selector = alt.Chart(df_sent).mark_rect().encode(
        y=alt.Y('overall_emo', title=None),
        color='overall_emo'
    ).add_selection(selection).properties(title="Click to Filter")

    i_sent_line = sent_line.transform_filter(brush).resolve_scale(y='shared').transform_filter(selection)

    out = (i_volume & (i_sent_line | make_selector)).configure_range(
        category={'scheme': 'tableau10'}
    ).properties(
        title={
            "text": ["Tweet Volume and Sentiment Over Time"],
            "subtitle": ["",
                         "I'm Interactive, select a section on the bar chart to zoom in on sentiment values",
                         "Select an emotion to focus", ""]}
    )

    save(out, "output_data/interactive_tweet_trends.html")
