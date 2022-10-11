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