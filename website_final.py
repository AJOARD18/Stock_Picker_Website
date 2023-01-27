import streamlit as st
import pandas as pd
from PIL import Image
from plotly import graph_objs as go
import yfinance as yf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None
    def title(self, text):
        self._markdown(text, "h1")
    def header(self, text):
        self._markdown(text, "h2", " " * 2)
    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)
    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()
    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    def _markdown(self, text, level, space=""):
        import re
        key = re.sub('[^0-9a-zA-Z]+', '-', text).lower()
        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")
toc = Toc()


st.title('Stock Picker Project')

image = Image.open('stock.jpg')
st.image(image)

st.header('Table of Contents')
toc.placeholder()

toc.subheader('1. Navigating the stocks data dashboard')
for a in range(1):
    image = Image.open('layout.png')
    st.image(image, caption='Dashboard Layout')
    st.markdown('**This chart includes:**')
    st.markdown('1. the main KPIs for each stock')
    st.markdown('2. daily stock prices')
    st.markdown('3. a scrollable filter tab to see values denoting each stock.')
    st.markdown(' ')

toc.subheader('2. Understanding the main KPIs')
for a in range(1):
    st.markdown('**ASK PRICE**')
    st.markdown('- the ask price is the minimum price that a seller is willing to take for a share of their stock or other security (Fernado, Scott, & Clarine, 2021).')
    st.markdown('**ASK SIZE**')
    st.markdown('- the quantity of a security a market maker or investor is willing to sell at a specific price (Martin, 2022).')
    st.markdown('**BID PRICE**')
    st.markdown('- the bid price is the maximum price that a buyer is willing to pay for a share of stock or other security (Fernado, Scott, & Clarine, 2021).')
    st.markdown('**BID SIZE**')
    st.markdown('- the bid size is the quantity of a security a market maker or investor is willing to purchase at a specific price (Martin, 2022).')
    st.markdown('**LAST PRICE**')
    st.markdown('- the last price represents the main quoted price for the security in question. It is usually the price of the last trade or the previous close price if the market has yet to open (Milton, 2022).')
    st.markdown('**SHARES MATCHED**')
    st.markdown('- shares matched is defined as the Shares (or notional Shares, where relevant) acquired or to be acquired (as appropriate in the context) by a Participant pursuant to a Matching Award (Law Insider, 2023).')
    st.markdown('**SHARES ROUTED**')
    st.markdown('- order routing is the process by which a buy or sell order in the stock market is placed (Online Tradin Academy, 2020).')
    st.markdown('**VOLUME**')
    st.markdown('- volume is defined as the number of shares traded in a particular stock, index, or other investment over a specific time period (Fidelity, 2022).')
    st.markdown(' ')

toc.subheader('3. Stock names and abbreviations table')
for a in range(1):
    st.markdown('Please find bellow a list of the S&P 500 stocks with their respective symbol and name (Stock Market MBA, 2023).')
    data = pd.read_csv('stocks.csv',encoding= 'unicode_escape')
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.markdown(' ')

toc.subheader('4. Stock picker data dashboard')
for a in range(1):
    st.markdown('*[dashboard to be imported here]*')
    st.markdown(' ')

toc.subheader('5. Time series forecast graph')
for a in range(1):
    data = yf.download("AMZN", start="2022-01-01", end="2023-01-01")
    st3 = data['Adj Close']
    st3 = st3.dropna()
    st3_log=np.log(st3)
    moving_average=st3_log.rolling(12).mean()
    st3_station = st3_log - (moving_average*20)
    train_data, test_data = st3_log[:int(len(st3_log)*0.75)], st3_log[int(len(st3_station)*0.75):]
    model = ARIMA(train_data,order=(0,1,0))
    fitted = model.fit()
    fc = fitted.forecast(63,alpha=0.5)
    fc_series = pd.Series(fc.values,index=test_data.index)
    conf = fitted.forecast(63,alpha=0.5,inplace=True)
    lower_series = pd.Series(conf[:0],index=test_data.index)
    upper_series = pd.Series(conf[:1],index=test_data.index)
    def plot_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data, name="Training"))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data, name="Actual Stock Price"))
        fig.add_trace(go.Scatter(x=fc_series.index, y=fc_series, name="Predicted Stock Price"))
        fig.layout.update(title_text='Time series forecast for Amazon', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_data()
    st.markdown(' ')

toc.subheader('6. References')
for a in range(1):
    st.markdown('[1] Fernado, J., Scott, G., & Clarine, S. (2021, August 25). Bid and Ask Definition, How Prices Are Determined, and Example. Retrieved from Investopedia : https://www.investopedia.com/terms/b/bid-and-ask.asp ')
    st.markdown('[2] Fidelity. (2022, October 28). What volume says about stocks. Retrieved from Fidelity: https://www.fidelity.com/viewpoints/active-investor/stock-volume#:~:text=Volume%20is%20simply%20the%20number,million%20shares%20traded%20per%20day.* ')
    st.markdown('[3] Law Insider. (2023, January 24). Matched Shares definition. Retrieved from Law Insider: https://www.lawinsider.com/dictionary/matched-shares#:~:text=Matched%20Shares%20means%20Shares%20(or,Sample%201 ')
    st.markdown('[4] Martin, M. (2022, September 15). Bid Size vs. Ask Size in Options & Stocks Explained. Retrieved from Project Finance : https://www.projectfinance.com/bid-size-ask-size/ ')
    st.markdown('[5] Milton, A. (2022, March 30). Bid, Ask, and Last Prices Defined. Retrieved from The Balnace: https://www.thebalancemoney.com/trading-definitions-of-bid-ask-and-last-market-prices-1031026 ')
    st.markdown('[6] Online Tradin Academy. (2020, January 16). ORDER ROUTING. Retrieved from Online Tradin Academy: https://www.tradingacademy.com/financial-education-center/order-routing.aspx ')
    st.markdown('[7] Stock Market MBA. (2023, January 24). Stocks in the S&P 500 Index. Retrieved from Stock Market MBA: https://stockmarketmba.com/stocksinthesp500.php ')
    st.markdown(' ')

toc.subheader('Authors')
for a in range(1):
    st.write('Jemma Russell')
    st.write('Abdul-Hafiz Joarder')
    st.write('Beatrice Acquah')
    st.write('Essam Nakra')

toc.generate()

st.markdown('[Back to Top](#stock-picker-project)')