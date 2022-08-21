import numpy as np
import pandas as pd

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go


# twitter stock data from 18th Feb to March 20 2022
data = yf.download('TWTR', start='2022-02-18', end='2022-06-01')
print(data)

#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name = 'market data'))

# Add titles
fig.update_layout(
    title='Twitter Stock Price',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)
# Add a vertical line on 2022-04-10 to show the start of the increase in share price
fig.add_vline(
    x=pd.to_datetime('2022-04-14'),
    line_width=3,
    line_color='red',
    line_dash='dot',
    name='Increase in share price'
)


fig.add_vrect(x0="2022-04-02", x1="2022-04-20",
              annotation_text="Elon Musk Twitter Purchase", annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)

#show legend
fig.update_layout(legend_orientation="h")

#Show
fig.show()