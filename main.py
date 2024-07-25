import math
import pandas_datareader as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from xgboost import XGBClassifier
from sklearn import metrics


import warnings

stocks = []
f = open("symbols.txt", "r")
for line in f:
    stocks.append(line.strip())

f.close()

#web.DataReader(stocks, "yahoo", start = "2000-1-1", end = "2022-11-16")["Adj Close"].to_csv("adjclose.csv")
#web.DataReader(stocks, "yahoo", start = "2000-1-1", end = "2022-11-16")["Volume"].to_csv("volume.csv")

warnings.filterwarnings("ignore")

prices = pd.read_csv("adjclose.csv", index_col="Date", parse_dates=True)
d = dt.date(2000,1,3)
#print(prices['MSFT'][0])
volumechanges = pd.read_csv("volume.csv", index_col="Date", parse_dates=True).pct_change()*100
#print(prices.index())

today = dt.date(2000, 1, 1)
start = dt.date(2000, 1, 1)
simend = dt.date(2022, 11, 16)
i = 0
tickers = []
transactionid = 0
money = 100
portfolio = {}
activelog = []
transactionlog = []



def getprice(date, tickerName):
    global prices
    day = date.strftime("%Y-%m-%d")
    return prices.loc[day][tickerName]


def transaction(id, tickerName, amount, price, type, info):
    global transactionid
    if type == "buy":
        exp_date = today + dt.timedelta(days=14)
        transactionid += 1
    else:
        exp_date = today
    if type == "sell":
        data = {"id": id, "ticker": tickerName, "amount": amount, "price": price, "date": today, "type": type,
                "exp_date": exp_date, "info": info}
    elif type == "buy":
        data = {"id": transactionid, "ticker": tickerName, "amount": amount, "price": price, "date": today, "type": type,
                "exp_date": exp_date, "info": info}
        activelog.append(data)
    transactionlog.append(data)


def buy(interestList, reservedCash):
    global money, portfolio
    for item in interestList:
        price = getprice(today, item)
        if not np.isnan(price):
            quantity = math.floor(reservedCash/price)
            money -= quantity*price
            portfolio[item] += quantity
            transaction(0, item, quantity, price, "buy", "")


def sell():
    global money, portfolio, prices, today
    itemstoremove = []
    for i in range(len(activelog)):
        log = activelog[i]
        if log["exp_date"] <= today and log["type"] == "buy":
            tickprice = getprice(today, log["ticker"])
            if not np.isnan(tickprice):
                money += log["amount"]*tickprice
                portfolio[log["ticker"]] -= log["amount"]
                transaction(log["id"], log["ticker"], log["amount"], tickprice, "sell", log["info"])
                itemstoremove.append(i)
            else:
                log["exp_date"] += dt.timedelta(days=1)
    itemstoremove.reverse()
    for elem in itemstoremove:
        activelog.remove(activelog[elem])


def simulation():
    global today, volumechanges, money
    start_date = today - dt.timedelta(days=14)
    series = volumechanges.loc[start_date:today].mean()
    interestlst = series[series > 100].index.tolist()
    sell()
    if len(interestlst) > 0:
        #moneyToAllocate = 500000/len(interestlst)
        moneyToAllocate = currentvalue()/(2*len(interestlst))
        buy(interestlst, moneyToAllocate)


def getindices():
    global tickers
    f = open("symbols.txt", "r")
    for line in f:
        tickers.append(line.strip())
    f.close()


def tradingday():
    global prices, today
    return np.datetime64(today) in list(prices.index.values)


def currentvalue():
    global cash, portfolio, today, prices
    value = cash
    for ticker in tickers:
        tickprice = getprice(today, ticker)
        if not np.isnan(tickprice):
            value += portfolio[ticker]*tickprice
    return int(value*100)/100


def main():
    analyze("MSFT")
    global today
    getindices()
    for ticker in tickers:
        portfolio[ticker] = 0
    while today < simend:
        while not tradingday():
            tomorrow = today + dt.timedelta(days=1)
            today = tomorrow
        simulation()
        currentpvalue = currentvalue()
        print(currentpvalue, today)
        today += dt.timedelta(days=7)



def analyze(ticker):
    global prices


    plt.figure(figsize=(15, 5))
    plt.plot(prices[ticker])
    plt.title('Facebook Adjusted Close price.', fontsize=15)
    plt.ylabel('Price in dollars.')
    plt.show()
    print(prices.head())

    fb_dwn = yf.download(ticker,start='2000-01-01',
                         end='2022-11-16',
                         progress=False,
                         )


    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(start_date='2000-01-01', end_date='2022-03-20', time_interval='daily')
    fb_df = pd.DataFrame(data[ticker]['prices'])



    fb_df = fb_df.drop('date', axis=1)#.set_index('formatted_date')

    print(ticker)
    print(fb_df.head())
    print(" ")


    features = ['open', 'high', 'low', 'adjclose', 'volume']

    plt.subplots(figsize=(20, 10))

    for x, col in enumerate(features):
        plt.subplot(2, 3, x + 1)
        sb.distplot(fb_df[col])
    plt.show()

    plt.subplots(figsize=(20, 10))
    for x, col in enumerate(features):
        plt.subplot(2, 3, x + 1)
        sb.boxplot(fb_df[col])
    plt.show()

    splitted = fb_df['formatted_date'].str.split('-', expand=True)

    fb_df['day'] = splitted[1].astype('int')
    print(type(fb_df['day']))
    fb_df['month'] = splitted[0].astype('int')
    fb_df['year'] = splitted[2].astype('int')

    fb_df['is_quarter_end'] = np.where(fb_df['month'] % 3 == 0, 1, 0)


    fb_df['open-close'] = fb_df['open'] - fb_df['adjclose']
    fb_df['low-high'] = fb_df['low'] - fb_df['high']
    fb_df['target'] = np.where(fb_df['adjclose'].shift(-1) > fb_df['adjclose'], 1, 0)

    features = fb_df[['open-close', 'low-high', 'is_quarter_end']]
    target = fb_df['target']

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features, target, test_size=0.3, random_state=202)
    print(X_train.shape, X_valid.shape)
    print(type(X_train[0][0]))


    models = [LogisticRegression(), SVC(
        kernel='poly', probability=True), XGBClassifier(), SVR(kernel='poly')]

    for i in range(3):
        models[i].fit(X_train, Y_train)

        print(f'{models[i]} : ')
        print('Training Accuracy : ', metrics.roc_auc_score(
            Y_train, models[i].predict_proba(X_train)[:, 1]))
        print('Validation Accuracy : ', metrics.roc_auc_score(
            Y_valid, models[i].predict_proba(X_valid)[:, 1]))
        print()
    
def mainA():
    X = prices[:]['MSFT']
    print(X)

def visualize():
    tickerName = 'FB'

    startDate = dt.datetime(2012, 1, 1)
    endDate = dt.datetime(2022, 11, 29)

    data = web.DataReader(tickerName, 'yahoo', startDate, endDate)

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    '''Test the model accuracy on existing data'''

    # Load Test Data
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(tickerName, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Test Data
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the test predictions
    plt.title(f"{tickerName} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{tickerName} Share Price")
    plt.plot(actual_prices, color="black", label=f"Actual {tickerName} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {tickerName} Price")
    plt.legend()
    plt.show()

    # Predict Next Day

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

visualize()
