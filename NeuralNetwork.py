import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os
from math import*
from decimal import Decimal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential





def visualize(tickerName):
    startDate = dt.datetime(2015, 1, 1)
    endDate = dt.datetime.now()

    try:
        tickerData = web.DataReader(tickerName, 'yahoo', startDate, endDate)

        # Cleaning Ticker Data and Sizing Training Arrays
        mmScaler = MinMaxScaler(feature_range=(0, 1))
        scaled_tData = mmScaler.fit_transform(tickerData['Close'].values.reshape(-1, 1))

        predictionDaySize = 100

        xTrain = []
        yTrain = []

        for num in range(predictionDaySize, len(scaled_tData)):
            xTrain.append(scaled_tData[num - predictionDaySize:num, 0])
            yTrain.append(scaled_tData[num, 0])

        xTrain, yTrain = np.array(xTrain), np.array(yTrain)
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

        #Build the Modified LSTM Network
        NNModel = Sequential()
        NNModel.add(LSTM(units=50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
        NNModel.add(Dropout(0.2))
        NNModel.add(LSTM(units=40, return_sequences=True))
        NNModel.add(Dropout(0.2))
        NNModel.add(LSTM(units=30))
        NNModel.add(Dropout(0.3))
        NNModel.add(Dense(units=1))
        NNModel.compile(optimizer='adam', loss='mean_squared_error')
        NNModel.fit(xTrain, yTrain, epochs=50, batch_size=32)

        # Defining Testing Data
        testStartDate = dt.datetime(2020, 1, 1)
        testEndDate = dt.datetime.now()
        testData = web.DataReader(tickerName, 'yahoo', testStartDate, testEndDate)
        actualPrices = testData['Close'].values
        total_dataset = pd.concat((tickerData['Close'], testData['Close']), axis=0)
        NNModelInputs = total_dataset[len(total_dataset) - len(testData) - predictionDaySize:].values
        NNModelInputs = NNModelInputs.reshape(-1, 1)
        NNModelInputs = mmScaler.transform(NNModelInputs)

        # Make Predictions on Test Data
        xTest = []
        for num in range(predictionDaySize, len(NNModelInputs)):
            xTest.append(NNModelInputs[num - predictionDaySize:num, 0])
        xTest = np.array(xTest)
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
        predicted_prices = NNModel.predict(xTest)
        predicted_prices = mmScaler.inverse_transform(predicted_prices)

        # Plot the test predictions
        plt.clf()
        plt.title(f"{tickerName} Share Price")
        plt.xlabel("Time Elapsed (Days)")
        plt.ylabel(f"{tickerName} Share Price")
        plt.plot(actualPrices, color="black", label=f"Actual {tickerName} Price")
        plt.plot(predicted_prices, color="lawngreen", label=f"Predicted {tickerName} Price")
        plt.legend()

        myPath = os.path.abspath(__file__)
        plt.savefig(f'folder/{tickerName}.pdf', format="pdf", bbox_inches="tight")

        plt.close()
        f = open("summaryB.txt", 'a')
        f.write(f"{tickerName} {testEndDate}:\n")


        print("Similarity: ")
        sim = similarity(actualPrices, predicted_prices)
        print(sim)
        f.write(f"Similarity: {sim}\n")

        print("Accuracy: ")
        acc = accuracy(actualPrices, predicted_prices)
        f.write(f"Accuracy: {acc}\n")
        print(acc)
        # Predict Next Day
        actualTickerData = [NNModelInputs[len(NNModelInputs) + 1 - predictionDaySize:len(NNModelInputs + 1), 0]]
        actualTickerData = np.array(actualTickerData)
        actualTickerData = np.reshape(actualTickerData, (actualTickerData.shape[0], actualTickerData.shape[1], 1))

        tomorrowPrediction = NNModel.predict(actualTickerData)
        save = tomorrowPrediction
        tomorrowPrediction = mmScaler.inverse_transform(tomorrowPrediction)



        f.write(f"Most recent Close price for {tickerName}: {actualPrices[-1]}\n")
        f.write(f"Tomorrow's Predicted Close Price for {tickerName}: {tomorrowPrediction}\n")
        print(f"Most recent Close price for {tickerName}: {actualPrices[-1]}")
        print(f"Tomorrow's Predicted Close Price for {tickerName}: {tomorrowPrediction}")

        actualTickerDataFollowing = [NNModelInputs[len(NNModelInputs) + 2 - predictionDaySize:len(NNModelInputs + 2), 0]]
        actualTickerDataFollowing = np.array(actualTickerDataFollowing)
        actualTickerDataFollowing = np.append(actualTickerDataFollowing, save[0][0])
        actualTickerDataFollowing = np.reshape(actualTickerDataFollowing, (actualTickerData.shape[0], actualTickerData.shape[1], 1))

        followingPrediction = NNModel.predict(actualTickerDataFollowing)
        followingPrediction = mmScaler.inverse_transform(followingPrediction)
        f.write(f"The following day after Tomorrow is Predicted Close Price for {tickerName}: {followingPrediction}\n")
        print(f"The following day after Tomorrow is Predicted Close Price for {tickerName}: {followingPrediction}")

        if (actualPrices[-1] > tomorrowPrediction and actualPrices[-1] > followingPrediction and tomorrowPrediction > followingPrediction):
            f.write("Sell all positions ASAP\n")
            print("Sell all positions ASAP")


        elif (actualPrices[-1] < tomorrowPrediction and tomorrowPrediction < followingPrediction):
            f.write(f"Hold and buy what you can of {tickerName}, prepare to sell if conditions change\n")
            print(f"Hold and buy what you can of {tickerName}, prepare to sell if conditions change")

        elif (actualPrices[-1] < tomorrowPrediction and tomorrowPrediction > followingPrediction):
            f.write("Sell all positions Tomorrow\n")
            print("Sell all positions Tomorrow")

        elif ((actualPrices[-1] > tomorrowPrediction and actualPrices[-1] < followingPrediction) or (actualPrices[-1] > tomorrowPrediction and actualPrices[-1] > followingPrediction and followingPrediction > tomorrowPrediction)):
            f.write("Buy what you can tomorrow and sell it the following day\n")
            print("Buy what you can tomorrow and sell it the following day")

        elif (actualPrices[-1] > tomorrowPrediction and actualPrices[-1] > followingPrediction and tomorrowPrediction < followingPrediction):
            f.write("Buy what you can tomorrow and sell the following day\n")
            print("Buy what you can tomorrow and sell the following day")
        else:
            f.write("Hold all positions\n")
            print("Hold all positions")

        f.write("\n")
        f.close()

    finally:
        print()



def similarity(x, y):
    results = []
    sim = 0.0
    if(len(x) != len(y)):
        return "Error"
    for i in range(0, len(x)-1):
        if(x[i] > y[i]):
            results.append(y[i]/x[i])
        elif(x[i] < y[i]):
            results.append(x[i]/y[i])
        else:
            results.append(1)
    for i in results:
            sim = sim + i
    sim = sim/ len(results)
    return sim



def accuracy(x, y):
    results = []
    for i in range(0, len(x)-2):
        if(i != 0):
            if(x[i] > x[i-1] and y[i] > x[i-1]):
                results.append(1)
            elif(x[i] < x[i-1] and y[i] < x[i-1]):
                results.append(1)
            else:
                results.append(0)
    sum = 0
    for i in results:
        sum = sum + i
    acc = sum/len(results)
    return acc


def main():
    r = open("summaryB.txt", "w")
    r.close()
    f = open("symbols.txt", "r")
    for line in f:
        visualize(line.strip())


main()
