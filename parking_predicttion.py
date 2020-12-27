import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import requests
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os

def get_request_query(url, solYear, solMonth, serviceKey):
    request_query = url + "?solYear=" + solYear + "&solMonth=" + solMonth + "&serviceKey=" + serviceKey
    return request_query

def getHoliDay(inputYear):
    URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    SERVICEKEY = "17sl4WcpRqvwbiEVl9QyQdMW9P9Y%2FC7O9B6Om%2FgSJhbSODZ5YyOxZS3l%2BjNbM6QjY6AhyhTclxQPY2BX8wSUCg%3D%3D"
    solYear  = str(inputYear)
    result = []

    for solMonth in range(1, 13):
        solMonth = "%02d" % solMonth
        request_query = get_request_query(URL, solYear, solMonth, SERVICEKEY)
        response = requests.get(url=request_query)
        cntIndex = response.text.find("<totalCount>") + 12
        cnt = int(response.text[cntIndex:cntIndex+1])
        offset = 0
        tmpList = response.text
        if True == response.ok:
            if cnt > 0:
                for i in range(cnt):
                    index = tmpList.find("<locdate>") + 9
                    result.append(tmpList[index:index+8])
                    tmpList = tmpList[index:]
    return result

def isHoliDay(inputDate, holiDayList, HoliDay):
    if HoliDay == 0:
        for i in range(len(holiDayList)):
            if inputDate == holiDayList[i]:
                return 1
        return 0
    else:
        return 1


# train data
df = pd.read_csv("./parking_data.csv", encoding="utf-8")
df = df.dropna()
df = df.astype({"no": "int", "maxParking": "int", "avilableParking": "int", "temperature": "int"})
date1 = df["dateAndTime"].str.split(" ")
df = df.rename(columns={"dateAndTime": "date"})
df["date"] = date1.str.get(0)
df["time"] = date1.str.get(1)
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors='raise')
df["weekDay"] = df["date"].dt.day_name()
df["holiDay"] = df.apply(lambda x : 1 if x["weekDay"] == "Sunday" or x["weekDay"] == "Saturday" else 0, axis = 1)
df["fullParking"] = df.apply(lambda x : 1 if x["avilableParking"] <= 0 else 0, axis = 1)

result = pd.to_datetime(getHoliDay(2020), format="%Y-%m-%d", errors='raise')

df["holiDay"] = df.apply(lambda x : isHoliDay(x["date"], result, x["holiDay"]), axis = 1)


# test data
cmp_df = pd.read_csv("./cmp_data.csv", encoding="utf-8")
cmp_df = cmp_df.dropna()
cmp_df = cmp_df.astype({"no": "int", "maxParking": "int", "avilableParking": "int", "temperature": "int"})
cmp_date1 = cmp_df["dateAndTime"].str.split(" ")
cmp_df = cmp_df.rename(columns={"dateAndTime": "date"})
cmp_df["date"] = cmp_date1.str.get(0)
cmp_df["time"] = cmp_date1.str.get(1)
cmp_df["date"] = pd.to_datetime(cmp_df["date"], format="%Y-%m-%d", errors='raise')
cmp_df["weekDay"] = cmp_df["date"].dt.day_name()
cmp_df["holiDay"] = cmp_df.apply(lambda x : 1 if x["weekDay"] == "Sunday" or x["weekDay"] == "Saturday" else 0, axis = 1)
cmp_df["fullParking"] = cmp_df.apply(lambda x : 1 if x["avilableParking"] <= 0 else 0, axis = 1)
cmp_df["holiDay"] = cmp_df.apply(lambda x : isHoliDay(x["date"], result, x["holiDay"]), axis = 1)


# make dummy data
y = df["fullParking"]
features = ["no", "temperature", "time", "weekDay", "holiDay"]
X = pd.get_dummies(df[features])
X_test = pd.get_dummies(cmp_df[features])


# RandomForestRegressor
RandomForestRegressor_model = RandomForestRegressor(random_state=1)
RandomForestRegressor_model.fit(X, y)
RandomForestRegressor_predict = RandomForestRegressor_model.predict(X_test)
RandomForestRegressor_mae = mean_absolute_error(RandomForestRegressor_predict, cmp_df["fullParking"])

# RandomForestClassifier
RandomForestClassifier_model = RandomForestClassifier(random_state=0)
RandomForestClassifier_model.fit(X, y)
RandomForestClassifier_predict = RandomForestClassifier_model.predict(X_test)
RandomForestClassifier_output = pd.DataFrame({"fullParking": RandomForestClassifier_predict})
RandomForestClassifier_mae = mean_absolute_error(RandomForestClassifier_predict, cmp_df["fullParking"])

# DecisionTreeRegressor
DecisionTreeRegressor_model = DecisionTreeRegressor(random_state=0)
DecisionTreeRegressor_model.fit(X, y)
DecisionTreeRegressor_predict = DecisionTreeRegressor_model.predict(X_test)
DecisionTreeRegressor_output = pd.DataFrame({"fullParking": DecisionTreeRegressor_predict})
DecisionTreeRegressor_mae = mean_absolute_error(DecisionTreeRegressor_predict, cmp_df["fullParking"])


# LinearRegression
LinearRegression_model = LinearRegression()
LinearRegression_model.fit(X, y)
LinearRegression_predict = LinearRegression_model.predict(X_test)
LinearRegression_output = pd.DataFrame({"fullParking": LinearRegression_predict})
LinearRegression_mae = mean_absolute_error(LinearRegression_predict, cmp_df["fullParking"])


# KMeans
KMeans_model = KMeans(2)
KMeans_model.fit(X, y)
KMeans_predict = KMeans_model.predict(X_test)
KMeans_output = pd.DataFrame({"fullParking": KMeans_predict})
KMeans_mae = mean_absolute_error(KMeans_predict, cmp_df["fullParking"])


# input data
input_parkingPlaceNo = int(input("input parking place no: "))
input_date = input("input date YYYY-MM-YY: ")
input_time = input("input time HH: ")
input_temperature = int(input("input temperature: "))
input_date = pd.to_datetime(input_date, format="%Y-%m-%d", errors="raise")
input_data = pd.DataFrame({"no": input_parkingPlaceNo,  "date": input_date, "temperature": input_temperature, "time": input_time}, index = [0])
input_data["weekDay"] = input_data["date"].dt.day_name()
input_data["holiDay"] = input_data["weekDay"].apply(lambda x : 1 if x == "Sunday" or x == "Saturday" else 0)
input_tmp = input_data["date"]
input_data["holiDay"] = input_data["holiDay"].apply(lambda x : isHoliDay(input_tmp, result, x))


# prediction
my_model = RandomForestClassifier_model
input_X = pd.concat([input_data, df])
input_X = pd.get_dummies(input_X[features])
input_X = input_X.loc[0]
my_predict = my_model.predict(input_X)
print("%d%%" % ((1-my_predict[0]) * 100))
