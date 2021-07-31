# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:45:58 2021

@author: james
"""



import pandas
# The below code will read source of the diabetes dataset and seperate it by the tab delimiter
new_diabetes_dataframe = pandas.read_csv("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt",delimiter="\t")
# I will import the numpy module to create two datasets 
import numpy
data_frame_with_data_for_y = new_diabetes_dataframe.loc[:,new_diabetes_dataframe.columns == "Y"].to_numpy()
data_frame_with_data_for_x = new_diabetes_dataframe.loc[:,new_diabetes_dataframe.columns != "Y"].to_numpy()
data_for_y = data_frame_with_data_for_y
data_for_x = data_frame_with_data_for_x
# I will set the print options to 1 and display the shape of the x and y datasets
numpy.set_printoptions(precision = 1,suppress = True)
print("dataForX.shape: " + str(data_for_x.shape))
print("dataForY.shape" + str(data_for_y.shape))

# I will import api from the statsmodels module to get the p-values
from statsmodels import api
p_values = api.OLS(data_for_y,data_for_x).fit().summary2().tables[1]["P>|t|"]
# I will create a data frame for x which will be included into the new dataframe with p values
data_frame_with_data_for_x = new_diabetes_dataframe.loc[:, new_diabetes_dataframe.columns != "Y"]
data_frame_with_data_for_x_new = pandas.DataFrame(data = {"feature" : data_frame_with_data_for_x.columns, "p-value" : p_values})
data_frame_with_data_for_x_new = data_frame_with_data_for_x_new.reset_index()
del(data_frame_with_data_for_x_new["index"])
pandas.set_option('display.float_format', "{:.3f}".format)
print(data_frame_with_data_for_x_new)

# 4.) I will import the model selection from the sklearn module
from sklearn import model_selection
# This will split the data from 75% training and 25% testing
training_data_for_x, testing_data_for_x, training_data_for_y, testing_data_for_y = model_selection.train_test_split(data_for_x,data_for_y,test_size = .25)
# This will display the shape of each of the new datasets
print("trainingDataForX.shape: " + str(training_data_for_x.shape))
print("testingDataForX.shape: " + str(testing_data_for_x.shape))
print("trainingDataForY.shape: " + str(training_data_for_y.shape))
print("testingDataForY.shape: " + str(testing_data_for_y.shape))

# I will import linear regression from the sklearn module
from sklearn import linear_model
# I will build a linear regression model to be tested
linear_regression_model = linear_model.LinearRegression().fit(training_data_for_x,training_data_for_y)
# the below use testing data for x to predict y
predictions_for_y = linear_regression_model.predict(testing_data_for_x)
# This will print out the number and list of coefficients and R2 and intercept.  I will need to import the metrics module
from sklearn import metrics
# This will print out the intercept, R2 and coefficients
print("intercept: " + str(linear_regression_model.intercept_).replace("[","").replace("]",""))
print(str(linear_regression_model.coef_.size) + " coefficients: " + str(linear_regression_model.coef_).replace("[","").replace("]",""))
print("R-squared (coefficient of determination): " + str(round(metrics.r2_score(testing_data_for_y, predictions_for_y),2)))


# I will store the list of columns into a new variable
list_of_column_names = [data_frame_with_data_for_x.columns]
new_dataframe_with_data_and_predictions= pandas.DataFrame(columns = list_of_column_names, data = testing_data_for_x)
new_dataframe_with_data_and_predictions["actual Y"] = testing_data_for_y
new_dataframe_with_data_and_predictions["predicted Y"] = predictions_for_y
# This will display the last 3 rows of the dataframe
new_dataframe_with_data_and_predictions.tail(3)

# I will create a variable for a list of features to keep based on the p - value
list_of_features_to_keep = data_frame_with_data_for_x_new[(data_frame_with_data_for_x_new["p-value"] != data_frame_with_data_for_x_new["p-value"].max()) & (data_frame_with_data_for_x_new["p-value"].max() > .05)]["feature"] 
# I will use a for loop to loop through the list of features to keep
counter = 1
for features in list_of_features_to_keep:
    print("feature " + str(counter) + ": " + str(features))
    counter += 1
    
    
# I will updated the data for X with the list of features to keep
data_frame_with_data_for_x = data_frame_with_data_for_x[list_of_features_to_keep]
data_for_x = data_frame_with_data_for_x.to_numpy()
# will rebuild the training and testing split
training_data_for_x, testing_data_for_x, training_data_for_y, testing_data_for_y = model_selection.train_test_split(data_for_x,data_for_y,test_size = .25)
# Will rerun the regression one more time and display the output
linear_regression_model = linear_model.LinearRegression().fit(training_data_for_x,training_data_for_y)
predictions_for_y = linear_regression_model.predict(testing_data_for_x)
print("intercept: " + str(linear_regression_model.intercept_).replace("[","").replace("]",""))
print(str(linear_regression_model.coef_.size) + " coefficients: " + str(linear_regression_model.coef_).replace("[","").replace("]",""))
print("R-squared (coefficient of determination): " + str(round(metrics.r2_score(testing_data_for_y, predictions_for_y),2)))