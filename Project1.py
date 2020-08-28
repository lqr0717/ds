import calendar
from datetime import timedelta

import pandas as pd
import numpy as np
from dateutil import relativedelta
from pandas import Series, DatetimeIndex

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import numpy as np
from statsmodels.stats.outliers_influence import summary_table
from sklearn import metrics

from statsmodels.tsa.arima_model import ARMA, ARIMA

from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Functions #
def line_plot_find_missing_value(data,title,  targets, columns, index, values):
    null_table = pd.DataFrame(index=data[columns].unique())
    with PdfPages(targets + title) as export_pdf:
        for target in data[targets].unique():
            data_by_main = data[data[targets] == target]
            data_by_target_pivot = data_by_main.pivot(index=index, columns=columns,
                                    values=values)
            mm = pd.DataFrame(data_by_target_pivot.isnull().sum())
            mm.columns = [target]
            null_table = mm.merge(null_table,how = "outer", left_index=True, right_index=True)
            data_by_target_pivot.plot()
            plt.xlabel(index)
            plt.ylabel(values)
            plt.title(target)
            export_pdf.savefig()
            plt.close()

    return null_table


def detect_outlier(data,title,  criteria1, criteria2,threshold, values):
    outlier_note = ""
    for i in data[criteria1].unique():
        for j in data[criteria2].unique():
            dataset = data[(data[criteria2] == j) & (data[criteria1] == i) ]
            outliers = []
            z_outliers = []
            threshold = threshold
            mean = np.mean(dataset[values])
            std = np.std(dataset[values])

            for y in dataset[values]:
                z_score = (y - mean) / std
                if np.abs(z_score) > threshold:
                    outliers.append(y)
                    z_outliers.append(z_score)
            if (len(outliers) != 0):
                outlier_note = outlier_note + j + " in " + i + " is " + " ".join(
                    str(i) for i in outliers) + " when mean is " + str(round(mean, 2)) + " and z-score is " + " ".join(
                    str(round(i, 2)) for i in z_outliers) + "\n"

    outlier_output = open(title, "w")
    outlier_output.write(outlier_note)
    outlier_output.close()


def correlation_plot(data,title,  targets,  columns, index,values, save_csv, plot_print):
    with PdfPages(targets + title) as export_pdf:
        for target in data[targets].unique():
            data_by_main = data[data[targets] == target]
            data_by_target_pivot = data_by_main.pivot(index=index, columns=columns,
                                                      values=values)
            corr = data_by_target_pivot.corr()
            if save_csv == True:
                corr.to_csv(target[0:4] + r'_correlation.csv')

            if plot_print == True:
                # Draw the heatmap with the mask and correct aspect ratio
                cmap = sns.diverging_palette(220, 10, as_cmap=True)
                sns.set(font_scale=0.4)
                ax = plt.axes()
                sns.heatmap(corr, vmax=1, vmin=0.5, cmap = cmap,
                            square=True, linewidths=.3, cbar_kws={"shrink": .5}, annot=True)
                ax.set_title(target)
                export_pdf.savefig()
                plt.close()


def encoder(dataset, catFeatures, qtyFeatures):
    dataset = dataset[catFeatures + qtyFeatures]
    dataset_encoded = pd.get_dummies(dataset,
                                     columns=catFeatures,
                                     drop_first=True)

    return (dataset_encoded)


def add_variable(data, variable_name, rolling_window):
                if variable_name == "Month":
                    data[variable_name] = DatetimeIndex(data['Period Begin']).month
                if variable_name == "Year":
                    data[variable_name] = DatetimeIndex(data['Period Begin']).year
                if variable_name in ["Mean(PastYear)" ,"Median(SameMonth)" ,"Mean(Past3Months)"]:
                    if variable_name == "Mean(Past3Months)":
                        weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.4])
                        x = data[values].rolling(window=rolling_window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True).dropna()
                    if variable_name ==  "Median(SameMonth)":
                        weights= np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                        x = data[values].rolling(window=rolling_window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True).dropna()
                    if variable_name == "Mean(PastYear)":
                        x = data[values].rolling(window=rolling_window).mean().dropna()
                    x = x.drop(x.index[len(x) - 1])
                    data = data.iloc[rolling_window:]
                    x.index = data.index
                    data[variable_name] = x

                return data


def backward_elimination(x, Y, pvalue_thred, columns):
    var_count = len(x[0])
    for i in range(0, var_count):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > pvalue_thred:
            for j in range(0, var_count - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns


# load raw data
file = "training_data.csv"
file_test = "test.csv"
data = pd.read_csv(file)
data_test = pd.read_csv(file_test)
print(data.head())
print(data.shape)
print(data.dtypes)
print(data.isnull().sum())
data["Period Begin"] = pd.to_datetime(data["Period Begin"])
data["Period End"] = pd.to_datetime(data["Period End"])
data["Median Sale Price (in 000's of dollars)"] = data["Median Sale Price (in 000's of dollars)"].str.replace(",", "").astype(float)
print(data.dtypes)
print(data.head())
print(data.shape)

# first look of data & find missings/outliers
line_plot_find_missing_value(data,"_initial_run.pdf", "City","Property Type","Period Begin", "Median Sale Price (in 000's of dollars)")
missing_value = line_plot_find_missing_value(data, "_initial_run.pdf","Property Type","City","Period Begin", "Median Sale Price (in 000's of dollars)")
missing_value.to_csv(r'Missing_Value.csv')
detect_outlier(data,"Outliers.txt", "City", "Property Type",4, "Median Sale Price (in 000's of dollars)")

# Data Cleansing
## update outlier with mean
values = "Median Sale Price (in 000's of dollars)"
data[values][(data[values] > 1000) & (data["City"] == "Olympia") & (data["Property Type"] == "Condo/Co-op")] = np.mean(data[values][(data[values] < 1000) & (data["City"] == "Olympia") & (data["Property Type"] == "Condo/Co-op")])
## fillin missing value except for multi family property type,  "Mercer Island" - Townhouse missing 104 majority missing ,"Snoqualmie" - Condo/Co-op missing 61 missing more than half
data_n = pd.DataFrame()
unique_cal = pd.DataFrame(index = data["Period Begin"].unique())
unique_cal = unique_cal.sort_index(axis = 0)
null_dict = {"Townhouse":["Issaquah","Kenmore","Olympia","Snoqualmie"], "Condo/Co-op":[ "Kenmore","Olympia","Mercer Island"]}
for key, items in null_dict.items():
    for item in items:
        data_1 = data[(data["City"] == item) & (data["Property Type"] == key)]
        data_1 = data_1.set_index("Period Begin", drop=False)
        data_1 = data_1.sort_index(axis=0)
        index_drop = data[(data["City"] == item) & (data["Property Type"] == key)].index
        data.drop(index_drop, inplace=True)
        data_1 = data_1.merge(unique_cal,how = "outer",left_index=True, right_index=True)
        data_1[values]= data_1[values].interpolate()
        data_1[values] = data_1[values].fillna(method = "ffill")
        data_1[values] = data_1[values].fillna(method = "bfill")
        data_1["City"] = "".join(data_1["City"].dropna().unique())
        data_1["Property Type"] = "".join(data_1["Property Type"].dropna().unique())
        data_1['Period Begin'][data_1['Period Begin'].index[data_1['Period Begin'].apply(np.isnan)]] = data_1['Period Begin'].index[data_1['Period Begin'].apply(np.isnan)]
        data = data.append(data_1)

print(data.shape)
data = data.set_index("Period Begin", drop=False)
data = data.sort_index(axis=0)

# check data one more time & check correlation
line_plot_find_missing_value(data, "_fillin_null_run.pdf","City","Property Type","Period Begin", "Median Sale Price (in 000's of dollars)")
correlation_plot(data,"_correlation.pdf", "City", "Property Type","Period Begin","Median Sale Price (in 000's of dollars)", False, True)
correlation_plot(data,"_correlation.pdf", "Property Type","City","Period Begin", "Median Sale Price (in 000's of dollars)", False, True)

# reformat data adding rolling avg & set X & Y & feature selection & backward elimination
data_length_threshold = len(data.index.unique())*0.8
rolling_window = 12
x_corr_thred = 0.9
pvalue_thred = 0.05
variable_selected = ""
result_table = pd.DataFrame(columns = ['City', 'Property Type', "R2","MSE", "MAE"])

data_ts = pd.DataFrame()
data_pred_jan = pd.DataFrame()
criteria1 = "City"
criteria2 = "Property Type"

for ii in data[criteria1].unique():
    for jj in data[criteria2].unique():
        # ii = "Olympia"
        # jj = "Townhouse"
        data_1 = data[(data[criteria2] == jj) & (data[criteria1] == ii)]
        data_1 = data_1.sort_index(axis=0)
        if(len(data_1.index.unique())>data_length_threshold):
            # trandform data & create X
            data_1 = add_variable(data_1, "Month", 0)
            data_1 = add_variable(data_1, "Year", 0)
            data_1 = add_variable(data_1,  "Mean(PastYear)", rolling_window)
            data_1 = add_variable(data_1, "Mean(Past3Months)", rolling_window)
            data_1 = add_variable(data_1,  "Median(SameMonth)", rolling_window)

            X = data_1[["Mean(PastYear)","Median(SameMonth)","Mean(Past3Months)", "Month","Year"]]
            y = data_1[[values]]

            # X feature selection
            corr = X.corr()
            columns = np.full((corr.shape[0],), True, dtype=bool)
            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[0]):
                    if corr.iloc[i, j] >= x_corr_thred:
                        if columns[j]:
                            columns[j] = False
            selected_columns = X.columns[columns]
            X = X[selected_columns]
            selected_columns = selected_columns[:].values

            X_modeled, selected_columns = backward_elimination(X.iloc[:,:].values, y.iloc[:,0].values, pvalue_thred,selected_columns)
            X_selected = pd.DataFrame(X_modeled, columns=selected_columns)

            cols = ""
            for col in selected_columns:
                cols += col + " "
            variable_selected = variable_selected + jj + " in " + ii + " uses variable " + cols + "\n"

            # Run linear regression model on selected variable - R2, MAE, MSE
            linreg = LinearRegression(fit_intercept=True)
            re = linreg.fit(X_selected, y)
            y_train_pred = linreg.predict(X_selected)
            y_train_pred = np.array(y_train_pred[:,0])
            y_actual = y.iloc[:,0].values
            mse = np.sqrt(sum((y_train_pred - y_actual)**2) / len(y_train_pred))
            mae = sum(abs(y_train_pred - y_actual)) / len(y_train_pred)
            result_table = result_table.append({'City' :ii ,'Property Type':jj, "R2":metrics.r2_score(y, y_train_pred), "MSE": mse, "MAE": mae} , ignore_index=True)

            # predict test data
            data_t = data_1.tail(12)
            t_df1 = pd.DataFrame([[ii, jj, data_t['Period Begin'][-1]+ relativedelta.relativedelta(months=1), 0, 0, 0, 0, 0, 0, 0]], columns = data_t.columns)
            t_df2 = pd.DataFrame([[ii, jj, data_t['Period Begin'][-1]+ relativedelta.relativedelta(months=2),  0, 0, 0, 0, 0, 0, 0]], columns = data_t.columns)
            t_df3 = pd.DataFrame([[ii, jj, data_t['Period Begin'][-1]+ relativedelta.relativedelta(months=3),  0, 0, 0, 0, 0, 0, 0]], columns = data_t.columns)
            t_df4 = pd.DataFrame([[ii, jj, data_t['Period Begin'][-1]+ relativedelta.relativedelta(months=4), 0, 0, 0, 0, 0, 0, 0]], columns = data_t.columns)
            data_t = pd.concat([ data_t, t_df1, t_df2, t_df3, t_df4], ignore_index=True)

            for i in [0, 1, 2,3]:
                data_tt = data_t.iloc[rolling_window:]
                for var in selected_columns:
                    data_temp = add_variable(data_t, var, rolling_window)
                    data_tt[var] = data_temp[var]

                n = list(selected_columns)
                X_pred = pd.DataFrame(data_tt[n].iloc[i]).transpose()
                y_pred = linreg.predict(X_pred)
                data_t[values].iloc[i+12] = y_pred

            data_ts = data_ts.append(data_t[[criteria1, criteria2,"Period Begin" ,"Period End",values]].tail(4))
            data_ts["Period End"] = data_ts["Period Begin"] + timedelta(-1)
            data_ts["Period End"] = data_ts["Period End"] .apply(lambda date: date+relativedelta.relativedelta(months=1))

            for_test = pd.DataFrame(data_t[[criteria1, criteria2, values]].iloc[-1]).transpose()
            data_pred_jan = data_pred_jan.append(for_test)

# Output prediction for 4 months result
data_ts.to_csv(r'Prediction_Result_Final.csv')

# Compare prediction to actual Jan 2020
data_pred_jan = data_pred_jan.set_index([criteria1, criteria2])
data_pred_jan.columns = ["Predicted_Median"]
data_test = data_test.set_index([criteria1, criteria2])
test_result_validation  = data_pred_jan.merge(data_test,how = "outer", left_index=True, right_index=True)
test_result_validation[values] = test_result_validation[values].str.replace("K", "")
test_result_validation[values] = test_result_validation[values].str.replace("$", "")
test_result_validation[values] = test_result_validation[values].str.replace(",", "").astype(float)
test_result_validation = test_result_validation.reset_index()
test_result_validation.to_csv(r'Test_Prediction_Comparison.csv')

plt.plot(  'Predicted_Median', data=test_result_validation, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot(  values, data=test_result_validation, marker='', color='olive', linewidth=2)
plt.legend()

# print out Training result
r2_table = result_table.pivot(index='Property Type', columns='City', values="R2")
mse_table = result_table.pivot(index='Property Type', columns='City',  values="MSE")
mae_table= result_table.pivot(index='Property Type', columns='City', values="MAE")
r2_table.to_csv(r'R2_result.csv')
mse_table.to_csv(r'MSE_Training_result.csv')
mae_table.to_csv(r'MAE_Training_result.csv')
variable_note = open("Variables_Selected_Output.txt", "w")
variable_note.write(variable_selected)
variable_note.close()
