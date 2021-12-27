import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


def get_variance_gt_thresh(corr_mat, thresh=0.9):
    list_cols = []
    not_drop = []
    for row in corr_mat.index:
        for col in corr_mat.loc[row].index:
            if abs(corr_mat[col].loc[row]) >= thresh:
                if row != col:
                    if col not in list_cols and col not in not_drop:
                        list_cols.append(col)
                        not_drop.append(row)

    print(list_cols)
    print(len(np.unique(list_cols)))
    return list_cols


def get_most_important_features(weight_list, feature_list, n=5):
    for i in range(n):
        max_arg = np.argmax(np.abs(weight_list))
        print(weight_list[max_arg])
        print(feature_list[max_arg])
        print("============================")
        weight_list[max_arg] = 0


def combine_with_others(country_name, year_name, all_df, cleaned_df):
    all_df = all_df.loc[(all_df["Country"] == country_name) & (all_df["Year"] == year_name)]
    # all_df = all_df.iloc[13:14]
    # all_df = all_df.drop(columns=list(all_df)[:3])
    all_df = all_df[list(cleaned_df)]
    cleaned_df = cleaned_df.append(all_df, ignore_index=True)
    return cleaned_df


if __name__ == "__main__":
    feature = pd.read_csv("feature_label_national_subnational.csv")
    # features of all countries
    all_country_features = pd.read_csv("data_cleaned.csv")

    name_year = feature[["Country", "Year"]]
    name_year.set_index(list(name_year)[0], inplace=True)

    all_country_name = all_country_features[["Country", "Year"]]
    # all_country_name.set_index(list(all_country_name)[0], inplace=True)
    # print(all_country_name.loc[(all_country_name["Country"] == "Germany") & (all_country_name["Year"] == 2000)])

    feature_drop = feature.drop(columns=list(feature)[:3])
    feature_drop = feature_drop.drop(columns=list(feature)[-1])
    feature_drop = feature_drop.dropna(thresh=int(feature_drop.shape[0] * 4 / 10), axis=1)

    list_cols = list(feature_drop)
    imp = IterativeImputer(max_iter=1000, random_state=0, initial_strategy="median")
    final_df = feature_drop[list_cols]
    final_df = imp.fit_transform(final_df)
    final_df = pd.DataFrame(final_df)
    final_df.columns = list_cols

    corr_matrix = feature_drop.corr(method='pearson')
    corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))

    list_cols_final = get_variance_gt_thresh(corr_matrix, 0.7)
    train_df = final_df[list_cols_final]
    train_df[train_df < 0] = 0

    train_df["crude_incidence_rate"] = feature["crude_incidence_rate"]

    X = train_df[list_cols_final].values
    y = train_df['crude_incidence_rate'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    regressor = MLPRegressor(hidden_layer_sizes=(10, 20, 10), max_iter=10000, learning_rate_init=0.001,
                             learning_rate="invscaling", alpha=100, random_state=2)
    regressor.fit(X, y.ravel())
    y_pred = regressor.predict(X)

    # y_pred_test = regressor.predict(test_df)

    print(metrics.mean_absolute_error(y, y_pred))
    print(np.sqrt(metrics.mean_squared_error(y, y_pred)))
    name_year["Ground truth"] = np.round(y, 2)
    name_year["Predicted"] = np.round(y_pred, 2)

    all_year = pd.DataFrame(columns=["Country", "Year", "Predicted Incidence Rate"])
    mean_year = pd.DataFrame(columns=["Country", "Predicted Incidence Rate"])
    country_list = np.unique(list(all_country_name["Country"]))
    for country in country_list:
        country_mean = []
        for year in all_country_name.loc[(all_country_name["Country"] == country)]["Year"]:
            test_df = combine_with_others(country, year, all_country_features, final_df)
            imp = IterativeImputer(max_iter=1000, random_state=0, initial_strategy="median")
            test_df = imp.fit_transform(test_df)
            test_df = pd.DataFrame(test_df)
            test_df.columns = list_cols
            test_df[test_df < 0] = 0
            test_df = scaler.fit_transform(test_df)
            test_df = pd.DataFrame(test_df)
            # test_df[test_df < 0] = 0
            test_df.columns = list_cols
            test_df = test_df.iloc[-1:]

            test_df = test_df[list_cols_final]
            y_test = regressor.predict(test_df)
            all_year = all_year.append({"Country": country, "Year": year, "Predicted Incidence Rate": y_test[0]},
                                       ignore_index=True)
            country_mean.append(y_test[0])

        if country in name_year.index:
            print(country)
            country_mean = name_year.loc[country]["Predicted"].mean()
            mean_year = mean_year.append({"Country": country, "Predicted Incidence Rate": country_mean},
                                         ignore_index=True)
        else:
            mean_year = mean_year.append({"Country": country, "Predicted Incidence Rate": np.median(country_mean)},
                                         ignore_index=True)
        print(mean_year)
    mean_year.to_csv("mean_incidence_all_countries_2.csv")
    all_year.to_csv("all_countries_incidence_2.csv")

    # name_year.to_csv("predicted.csv")
    # print(y_pred_test)
    # print(np.mean(y_pred_test))
    #
    # one_country_name = all_country_name.loc[name_country].iloc[0]
    # one_country_name["Predicted"] = np.round(y_pred_test, 2)
    # one_country_name.to_csv("predicted_all_countries.csv")
