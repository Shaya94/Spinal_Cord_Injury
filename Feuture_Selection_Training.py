import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
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


if __name__ == "__main__":
    feature = pd.read_csv("feature_label_national_subnational.csv")
    feature_drop = feature.drop(columns=list(feature)[:3])
    feature_drop = feature_drop.drop(columns=list(feature)[-1])
    feature_drop = feature_drop.dropna(thresh=int(feature_drop.shape[0] * 4 / 10), axis=1)
    # corr_matrix.to_csv("corr_pearson.csv")

    # sns.set(rc={'figure.figsize': (20, 20)})
    # lm = sns.heatmap(corr_matrix, annot=True)
    # plt.show()
    list_cols = list(feature_drop)
    imp = IterativeImputer(max_iter=1000, random_state=0, initial_strategy="median")
    final_df = feature_drop[list_cols]
    final_df = imp.fit_transform(final_df)
    final_df = pd.DataFrame(final_df)
    final_df.columns = list_cols

    corr_matrix = feature_drop.corr(method='pearson')
    corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))

    list_cols = get_variance_gt_thresh(corr_matrix, 0.7)
    final_df = final_df[list_cols]
    final_df["crude_incidence_rate"] = feature["crude_incidence_rate"]

    X = final_df[list_cols].values
    y = final_df['crude_incidence_rate'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(list_cols)
    X = SelectKBest(mutual_info_regression, k=20).fit_transform(X, y)

    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    MAE_test, MAE_train, R_squared_test, R_squared_train, RMSE_test, RMSE_train = [], [], [], [], [], []
    coef_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # regressor = NuSVR(kernel="linear", nu=0.3)
        # regressor = SVR(kernel="linear", epsilon=1, C=1)
        # regressor = LinearRegression()
        regressor = MLPRegressor(hidden_layer_sizes=(10, 20, 10), max_iter=10000, learning_rate_init=0.01,
                                 learning_rate="invscaling", alpha=119, random_state=2)
        regressor.fit(X_train, y_train.ravel())
        y_pred = regressor.predict(X_test)
        y_train_pred = regressor.predict(X_train)
        # coef_list.append(regressor.coef_.reshape(-1))

        MAE_test.append(metrics.mean_absolute_error(y_test, y_pred))
        MAE_train.append(metrics.mean_absolute_error(y_train, y_train_pred))
        R_squared_test.append(metrics.r2_score(y_test, y_pred))
        R_squared_train.append(metrics.r2_score(y_train, y_train_pred))
        RMSE_test.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        RMSE_train.append(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

    print("Mean Absolute Error (test): ", np.mean(MAE_test))
    print("Mean Absolute Error (train): ", np.mean(MAE_train))
    # print("Root Mean Squared Error (test): ", np.mean(RMSE_test))
    # print("Root Mean Squared Error (train): ", np.mean(RMSE_train))
    print("Root Mean Squared Error (test): ", np.sqrt(np.mean(np.power(RMSE_test, 2))))
    print("Root Mean Squared Error (train): ", np.sqrt(np.mean(np.power(RMSE_train, 2))))
    print("R squared score (test): ", np.mean(R_squared_test))
    print("R squared score (train): ", np.mean(R_squared_train))
    print(np.std(MAE_test))
    #
    # coef_list = np.array(coef_list)
    # coef_list = np.mean(coef_list, axis=0)
    # get_most_important_features(coef_list, list_cols)
