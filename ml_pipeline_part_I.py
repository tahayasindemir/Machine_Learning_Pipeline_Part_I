# ML PIPELINE PART - I
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from helpers.data_prep import *
from helpers.eda import *
import warnings
warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
df = pd.read_csv(r"...\hitters.csv")
df.head()

#############################################################
# Exploratory Data Analysis
#############################################################


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


cat_cols, num_cols, cat_but_car = grab_col_names(df)

check_df(df)
for col in cat_cols:
    cat_summary(df, col, True)

for col in num_cols:
    num_summary(df, col, True)
# Even in hist plots, we can see clearly that we have outlier values on target variable

df.columns = [col.upper() for col in df.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.describe().T

for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()


for col in num_cols:
    plot_numerical_col(df, col)

############################################################
# Feature Engineering & Data Preprocessing
############################################################

# Against the N/A values that can be taken in division and multiplication operations,
# we draw those with a minimum value of 0 to 1
for col in num_cols:
    if col != 'SALARY':
        if df[col].min() == 0:
            df[col] = df[col] + 1

# We derive new variables by proportioning the last season played according to all career data.
df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]
df["NEW_C_HIT_RATE"] = df["CHITS"] / df["CATBAT"]
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHITS"]
df["NEW_C_HIT-AND-RUN"] = df["CRUNS"] / df["CHITS"]
df["NEW_C_HMHITS_RATIO"] = df["CHMRUN"] / df["CHITS"]
df["NEW_C_HMATBAT_RATIO"] = df["CATBAT"] / df["CHMRUN"]

# We can also average all career data by years.
df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]

# We can levelise players by experience.
df['NEW_EXP_LEVEL'] = pd.qcut(df['YEARS'], 6, labels=[1, 2, 3, 4, 5, 6])

# Other variables that we can derive according to the performance metrics in the season:
df["NEW_ASSISTS_RATIO"] = df["ASSISTS"] / df["ATBAT"]
df["NEW_HITS_RECALL"] = df["HITS"] / (df["HITS"] + df["ERRORS"])
df["NEW_NET_HELPFUL_ERROR"] = (df["WALKS"] - df["ERRORS"]) / df["WALKS"]
df["NEW_TOTAL_SCORE"] = (df["RBI"] + df["ASSISTS"] + df["WALKS"] - df["ERRORS"]) / df["ATBAT"]
df["NEW_HIT_RATE"] = df["HITS"] / df["ATBAT"]
df["NEW_TOUCHER"] = df["ASSISTS"] / df["PUTOUTS"]
df["NEW_RUNNER"] = df["RBI"] / df["HITS"]
df["NEW_HIT_RUN"] = df["RUNS"] / (df["HITS"])
df["NEW_HMHITS_RATIO"] = df["HMRUN"] / df["HITS"]
df["NEW_HMATBAT_RATIO"] = df["ATBAT"] / df["HMRUN"]
df["NEW_TOTAL_CHANCES"] = df["ERRORS"] + df["PUTOUTS"] + df["ASSISTS"]

df.isnull().sum()
# na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
# only N/A having column remained is SALARY

# We will have a look at the independent variable correlations
corr_df = pd.DataFrame(df.corr())
corr_df = corr_df[corr_df > 0.98]
corr_df
# So we can drop the highly correlated columns
# CATBAT is the most important feature as I saw in previous trials for the Random Forest Regressor
df.drop('CHITS', axis=1, inplace=True)  # 0.99 corr, CHITS with CATBAT
df.drop('CRUNS', axis=1, inplace=True)  # 0.98 corr, CRUNS with CATBAT
df.drop('NEW_ASSISTS_RATIO', axis=1, inplace=True)  # 0.99 corr with NEW_TOTAL_SCORE

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Outliers:
# We will not touch the outliers, cause we will use decision tree methods.
for col in num_cols:
    if check_outlier(df, col):
        print(col + ":", check_outlier(df, col))

# One of the best ways of checking for outliers is Boxplot
df["SALARY"].describe([0.05, 0.25, 0.45, 0.50, 0.65, 0.75, 0.95, 0.99]).T
sns.boxplot(x=df["SALARY"])
plt.show()
# There seems some outliers on the right side of plot (upper quantile side) and we reduced the upper limit.
q3 = 0.60
salary_up = int(df["SALARY"].quantile(q3))
df = df[(df["SALARY"] < salary_up)]

# Rare Analysis
rare_analyser(df, "SALARY", cat_cols)
# no need for rare encoding

# KNN Imputer
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

df.isnull().values.any()
# We made the imputation and no null values remained

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# RobustScaler
for col in num_cols:
    if col != 'SALARY':
        df[col] = RobustScaler().fit_transform(df[[col]])

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)
# So let's define all these operations inside a function:


def diabetes_data_prep(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    for col in num_cols:
        if col != 'SALARY':
            if dataframe[col].min() == 0:
                dataframe[col] = dataframe[col] + 1

    dataframe["NEW_C_RUNS_RATIO"] = dataframe["RUNS"] / dataframe["CRUNS"]
    dataframe["NEW_C_ATBAT_RATIO"] = dataframe["ATBAT"] / dataframe["CATBAT"]
    dataframe["NEW_C_HITS_RATIO"] = dataframe["HITS"] / dataframe["CHITS"]
    dataframe["NEW_C_HMRUN_RATIO"] = dataframe["HMRUN"] / dataframe["CHMRUN"]
    dataframe["NEW_C_RBI_RATIO"] = dataframe["RBI"] / dataframe["CRBI"]
    dataframe["NEW_C_WALKS_RATIO"] = dataframe["WALKS"] / dataframe["CWALKS"]
    dataframe["NEW_C_HIT_RATE"] = dataframe["CHITS"] / dataframe["CATBAT"]
    dataframe["NEW_C_RUNNER"] = dataframe["CRBI"] / dataframe["CHITS"]
    dataframe["NEW_C_HIT-AND-RUN"] = dataframe["CRUNS"] / dataframe["CHITS"]
    dataframe["NEW_C_HMHITS_RATIO"] = dataframe["CHMRUN"] / dataframe["CHITS"]
    dataframe["NEW_C_HMATBAT_RATIO"] = dataframe["CATBAT"] / dataframe["CHMRUN"]
    dataframe["NEW_CATBAT_MEAN"] = dataframe["CATBAT"] / dataframe["YEARS"]
    dataframe["NEW_CHITS_MEAN"] = dataframe["CHITS"] / dataframe["YEARS"]
    dataframe["NEW_CHMRUN_MEAN"] = dataframe["CHMRUN"] / dataframe["YEARS"]
    dataframe["NEW_CRUNS_MEAN"] = dataframe["CRUNS"] / dataframe["YEARS"]
    dataframe["NEW_CRBI_MEAN"] = dataframe["CRBI"] / dataframe["YEARS"]
    dataframe["NEW_CWALKS_MEAN"] = dataframe["CWALKS"] / dataframe["YEARS"]
    dataframe['NEW_EXP_LEVEL'] = pd.qcut(dataframe['YEARS'], 6, labels=[1, 2, 3, 4, 5, 6])
    dataframe["NEW_ASSISTS_RATIO"] = dataframe["ASSISTS"] / dataframe["ATBAT"]
    dataframe["NEW_HITS_RECALL"] = dataframe["HITS"] / (dataframe["HITS"] + dataframe["ERRORS"])
    dataframe["NEW_NET_HELPFUL_ERROR"] = (dataframe["WALKS"] - dataframe["ERRORS"]) / dataframe["WALKS"]
    dataframe["NEW_TOTAL_SCORE"] = (dataframe["RBI"] + dataframe["ASSISTS"] + dataframe["WALKS"] - dataframe["ERRORS"]) / dataframe["ATBAT"]
    dataframe["NEW_HIT_RATE"] = dataframe["HITS"] / dataframe["ATBAT"]
    dataframe["NEW_TOUCHER"] = dataframe["ASSISTS"] / dataframe["PUTOUTS"]
    dataframe["NEW_RUNNER"] = dataframe["RBI"] / dataframe["HITS"]
    dataframe["NEW_HIT_RUN"] = dataframe["RUNS"] / (dataframe["HITS"])
    dataframe["NEW_HMHITS_RATIO"] = dataframe["HMRUN"] / dataframe["HITS"]
    dataframe["NEW_HMATBAT_RATIO"] = dataframe["ATBAT"] / dataframe["HMRUN"]
    dataframe["NEW_TOTAL_CHANCES"] = dataframe["ERRORS"] + dataframe["PUTOUTS"] + dataframe["ASSISTS"]

    dataframe.drop('CHITS', axis=1, inplace=True)
    dataframe.drop('CRUNS', axis=1, inplace=True)
    dataframe.drop('NEW_ASSISTS_RATIO', axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    q3 = 0.60
    salary_up = int(dataframe["SALARY"].quantile(q3))
    dataframe = dataframe[(dataframe["SALARY"] < salary_up)]

    # KNN Imputer
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dataframe = pd.get_dummies(dataframe[cat_cols + num_cols], drop_first=True)

    scaler = RobustScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)

    imputer = KNNImputer(n_neighbors=5)
    dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)

    dataframe = pd.DataFrame(scaler.inverse_transform(dataframe), columns=dataframe.columns)

    dataframe.isnull().values.any()

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    for col in num_cols:
        if col != 'SALARY':
            dataframe[col] = RobustScaler().fit_transform(dataframe[[col]])

    y = dataframe["SALARY"]
    X = dataframe.drop(["SALARY"], axis=1)
    return X, y

# MODELLING

###############################################################################
# Base Models
###############################################################################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 103.0764 (LR)
# RMSE: 94.967 (Ridge)
# RMSE: 90.941 (Lasso)
# RMSE: 90.0642 (ElasticNet)
# RMSE: 97.6488 (KNN)
# RMSE: 115.0021 (CART)
# RMSE: 85.6114 (RF)
# RMSE: 138.0191 (SVR)
# RMSE: 93.2643 (GBM)
# RMSE: 94.4312 (XGBoost)
# RMSE: 90.0676 (LightGBM)
###############################################################################
# Hyperparameter Optimization
###############################################################################

rf_params = {"max_depth": [5, 8, None],
             "max_features": [10, 7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 300, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [5, 8, 12],
                  "n_estimators": [100, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [100, 300, 500],
                   "colsample_bytree": [0.5, 0.7, 1]}

gbm_params = {"learning_rate": [0.1, 0.01, 0.001],
              "n_estimators": [100, 300, 500],
              "min_samples_split": [2, 5, 8],
              "max_depth": [3, 5, 8]}

regressors = [("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              ('GBM', GradientBoostingRegressor(), gbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

# ########## RF ##########
# RMSE: 84.8306 (RF)
# RMSE (After): 81.9663 (RF)
# RF best params: {'max_depth': 8, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 100}
# ########## XGBoost ##########
# RMSE: 94.4312 (XGBoost)
# RMSE (After): 91.141 (XGBoost)
# XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100}
# ########## LightGBM ##########
# RMSE: 90.0676 (LightGBM)
# RMSE (After): 86.7892 (LightGBM)
# LightGBM best params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 300}
# ########## GBM ##########
# RMSE: 93.115 (GBM)
# RMSE (After): 91.7813 (GBM)
# GBM best params: {'learning_rate': 0.01, 'max_depth': 3, 'min_samples_split': 5, 'n_estimators': 300}

###############################################################################
# # Stacking & Ensemble Learning
###############################################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"]),
                                         ('XGBoost', best_models["XGBoost"])])

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))
# 84.600
# We reduced the error to 84.600 that is highly better than the first one we get as nearly 250.

# feature importances
pre_model = RandomForestRegressor().fit(X, y)
feature_imp = pd.DataFrame({'Feature': X.columns, 'Value': pre_model.feature_importances_})
feature_imp.sort_values("Value", ascending=False).head(15)

#                   Feature  Value
# 7                  CATBAT 0.3876
# 9                    CRBI 0.1962
# 10                 CWALKS 0.0580
# 15      NEW_C_ATBAT_RATIO 0.0229
# 19      NEW_C_WALKS_RATIO 0.0217
# 5                   WALKS 0.0169
# 36             NEW_RUNNER 0.0154
# 16       NEW_C_HITS_RATIO 0.0147
# 32  NEW_NET_HELPFUL_ERROR 0.0143
# 1                    HITS 0.0137
# 18        NEW_C_RBI_RATIO 0.0136
# 4                     RBI 0.0125
# 33        NEW_TOTAL_SCORE 0.0122
# 29          NEW_CRBI_MEAN 0.0118
# 0                   ATBAT 0.0115
