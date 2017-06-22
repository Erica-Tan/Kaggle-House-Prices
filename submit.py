import settings
import helper as hl
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



# Load data
train = hl.load_csv("train", "data")
test = hl.load_csv("test", "data")


# combine the datasets
df_data = pd.concat([train, test])



#========= Start Missing Data==============
missing  = df_data.isnull().sum()
missing = missing[missing > 0]
total  = missing.sort_values(ascending=False)
percent = (missing/len(df_data)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])

#plot missing data
missing.plot.bar()
plt.show()

# PoolQC
filtered = (df_data["PoolQC"].isnull())
df_data.loc[filtered,["PoolQC","PoolArea"]]

#PoolQC can be filled with ‘None’ since each has a corresponding PoolArea of 0
hl.fill_missing_data(df_data, "PoolQC", "None")

#df_data["PoolQC"].value_counts(dropna=False)


# GarageType/GarageYrBlt/GarageFinish/GarageCars/GarageArea/GarageQual/GarageCond
garage_cols=["GarageType","GarageQual","GarageCond","GarageYrBlt","GarageFinish","GarageCars","GarageArea"]

filtered = df_data[garage_cols].isnull().apply(lambda x: any(x), axis=1)
df_data.loc[filtered, garage_cols]

for column in garage_cols:
    if df_data.dtypes[column] == "object":
        hl.fill_missing_data(df_data, column, "None")
    else:
        hl.fill_missing_data(df_data, column, 0)

#df_data[garage_cols].isnull().sum()

# float to int64
df_data["GarageYrBlt"] = df_data["GarageYrBlt"].astype("int64")

   
# Electrical: Electrical system
sns.countplot(df_data["Electrical"])
plt.show()

# fill with most frequent value 
hl.fill_missing_data(df_data, "Electrical", "SBrkr")


# BsmtQual / BsmtCond / BsmtExposure / BsmtFinType1 / BsmtFinType2
basement_cols=["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtFinSF1","BsmtFinSF2"]

filtered = df_data[basement_cols].isnull().apply(lambda x: any(x), axis=1)
df_data.loc[filtered, basement_cols]

# Rows 948 are the only missing value from BsmtExposure, fill this with No
df_data.loc[948, "BsmtExposure"] = "No"

# Rows 332 are the only missing values from BsmtFinType2,
# and fill the missing values of each BsmtFinType2 based on BsmtFinSF2.
grouped = df_data.groupby("BsmtFinType2")
grouped = grouped["BsmtFinSF2"].agg(np.mean)
grouped

df_data.loc[332, "BsmtFinType2"] = "LwQ"


# The rest of the basement columns fill with 0 since they likely don’t have a basement
# and the categoric missing values will be filled with None.
for column in basement_cols:
    if df_data.dtypes[column] == "object":
        hl.fill_missing_data(df_data, column, "None")
    else:
        hl.fill_missing_data(df_data, column, 0)

df_data[basement_cols].isnull().sum()


# MasVnrType/MasVnrArea
filtered = (df_data["MasVnrType"].isnull()) | (df_data["MasVnrArea"].isnull())
df_data.loc[filtered,["MasVnrType","MasVnrArea"]]

hl.fill_missing_data(df_data, "MasVnrType", "None")
hl.fill_missing_data(df_data, "MasVnrArea", 0)

#df_data[["MasVnrType","MasVnrArea"]].isnull().sum()


# LotFrontage
# group by each neighborhood and take the median of each LotFrontage
# and fill the missing values of each LotFrontage based on what neighborhood the house comes from
grouped = df_data.groupby("Neighborhood")
grouped = grouped["LotFrontage"].agg(np.mean)
    
filtered = df_data["LotFrontage"].isnull()
df_data.loc[filtered,"LotFrontage"] = df_data.loc[filtered,"Neighborhood"].map(lambda neighbor : grouped[neighbor])

#df_data[["LotFrontage"]].isnull().sum()


# Fence/MiscFeature
hl.fill_missing_data(df_data, "Fence", "None")
hl.fill_missing_data(df_data, "MiscFeature", "None")


# FireplaceQu
# check to see if any of the missing values for FireplaceQu come from houses that recorded having at least 1 fireplace.
filtered = (df_data["Fireplaces"] > 0) & (df_data["FireplaceQu"].isnull())
len(df_data.loc[filtered,["Fireplaces","FireplaceQu"]])

# All the houses that have missing values did not record having any fireplaces, replace the NA’s with ‘None’
hl.fill_missing_data(df_data, "FireplaceQu", "None")


# Alley
hl.fill_missing_data(df_data, "Alley", "None")



# test
hl.fill_missing_data(df_data, "BsmtHalfBath", 0)
hl.fill_missing_data(df_data, "BsmtFullBath", 0)
hl.fill_missing_data(df_data, "TotalBsmtSF", 0)
hl.fill_missing_data(df_data, "BsmtUnfSF", 0)

#=========End missing Data==============



# Some numerical features are actually really categories
df_data = df_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                           "OverallQual" : {10 : "Ve", 9 : "Ex", 8 : "Vg" , 7 : "Gd", 6 : "AA",
                                        5 : "TA", 4 : "BA" , 3 : "Fa", 2 : "Po", 1 : "Vp"},
                           "OverallCond" : {10 : "Ve", 9 : "Ex", 8 : "Vg" , 7 : "Gd", 6 : "AA",
                                        5 : "TA", 4 : "BA" , 3 : "Fa", 2 : "Po", 1 : "Vp"}
                           })

# Year / Month
# records that a house was remodelled if the year it was built is different than the remodel year.
df_data["Remodeled"] = (df_data["YearRemodAdd"] != df_data["YearBuilt"]) * 1

# if the houses have been recently remodelled 
df_data["RecentRemodel"] = (df_data["YearRemodAdd"] == df_data["YrSold"]) * 1
    
# Was this house sold in the year it was built?
df_data["NewHouse"] = (df_data["YearBuilt"] == df_data["YrSold"]) * 1

# how old a house is
df_data["HouseAge"] = date.today().year - df_data["YearBuilt"]

# how long ago the house was sold
df_data["TimeSinceSold"] = date.today().year - df_data["YrSold"]

# how many years since the house was remodelled and sold 
df_data["YearSinceRemodel"] = df_data["YrSold"] - df_data["YearRemodAdd"]


# the largest proportion of houses sold is during the summer months: May, June, July
df_data["HighSeason"] = (df_data["MoSold"].isin([5,6,7])) * 1


# Area
area_columns = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
             "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF", 
             "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "LowQualFinSF", "PoolArea"]

df_data["TotalArea"] = df_data[area_columns].sum(axis =1)

# Binarize Data
df_data["HasLowQualArea"] = (df_data["LowQualFinSF"] > 0) * 1
df_data["HasBasement"] = (df_data["TotalBsmtSF"] > 0) * 1
df_data["HasGarage"] = (df_data["GarageArea"] > 0) * 1
df_data["Has2ndFloor"] = (df_data["2ndFlrSF"] > 0) * 1
df_data["HasMasVnr"] = (df_data["MasVnrArea"] > 0) * 1
df_data["HasWoodDeck"] = (df_data["WoodDeckSF"] > 0) * 1
df_data["HasOpenPorch"] = (df_data["OpenPorchSF"] > 0) * 1
df_data["HasScreenPorch"] = (df_data["ScreenPorch"] > 0) * 1
df_data["HasEnclosedPorch"] = (df_data["EnclosedPorch"] > 0) * 1
df_data["Has3SsnPorch"] = (df_data["3SsnPorch"] > 0) * 1
df_data["HasPool"] = (df_data["PoolArea"] > 0) * 1


# Neighborhood
nbrh_rich = ["NridgeHt", "NoRidge", "StoneBr", "Veenker", "Timber", "Somerst"]
df_data["NbrhRich"] = df_data["Neighborhood"].apply(lambda x: 1 if x in nbrh_rich else 0)

# Remove columns
#sns.countplot(df_data["Utilities"])
#plt.show()

#Garage area is strongly correlated with number of cars, remove one
#1stFlrSF + 2nFlrSF + LowQualFinSF = GrLivArea.
#Utilities only has 1 value for NoSeWa and the rest AllPub
#Since we have constructed some features for date attribures, we removed the original attributes
del_columns = ["GarageCars", "1stFlrSF", "2ndFlrSF", "Utilities", "YearBuilt",
               "YearRemodAdd", "YrSold", "MoSold", "GarageYrBlt"]

df_data = df_data.drop(del_columns, axis=1)

# Record numeric features
num_columns = df_data.select_dtypes(include = ["float64", "int64"]).columns.values.tolist()
num_columns.remove("SalePrice")
num_columns.remove("Id")

# Transform each categorical variable into binary features.
df_data = pd.get_dummies(df_data)



# log transform the attributes that follow the normal distribution 
log_columns = ["LotFrontage", "LotArea","BsmtUnfSF",
               "TotalBsmtSF", "GrLivArea", "TotalArea", "SalePrice"]

# a skewness with an absolute value > 0.8 is considered at least moderately skewed
skewed = df_data[log_columns].apply(lambda x: x.dropna().astype(float).skew())
skewed = skewed[(skewed > 0.8) | (skewed < -0.8)]
skewed = skewed.index.tolist()

hl.log_transform(df_data, skewed)


train = df_data.iloc[:1460,:]

test = df_data.iloc[1460:,:]


scaler = MinMaxScaler()
train.loc[:,num_columns] = scaler.fit_transform(train[num_columns])
test.loc[:,num_columns] = scaler.transform(test[num_columns])


train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train = train['SalePrice']
x_train = train.drop(['SalePrice', 'Id'], axis=1)
ids = test['Id']
x_test = test.drop(['SalePrice', 'Id'], axis=1)

model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=300,
     max_depth=3,
     min_child_weight=1,
     gamma=0,
     subsample=0.9,
     colsample_bytree=0.4,
     nthread=2,
     scale_pos_weight=1,
     reg_alpha=0.01,
     seed=settings.SEED)

model.fit(x_train, y_train, eval_metric='rmse')

# make predictions for test data
y_test = model.predict(x_test)


submission = pd.DataFrame()
submission['Id'] = ids
submission['SalePrice'] =  np.expm1(y_test)


# saving submission
submission.to_csv('submission.csv', index=False)



