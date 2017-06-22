import os
import settings
import helper as hl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib

# Load data
df_data = hl.load_csv("train+dummy+log+minmax")

# split data into train and validation
train, validation = train_test_split(df_data, test_size=0.2, random_state=settings.SEED)

# split into input (X) and output (Y) variables
y_train = train['SalePrice']
x_train = train.drop('SalePrice', axis=1)
y_validation = validation['SalePrice']
x_validation = validation.drop('SalePrice', axis=1)
Y = df_data['SalePrice']
X = df_data.drop('SalePrice', axis=1)

# Function to create model
def model_fit(model, x_train, y_train, x_validation, y_validation, early_stopping_rounds, show_results = 0, show_plot = 0):
    #Fit the algorithm on the data
    eval_set = [(x_train, y_train),(x_validation, y_validation)]
    model.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric='rmse', eval_set=eval_set, verbose=False)

    if show_results:
        # make predictions for train data
        y_train_pred = model.predict(x_train)
        y_train_pred = [value for value in y_train_pred]
        # evaluate predictions
        y_train_true = np.expm1(y_train.values)
        y_train_pred = np.expm1(y_train_pred)
        print("train RMSE: %f" % (np.sqrt(mean_squared_error(y_train_true, y_train_pred))))

        # make predictions for validation data
        y_validation_pred = model.predict(x_validation)
        y_validation_pred = [value for value in y_validation_pred]
        # evaluate predictions
        y_validation_true = np.expm1(y_validation.values)
        y_validation_pred = np.expm1(y_validation_pred)
        print("validation RMSE: %f" % (np.sqrt(mean_squared_error(y_validation_true, y_validation_pred))))

    if show_plot:
        #Plot important
        xgb.plot_importance(model, max_num_features = 20)
        plt.show()

    return model


# Baseline model
early_stopping_rounds = 10
show_results = 1
show_plot = 0

model = xgb.XGBRegressor(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=2,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     seed=settings.SEED)

model = model_fit(model, x_train, y_train, x_validation, y_validation, early_stopping_rounds, show_results, show_plot)


#===========Start Parameters Tuning========
#Tune learning rate and number of estimators
model = xgb.XGBRegressor(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=2,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     seed=settings.SEED)

# define the grid search parameters
# best parameters {'n_estimators': 250 / 150, 'learning_rate': 0.1}
learning_rate = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
n_estimators = range(50, 400, 50)

param_test = {
 'learning_rate': learning_rate,
  'n_estimators': n_estimators
}

gsearch = GridSearchCV(estimator=model, scoring="neg_mean_squared_error", n_jobs=-1,
                        param_grid=param_test, verbose=0)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

'''
# plot results
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Mean squared error')
plt.show()

# Plot performance for learning_rate=0.1
plt.plot(n_estimators, scores[1])
plt.xlabel('n_estimators')
plt.ylabel('Mean squared error')
plt.title('XGBoost learning_rate=0.1 n_estimators vs Log Loss')
plt.show()
'''


# Tune max_depth and min_child_weight
model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=150,
     max_depth=5,
     min_child_weight=2,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     seed=settings.SEED)

# define the grid search parameters
# best parameters {'min_child_weight': 5, 'max_depth': 5}
param_test = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}

gsearch = GridSearchCV(estimator=model, scoring="neg_mean_squared_error", n_jobs=-1,
                        param_grid=param_test, verbose=0)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# Tune gamma
model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=150,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     seed=settings.SEED)

# define the grid search parameters
# best parameters {'gamma': 0.0}
param_test = {
 'gamma':[i/10.0 for i in range(0,5)]
}

gsearch = GridSearchCV(estimator=model, scoring="neg_mean_squared_error", n_jobs=-1,
                        param_grid=param_test, verbose=0)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# Tune subsample and colsample_bytree
model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=150,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     seed=settings.SEED)

# define the grid search parameters
# best parameters {'colsample_bytree': 0.6, 'subsample': 0.8}
subsample = [i/100.0 for i in range(10,110,10)]
colsample_bytree = [i/100.0 for i in range(10,110,10)]

param_test = {
 'subsample': subsample,
 'colsample_bytree':colsample_bytree
}

gsearch = GridSearchCV(estimator=model, scoring="neg_mean_squared_error", n_jobs=-1,
                        param_grid=param_test, verbose=0)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

'''
# plot results
scores = np.array(means).reshape(len(colsample_bytree), len(subsample))
plt.errorbar(subsample, scores[7])
plt.title("XGBoost subsample vs Log Loss")
plt.xlabel('subsample')
plt.ylabel('Mean squared error')
plt.show()
'''

# Tuning Regularization Parameters
model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=150,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.6,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     seed=settings.SEED)

# define the grid search parameters
# best parameters {'reg_alpha': 0.005}
param_test = {
  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

gsearch = GridSearchCV(estimator=model, scoring="neg_mean_squared_error", n_jobs=-1,
                        param_grid=param_test, verbose=0)
gsearch.fit(x_train,y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#===========End Parameters Tuning========

'''
# Visualizing the Bias-Variance Trade-Off
model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=150,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.6,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     reg_alpha=0.005,
     seed=settings.SEED)

early_stopping_rounds = 10
show_results = 1

model = model_fit(model, x_train, y_train, x_validation, y_validation, early_stopping_rounds, show_results)

results = model.evals_result()

# retrieve performance metrics
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# plot curve
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
ax.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Validation Curve for XGBoost')
#plt.show()
plt.savefig(os.path.join(settings.FIGURE_DIR, "xgboost_curve.png"))
plt.close()
'''

# Final model
model = xgb.XGBRegressor(
     learning_rate=0.1,
     n_estimators=150,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.6,
     colsample_bytree=0.8,
     nthread=2,
     scale_pos_weight=1,
     reg_alpha=0.005,
     seed=settings.SEED)

# Save final model (combine train and validation)
model.fit(X, Y, eval_metric='rmse')

# save model to file
joblib.dump(model, os.path.join(settings.OUTPUT_DIR, "xgboost.dat"))


# load model from file
loaded_model = joblib.load(os.path.join(settings.OUTPUT_DIR, "xgboost.dat"))

# load test data
test_data = hl.load_csv("test+dummy+log+minmax")
y_test = test_data['SalePrice']
x_test = test_data.drop('SalePrice', axis=1)

# make predictions for test data
y_test_pred = loaded_model.predict(x_test)
y_test_pred = [value for value in y_test_pred]
# evaluate predictions
y_test_real = np.expm1(y_test.values)
y_test_pred = np.expm1(y_test_pred)
print("test RMSE: %f" % (np.sqrt(mean_squared_error(y_test_real, y_test_pred))))



'''
# Plot Predicted vs Actual
sns.regplot(x = y_test_pred, y = y_test_real)
plt.title("XGBoost Predicted vs Actual")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
#plt.show()
plt.savefig(os.path.join(settings.FIGURE_DIR, "xgboost_predictions.png"))
plt.close()
'''

'''
# plot single tree
plot_tree(loaded_model, num_trees=50, rankdir='LR')
plt.show()
'''


#========Feature selection
'''
# Fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
thresholds = np.unique(thresholds)

thresholds = thresholds[:-1]



train_results = []
validation_results = []

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)

    # train model
    selection_model = xgb.XGBRegressor(
                                     learning_rate =0.1,
                                     n_estimators=100,
                                     max_depth=20,
                                     min_child_weight=5,
                                     gamma=0,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     nthread=2,
                                     scale_pos_weight=1,
                                     seed=settings.SEED)
    selection_model.fit(select_x_train, y_train)


    # eval model
    y_pred = selection_model.predict(select_x_train)
    predictions = [round(value) for value in y_pred]
    train_rmse = hl.rmse_eval(y_train, predictions)
    train_results.append(train_rmse)

    # eval model
    select_x_validation = selection.transform(x_validation)
    y_pred = selection_model.predict(select_x_validation)
    predictions = [round(value) for value in y_pred]
    validation_rmse = hl.rmse_eval(y_validation, predictions)
    validation_results.append(validation_rmse)

    print("Thresh=%f, n=%d, train RMSE: %f, validation RMSE: %f" % (thresh, select_x_train.shape[1], train_rmse, validation_rmse))

'''

