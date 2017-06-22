import settings
import numpy as np
# fix random seed for reproducibility
np.random.seed(settings.SEED)

import helper as hl
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

# Load data
df_data = hl.load_csv("train+dummy+log+minmax")

# split data into train and validation
train, validation = train_test_split(df_data, test_size=0.2, random_state=settings.SEED)

# split into input (X) and output (Y) variables
y_train = train['SalePrice'].as_matrix()
x_train = train.drop('SalePrice', axis=1).as_matrix()
y_validation = validation['SalePrice'].as_matrix()
x_validation = validation.drop('SalePrice', axis=1).as_matrix()
Y = df_data['SalePrice'].as_matrix()
X = df_data.drop('SalePrice', axis=1).as_matrix()

# number of features
num_features = x_train.shape[1]

# Function to create baseline model
def baseline_model(num_features):
	# create model
	model = Sequential()
	model.add(Dense(num_features, input_dim=num_features, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')

	return model


# Baseline model
model=baseline_model(num_features)

#Fit the model
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=150, batch_size=100, verbose=0)

# make predictions for train data
y_train_pred = model.predict(x_train)
y_train_pred = np.asarray([value[0] for value in y_train_pred])
y_train_real = np.expm1(y_train)
y_train_pred = np.expm1(y_train_pred)
print("train RMSE: %f" % (np.sqrt(mean_squared_error(y_train_real, y_train_pred))))

# make predictions for validation data
y_validation_pred = model.predict(x_validation)
y_validation_pred = np.asarray([value[0] for value in y_validation_pred])
y_validation_real = np.expm1(y_validation)
y_validation_pred = np.expm1(y_validation_pred)
print("validation RMSE: %f" % (np.sqrt(mean_squared_error(y_validation_real, y_validation_pred))))


#==========Start Parameters Tuning========

# Tune batch_size / epochs
# Function to create model
def create_model1(num_features):
	# create model
	model = Sequential()
	model.add(Dense(num_features, kernel_initializer='random_normal', input_dim=num_features, activation='relu'))
	model.add(Dense(num_features, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model

# create model
model1 = KerasRegressor(build_fn=create_model1, num_features=num_features, verbose=0)

# define the grid search parameters
# best parameters {'batch_size': 80, 'epochs': 100}
param_grid = {
	'batch_size': [80, 100, 120, 140],
	'epochs': [100, 150, 200, 250, 300, 350, 400]
}

# default 3-fold cross validation
# n_jobs is set to -1, the process will use all cores on your machine. 
gsearch = GridSearchCV(estimator=model1, scoring="neg_mean_squared_error", 
					param_grid=param_grid, n_jobs=-1)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# Tune the number of Neurons on secone layer
# Function to create model
def create_model2(num_features, neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, kernel_initializer='random_normal', input_dim=num_features, activation='relu'))
	model.add(Dense(num_features, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model

# create model
model2 = KerasRegressor(build_fn=create_model2, num_features=num_features, epochs=100, batch_size=80, verbose=0)

# define the grid search parameters
# best parameters {'neurons': 450}
param_grid = {
 'neurons': [i for i in range(100, 500, 50)]
}

gsearch = GridSearchCV(estimator=model2, scoring="neg_mean_squared_error", 
					param_grid=param_grid, n_jobs=-1)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# Tune the number of Neurons on third layer
# Function to create model
def create_model3(num_features, neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(450, kernel_initializer='random_normal', input_dim=num_features, activation='relu'))
	model.add(Dense(neurons, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model

# create model
model3 = KerasRegressor(build_fn=create_model3, num_features=num_features, epochs=100, batch_size=80, verbose=0)

# define the grid search parameters
# best parameters {'neurons': 100}
param_grid = {
 'neurons': [i for i in range(100, 500, 50)]
}

gsearch = GridSearchCV(estimator=model3, scoring="neg_mean_squared_error", 
					param_grid=param_grid, n_jobs=-1)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# Tune Dropout Regularization 
# Function to create model
def create_model4(num_features, dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(100, kernel_initializer='random_normal', input_dim=num_features, activation='relu', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(100, kernel_initializer='random_normal', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model

# create model
model4 = KerasRegressor(build_fn=create_model4, num_features=num_features, epochs=100, batch_size=80, verbose=0)

# define the grid search parameters
# best parameters {'weight_constraint': 1, 'dropout_rate': 0.0}
param_grid = {
	'weight_constraint': [1, 2, 3, 4, 5],
	'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

gsearch = GridSearchCV(estimator=model4, scoring="neg_mean_squared_error", 
					param_grid=param_grid, n_jobs=-1)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




# Tune learn_rate
# Function to create model
def create_model5(num_features, learn_rate=0.001):
	# create model
	model = Sequential()
	model.add(Dense(100, kernel_initializer='random_normal', input_dim=num_features, activation='relu'))
	model.add(Dense(100, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	optimizer = Adam(lr=learn_rate)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model

# create model
model5 = KerasRegressor(build_fn=create_model5, num_features=num_features, epochs=100, batch_size=80, verbose=0)

# define the grid search parameters
# best parameters {'learn_rate': 0.001}
learning_rate = [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
param_grid = {
 'learn_rate': learning_rate
}

gsearch = GridSearchCV(estimator=model5, scoring="neg_mean_squared_error", 
					param_grid=param_grid, n_jobs=-1)
gsearch.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (gsearch.best_score_, gsearch.best_params_))
means = gsearch.cv_results_['mean_test_score']
stds = gsearch.cv_results_['std_test_score']
params = gsearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#==========End Parameters Tuning========

# Final model
def create_final_model(num_features):
	# create model
	model = Sequential()
	model.add(Dense(100, kernel_initializer='random_normal', input_dim=num_features, activation='relu'))
	model.add(Dense(100, kernel_initializer='random_normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='random_normal'))
	# Compile model
	optimizer = Adam(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model


model6 = create_final_model(num_features)

#Fit the model
model6.fit(x_train, y_train, epochs=100, batch_size=80, verbose=0)

# make predictions for train data
y_train_pred = model6.predict(x_train)
y_train_pred = np.asarray([value[0] for value in y_train_pred])
y_train_real = np.expm1(y_train)
y_train_pred = np.expm1(y_train_pred)
print("train RMSE: %f" % (np.sqrt(mean_squared_error(y_train_real, y_train_pred))))

# make predictions for validation data
y_validation_pred = model6.predict(x_validation)
y_validation_pred = np.asarray([value[0] for value in y_validation_pred])
y_validation_real = np.expm1(y_validation)
y_validation_pred = np.expm1(y_validation_pred)
print("validation RMSE: %f" % (np.sqrt(mean_squared_error(y_validation_real, y_validation_pred))))


'''
# Visualizing the Bias-Variance Trade-Off
param_range = [i for i in range(1, 400, 20)]

model = KerasRegressor(build_fn=create_model, num_features=num_features, epochs=100, batch_size=80, verbose=0)

train_scores, test_scores = validation_curve(
    model, x_train, y_train, param_name="epochs", param_range=param_range,
    cv=settings.CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=-1)

train_scores_mean_rmse = np.sqrt(np.mean(np.abs(train_scores), axis=1))
test_scores_mean_rmse = np.sqrt(np.mean(np.abs(test_scores), axis=1))

# plot
fig, ax = plt.subplots()
ax.plot(param_range, train_scores_mean_rmse, label='Train')
ax.plot(param_range, test_scores_mean_rmse, label='Cross validation')
ax.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Validation Curve for Neural Network')
#plt.show()
plt.savefig(os.path.join(settings.FIGURE_DIR, "nn_curve.png"))
plt.close()
'''


# Save final model (combine train and validation)
final_model = create_final_model(num_features)

# fit the model
final_model.fit(X, Y, epochs=100, batch_size=80, verbose=0)

#model.summary()

# save model to file
# serialize model to JSON
model_json = final_model.to_json()
with open(os.path.join(settings.OUTPUT_DIR, "nn_model.json"), "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
final_model.save_weights(os.path.join(settings.OUTPUT_DIR, "nn_model.h5"))

# plot the model
#plot_model(model, to_file=os.path.join(settings.FIGURE_DIR, "nn_architecture.png"), show_shapes=True, show_layer_names=True)



# load json and create model
json_file = open(os.path.join(settings.OUTPUT_DIR, "nn_model.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(os.path.join(settings.OUTPUT_DIR, "nn_model.h5"))

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

# load test data
test_data = hl.load_csv("test+dummy+log+minmax")
y_test = test_data['SalePrice'].as_matrix()
x_test = test_data.drop('SalePrice', axis=1).as_matrix()

# make predictions for test data
y_test_pred = loaded_model.predict(x_test)
y_test_pred = np.asarray([value[0] for value in y_test_pred])
y_test_real = np.expm1(y_test)
y_test_pred = np.expm1(y_test_pred)
print("test RMSE: %f" % (np.sqrt(mean_squared_error(y_test_real, y_test_pred))))

'''
# Plot Predicted vs Actual
sns.regplot(x = y_test_pred, y = y_test_real)
plt.title("Neural Network Predicted vs Actual")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
#plt.show()
plt.savefig(os.path.join(settings.OUTPUT_DIR, "nn_predictions.png"))
plt.close()
'''
