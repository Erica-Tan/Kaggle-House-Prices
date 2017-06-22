# Kaggle House Prices Prediction

This project aims to create some regression models to predict the sale price of each home in Ames, Lowa.The dataset was collected from Kaggle competition website (https://www.kaggle.com/c/house-prices-advanced-regression-techniques), which includes 1460 observations and 81 variables that cover all aspects of residential homes. 


Installation
----------------------

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.

Usage
-----------------------

* Run `data_preposessing.py` to cteate the `train` and `test` datasets.
    * This will create `train+dummy+log+minmax.csv` and `test+dummy+log+minmax.csv` in the `processed` folder.
* Run `model_xgboost.py`.
    * This will run XGBoost across the training set, and evaluate the model on test set.
    * It will save the final model called `xgboost.dat` to the `output` folder.
* Run `model_neural_network.py`.
    * This will run Neural Network across the training set, and evaluate the model on test set.
    * It will save the final model called `nn_model.h5` and `nn_model.json` to the `output` folder.

