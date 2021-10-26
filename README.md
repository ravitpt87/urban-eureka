# Women-headed households in South Africa

Female household headship has been on the rise in South Africa in recent years. Compared to male-headed households, female-headed households tend to face greater social and economic challenges. Female-headed households, in general, are more vulnerable to lower household incomes and higher rates of poverty.

Here, we work with South African census data on female headship and income levels of every household across the country from 2011.

The objective of this challenge is to build a predictive model that accurately estimates the % of households per ward that are female-headed and living below a particular income threshold by using data points that can be collected through other means without an intensive household survey like the census.

This solution can potentially reduce the cost and improve the accuracy of monitoring key population indicators such as female household headship and income level in between census years. The winning solutions will be made publicly available at the end of the competition.

Data used are downloaded from the [Zindi challenge](https://zindi.africa/competitions/womxn-in-big-data-south-africa-female-headed-households-in-south-africa)

We train a simple model in the jupyter notebooks, where we select only some features and do minimal cleaning.


## Models Groupwork

### Template Model Notebook
A template with data import and the same train-test-split for easier comparison between different models.

### Baseline Model
Notebook containing our baseline model, predicting based on 1 feature.

### Linear Regression Model
Notebook containing models using Linear Regression and Polynomial Regression.

### Ridge and Lasso Regularization Model
Notebook containing models using Ridge and Lasso.

### KNN Model
Notebook containing our KNN modelling approach.

### Final Model
Notebook containing our final model.

## Plots
Contains our baseline model, output from error analysis and feature importance.

## Notebooks


### Final Notebook
Our final notebook, containing the most useful steps in EDA, data transformation, baseline model and resulting in our final model.

### Firstdraft
Notebooks for experimenting with dataset, mostly EDA and feature investigations.

### Starter Notebook
[Starter Notebook](https://colab.research.google.com/drive/19dHG6RIQapPZTnPYGtsBWRbqW0-t07oJ), provided by Zindi.



##
Requirements:
- pyenv with Python: 3.9.4

### Environment

Same procedure as last time...

Use the requirements file in this repo to create a new environment.

```BASH
make setup 

#or 

pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install eli5
pip install yellowbrick

```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python train.py  
```

In order to test that predict works on a test set you created run:

```bash
python predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible
