# Lucien's work for Engie V.I.E: forecasting of the daily baseload price

Note: please click on **crtl + shift + v** to execute this markdown and see the page.

**How to read this document**

- in src: 

    - **notebooks** are drafts used to make some researchs in the dataset and to build our algorithm step by step (data visualization and cleaning, research for the best model...). **Please read the Notebook if you want to understand how we delt with data and how we understand this work.**
    - **code**: code is the final and clean code that you will be able to run in one button, this is your main. (even if you have also clean code in notebook)

- in data: here are stored the two csv files

- venv for python installations

- gitignore for the git repository

**Tasks:**
- Build a forecasting model for the German price, specifically predicting the "daily baseload price." 
- Construct confidence metrics to determine when the model is reliable or not. 
- Provide your insights on additional inputs that, in your opinion, could improve the model’s quality.

**Steps:**

**1st step: data cleaning and data analysis**

We have two csv files and we need to conduct a study on: type value, Nan values, Null value (how to fill them?) potentials outliers, correlation and cointegration, statistical analysis on every distributions, potential scaling of the values?

**2nd step: choose of a simple algorithm to have a benchmark and then more sophisticated algorithms**

- first: linear regression

After this first regression: potential feature selection techniques?

- second: we try simple ML models adapted for time series like XGBoost (Tslearn) and do a finetuning of the hyper parameters with a genetical algorithm

- third: implementation of a neural network like LSTM

**3thd step: backtesting of the model**

We choose MEA metric to have the mean absolute error value of our model.

We study when the model can be trusted or not, to do that:

- we can make a lot of prediction, sort and keep predictions with a lot of errors, then plot the derivative of every feature at the moment of the error
- we can create an index of uncertancy based on the prediction models of sun and wind and see if the errors are related to this index

**4th step: find other inputs that could improve the quality of the forecasting**

