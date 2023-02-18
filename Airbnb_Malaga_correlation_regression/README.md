## Data Science Project: Predict price for Airbnb accomodations

This notebook is intended to be a first approximation to regression models using Sklearn, avoiding more "statistical" approaches using IBM SPSS or StatsModels.

To offer a general overview of the purpose of this notebook, I will:

* Perform some data cleaning and data exploration. New features will be created in the process and added to the predictor variables.

* Explore the distribution of the predictor and target variables, applying data transformation when neccesary.

* A first feature selection for categorical data will be carried out running Analysis of Variance (ANOVA) before performing ordinal and dummy coding of these variables.

* Multicollinearity will be checked based on Variance Inflation Factor (VIF), removing features when the VIF is above the established threshold.

* After standardize the data, correlation analysis will be performed to have a general idea of significant predictors in the data. Predictors with a correlation values above 0.9 and also features with a high fraction of constant values will be removed.

* Multiple methods are applied for feature selection (F Regression, Random Forest, Recursive Feature Elimination with Lasso Regression, Backward Elimination and Stepwise Elimination). These different selected features will be saved and models will be fitted in order to compare model performance based on these.

* Finally, Lasso, Ridge, ElasticNet and Random Forest Regression models are meant to be build. Model performance will be evaluated on different metrics.
