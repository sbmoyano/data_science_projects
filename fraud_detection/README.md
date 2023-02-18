## Data Science Project: Fraud Detection

### **Aim**

The aim of the current project is to **train boosting and neural network models to discriminate between fraud and non-fraud transactions, prioritizing the detection of fraud transactions**. In general, it tries to solve a binary classification with unbalanced data (i.e., fraud transactions represent less than1% of all transactions)

As False Positives could also have a negative impact on the user experience with banking services, I will also try to **reduce** as much as possible the **False Positives**, without negatively impacting too much on True Positives. 

Throughout different notebooks created on Google Colab, I will perform: 
1. Exploratory Data Analysis (e.g., differences in fraud transactions by time of the day, day of the week, person’s sex, age, location, etc.)
2. Feature Engineering
3. Train boosting models (i.e., XGBoost and LightGBM models)
4. Train deep learning models (i.e., MLP and LSTM models)

### **Data**

Data for this project was downloaded from **Kaggle**: [Kaggle link](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

I choose this dataset as although it is synthetic data, **it provides with non-transformed data**. That is, all the raw variables are available, which enables to create new variables through feature engineering (*most of datasets for fraud detection are already transformed (e.g., using PCA) to keep all information anonymous*).

The dataset will be divided into train, validation and test sets depending on the requirements for each model.

* For **XGBoost, LightGBM and MLP** data will be **divided based on datetime information**: I will keep the most recent transactions to test the model, while the oldest ones will be used to train and validate.

* For **LSTM** data will be **divided based on sequences**, considering each cardholder a unique sequence: data will be reshaped to (cardholders [sequences], transactions [time steps], features [features])

### **Training**

To train the models, I considered the **imbalance in the dataset between fraud and non-fraud transactions**. 

* For **XGBoost and LightGBM** I have specified the **weight for the positive class**, which in our case is the underrepresented class.

* For the **MLP model**, I have specified the **class weights for the positive and negative class**. 

* As this parameter can not be specified for the **LSTM model**, I will have to introduce the weight of the classes in the loss function (i.e., **Weighted Binary Crossentropy**). Additionally, as for the LSTM all the sequences in the train, validation and test sets need to have the same length, I will have to apply **Padding**. In order to tell the model which padded values to ignore, I will use **sample weights** for the train and validation sets, as well as for the test set when evaluating the model (weight of 0 for padded values and 1 for the rest)

During training, a first “baseline” model with the default hyperparameters established by each library will be trained. Next, I will do a Random Search on boosting, regularization or learning rate parameters, comparing if the new hypermeters helped to improved the model based on our previous trained model.
For MLP and LSTM I will explore how reducing the learning rate, introducing an initial bias, adding regularization techniques (e.g., dropout) contributed to improve model predictions.

Finally, I will also **try different decision thresholds** instead of keeping the default .50

### **Results**

LSTM outperforms the rest of the models in AUC and False Positive Rate. Although, for True Positive Rate it is MLP the model with the highest score.

### **Limitations**

The main limitations of the current project are: 1. Amount of data; 2. Time for GPU use in Google Colab. Both of these aspects limit training times. Other limitations and future steps can be found at the end of the LSTM notebook.

