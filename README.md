# Starbucks-Offer-Analysis

The motivation of this project is to get as many insights from the Starbucks 
datasets to answers our questions of interest, which are crucial for 
business decision-making. 

Here, we analyse the simulated data that mimics customer behavior on the 
Starbucks rewards mobile app. The dataset is a simplified version 
of the real Starbucks app.

There are three kinds of data **demographic**, **transaction** and **offer data**. 

**Demographic** data shows the demographic details of different users.

**Transactional** data shows user purchases made on the app including the timestamp 
of purchase and the amount of money spent on a purchase. It also has a record 
for each offer that a user receives as well as a record for when a user actually 
views the offer. There are also records for when a user completes an offer. 

**Offer** data shows different offer types along with other details as follows:

    * Minimum required spend to complete an offer
    * Reward given for completing an offer
    * Time for offer to be open, in days

An offer can be merely an advertisement for a drink or an actual offer such 
as a discount or BOGO (buy one get one free).


#### Offer related facts considered throughout the project

* Every offer has a validity period before the offer expires.
* Informational offers have a validity period even though these ads 
are merely providing information about a product.
* Someone using the app might make a purchase through the app without 
having received an offer or seen an offer.
* Some demographic groups will make purchases even if they don't receive 
an offer
    

Prerequisites
-------------
The following libraries are used for the project:

    argparse
    pandas
    SQLAlchemy
    scikit-learn

## CRISP-DM

* ### Business Understanding

   **The questions of interest for the Starbucks dataset are as follows:**

    * Which demographic groups respond best to which offer type?
    * Which demographic groups will make purchases even if they don't receive an offer?
    * Who spends the most amount(within each demographics)?
    * Build a machine learning model that predicts how much someone will spend based on demographics and offer type.
    * Build a model that predicts whether or not someone will respond to an offer.

    The first three problem statement can be solved using statistical analysis.
    The fourth problem statement can be solved using linear regression and the last
    problem statement can be solved using a classifier.
    
    For linear regression we can use the **r2 score** to evaluate the model, 
    whereas for the classifier we can used the **f1-score**.

* ### Data Understanding

    Now we have the question, we need to move the question into the data. Find the columns from the datasets that would answer these questions.
    
    **The columns identified to answer the necessary questions are as below:**
  
    * Which demographic groups respond best to which offer type?
         -  age, gender, became_member_on, income, offer type
     
    * Which demographic groups will make purchases even if they don't receive an offer?
        -  age, offer id, gender, became_member_on, income, amount, person  
    
    * Who spends the most amount(within each demographics)?
        -  age, offer id, gender, became_member_on, income, amount, person  
     
    * Build a machine learning model that predicts how much someone will spend based on demographics and offer type.
        - difficulty, duration, reward, age, income, amount, year, month, offer_type, gender 
     
    * Build a model that predicts whether or not someone will respond to an offer.
        - reward, age, income, year, month, offer_type, gender
        
* ### Data Preparation

    [Data Preparation](https://github.com/jyothishkjames/Starbucks-Offer-Analysis/tree/master/data)
    
    The data preparation has various rigorous steps including the following:
    * Filling missing data
    * Removing data
    * Transforming data
    
* ### Modeling

    [Modelling](https://github.com/jyothishkjames/Starbucks-Offer-Analysis/tree/master/model)
    
    For predicting whether or not someone will respond to an offer, we split the prepared data into 
    train and test data. The training data is then used to fit Random Forest Classifier and 
    Support Vector Classifier respectively. Thereafter, the test data is used to test and compare the models.

* ### Evaluation

    [Evaluation](https://github.com/jyothishkjames/Starbucks-Offer-Analysis/tree/master/model)
    
    For the evaluation, we use the f1 score score to get an understanding of how well our model works. 
    The Random Forest Classifier model has an f1 score 0.68 and 0.75 for prediction 'yes' and 'no' respectively.
    The Support Vector Classifier model has an f1 score 0.71 and 0.77 for prediction 'yes' and 'no' respectively.
    The closer the score is to 1, the better your model fits the data. Hence we conclude that the Support Vector 
    Classifier model performs better than the Random Forest Classifier model.

* ### Deployment

    Deployment is the stage where we applying the conclusion to our Business.  To recap, the conclusions are based on the statistical inference and the model prediction.        
        
        
 ## Summary of Data Analysis
 
 Form the statistical analysis, the following were inferred:
 
 * 62.2 % of Men who were 35 years old responded to discount offer type
 * 59.3 % of Women who were 35 years old responded to discount offer type
 * 63.8 % of Women who were 18 years old responded to bogo offer type
 * 54.4 % of Men who were 18 years old responded to bogo offer type
 * 50.6 % of Women who earned income in between 60000 and 100000 responded to bogo offer type
 * 51.5 % of Women who earned income in between 40000 and 60000 responded to bogo offer type
 * People of age 58 makes the most purchases without an offer
 * Men makes the most purchases without an offer
 * People who became member in the year 2017 makes the most purchases without an offer
 * People who has an income around 73000.0 makes the most purchases without an offer
 
 ## Link to the Blog
  
 [Link to the Blog]()          
  
 
