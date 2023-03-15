
# Methodology 

 

The main goal of this project is to design or develop a web application which can help the banks or financial organisations to understand if the loan applications of the customers will be approved or not, how much risk percentage is associated with each loan application and make informed decisions about the loan approvals or rejections. We also wanted to focus on the needs of customers to help them understand what are the financial factors they can improve to reduce their risk percentage and get their loan approved. To design a web application which can fulfill the needs of both banks and customers there are few data processing and data prediction steps involved.  

First, we collected our data from the FICO Community official website which is the real data (HELOC applications from real owners). Real data is often helpful in building the model more appropriately. We performed few data pre-processing techniques like data cleaning and data manipulation. We also performed feature selection by removing highly correlated variables. To make our model selection more explainable and improve the performance of our web application or the predictions made, we performed hyper parameter tuning of different machine learning models or classifiers and selected the best classifier to train our model and make predictions which can consistently improve accuracy, reduce errors, minimise bias and overfitting issues. We validated our model using ROC-AUC score and deployed the model to design our web application using Stream lit and Ngrok. We used SHAP plots to explain the feature importance of the model and make informed decisions about the customers' loan approvals and rejections using uniquely generated SHAP plot for each customer. Following processes below will give you more detailed explanation about the steps involved in data pre-processing, model building and designing web application. 

## 1.1 Data collection  

The data we used training and developing our model is the HELOC applications from the real customers. Home Equity Line of Credit Dataset is sourced from the official website https://community.fico.com/s/explainable-machine-learning-challenge. FICO Company developed this real dataset from the HELOC applications of real homeowners. This dataset has 24 attributes where 23 attributes are predictors and 1 attribute is target variable and 10459 entries. The 23 characteristics/attributes might be categorical or quantitative. The target variable is Risk Performance, which has two labels: 'Good' and 'Bad.' Based on the 23 other variables (predictors), the risk performance is predicted. As the dataset has 10459 observations out of which 5000 observations are labelled as 'Good' which signifies that the applicants repaid their loan within 2 years without any overdue, while 5459 observations are labelled as ‘Bad’ which signifies that the applicants did not repay their loan within 2 years having history of overdue payments. 

## 1.2 Data preparation and manipulation 

### Data retrieval 

The data is retrieved from a single source as mentioned. The Pandas library was crucial in retrieving and understanding the data. As 48 percent of the data comprises applications which are labelled as 'Bad’ while 52 percent of the data comprises applications which are labelled as 'Good’ so we can conclude that our data is balanced. Even though the data is in tidy format (each observation has its own row, each variable/attribute has its own column, and each value has its own cell), this data requires data cleaning (removing duplicate entries/applications, handling missing values), data manipulation and feature selection before training the data and deploying the model. 

### Data cleaning 

We discovered that the data contains duplicate values. 587 of the 10459 entries are duplicates. Almost all the characteristics in those duplicates have missing values. Since all the attributes/columns of those observations are duplicated (I.e., the entire row is duplicated), they are dropped using the using the function drop_duplicates(). 

The data is checked for missing values using the function data_frame.isnull().sum(). It showed 0 null values because the data specification specifies -9 as the missing value. The data having –9 value in their cells is changed to null value. We then drop the null values using the function data_frame.dropna() as they degrade the performance of the model, and they are just contributing to less than 5% of the whole data and we will still have 9860 observations to build our model which are enough to build a good model. 

### Feature Selection  

We used correlation heat map of all the attributes/ features to perform feature selection and remove one of the attributes/features in the pair of the attributes/features have highest correlation. 

 
![image](https://user-images.githubusercontent.com/76246283/225264379-64e9bc03-1725-49ae-bd40-057f4eddf2d2.png)

Figure 1: Correlation heat map of dataset attributes 

We can notice one pair of attributes having high correlation (higher than 0.9) between them based on the correlation heatmap above. NumInqLast6M is significantly correlated with NumInqLast6Mexcl7days as they have highest correlation (0.99). As a result, either variable can be removed. It is because both variables have the same meaning, which is the number of inquiries in the previous 6 months, except that the second variable excludes the latest 7 days. As a result, the second variable is eliminated.   

The final dataset has following columns as detailed below: 

ExternalRiskEstimate - consolidated indicator of risk markers ("Good" or "Bad")  

MSinceOldestTradeOpen - number of months that have elapsed since first trade  

MSinceMostRecentTradeOpen - number of months that have elapsed since last opened trade  

AverageMInFile - average months in file  

NumSatisfactoryTrades - number of satisfactory trades  

NumTrades60Ever2DerogPubRec - number of trades which are more than 60 past due  

NumTrades90Ever2DerogPubRec - number of trades which are more than 90 past due  

PercentTradesNeverDelq - percent of trades, that were not delinquent 

MSinceMostRecentDelq - number of months that have elapsed since last delinquent trade  

MaxDelq2PublicRecLast12M - the longest delinquency period in last 12 months  

MaxDelqEver - the longest delinquency period  

NumTotalTrades - total number of trades  

NumTradesOpeninLast12M - number of trades opened in last 12 months  

PercentInstallTrades - percent of installments trades  

MSinceMostRecentInqexcl7days - months since last inquiry (excluding last 7 days)  

NumInqLast6M - number of inquiries in last 6 months  

NetFractionRevolvingBurden - revolving balance divided by credit limit 

NetFractionInstallBurden - installment balance divided by original loan amount 

NumRevolvingTradesWBalance - number of revolving trades with balance 

NumInstallTradesWBalance - number of installment trades with balance 

NumBank2NatlTradesWHighUtilization - number of trades with high utilization ratio (credit utilization ratio - the amount of a credit card balance compared to the credit limit)  

PercentTradesWBalance - percent of trades with balance 

 

 

### Data Manipulation 

To make the model easier to comprehend, label or categorical encoding might be used. Label encoding is assigning numerical values to category variables. There are several ways available, such as one-hot encoding, label encoding, and the use of cat.codes. In our data, the target variable is a binary variable (I.e., it has 2 classes only) so the target variable (categorical variable), is encoded as 0 for ‘Bad’ and 1 for ‘Good’. 

Detecting Outliers 

Even though there are outliers observed in all 23 variables/attributes, we did not exclude any outliers because the data we are using is the real data (HELOC applications of real homeowners) assuming there are no data entries made, we trained the model without excluding outliers because the data contains real information and when there are outliers observed in real data, there is no need of excluding them because they might hold an important information that can be used for training. 

## 1.3 Model Selection 

Before training and testing the model or building the model to predict the risk performance, we need to choose the best model that performs better than other models on the unseen data or generalises the data well without overfitting or underfitting so, we performed model selection using hyperparameter tuning of different classifiers. In model selection different classifiers are used and each classifier uses different set of parameters for training the model so, hyperparameter tuning is performed using a technique called Grid Search. Using grid search we build a model for each possible combination of hyper parameters used in different classifiers and choose an optimal set of hyper parameters for each classifier that gives best test score (in our case, the metric we are using for validating is ROC_AUC score so, we check for the best classifier with best set of hyper parameters that gives you best roc_auc_score).  Below is the best set of hyper parameters chosen for each classifier using grid search and the accuracy and ROC_AUC scores of classifiers after hyperparameter tuning. 
![image](https://user-images.githubusercontent.com/76246283/225264604-bed47001-0a53-45f1-a6d3-7668b4bc778f.png)
               

Figure 2: List of classifiers used, parameters, ROC AUC score and accuracy score 

Even though SVM performs better in terms of accuracy and ROC_AUC score, we chose XgBoost classifier to develop our model because XgBoost is the most efficient classifier which can minimise bias, errors and overfitting issues. It uses boosting technique which can learn from its own errors from every tree generated and tries to minimise those errors and loss while generating the new tree (number of trees generated depends on the value we give to hyper parameter ‘n_estimators’, in our case number of trees that will be generated to make a prediction is 100). The prediction is made based on the sum of the values generated in the leaf nodes of all the trees so, in this way it can minimise the errors, bias and can generalise more when compared to other models or classifiers.    

## 1.4 Model Development    

Model training and testing       

For model training and testing, we split the data as 80% for training and 20% for testing the data. We used XgBoost classifier with best set of hyper parameters resulted from the hyper parameter tuning technique we performed with 80% of the random data for training. We tested and validated the performance of the model or classifier using a ROC_AUC score and it gained 74% of accuracy.  

Feature Importance 

Machine learning models are often black boxes that makes their interpretation and explainability difficult [22]. In order to understand what are the main features that affect the output of the model, we need explainable machine learning techniques that can make their interpretation easier. One of these techniques is the SHAP plots which are used to explain how each feature affects the model prediction and gives an explainable analysis of the overall model and customised analysis of each prediction. SHAP plots are easily interpretable, explainable and are transparent. So, we used SHAP plots for plotting overall feature importance of the model and explain what are the major features that are contributing to most of the predications and how (which range of values of which feature are contributing to good or bad prediction).  

              
![image](https://user-images.githubusercontent.com/76246283/225260505-aae2a744-4e88-45fb-801b-af21f533b4fc.png)

 

Figure 3: SHAP values of attributes 

 

The side colours bar from high to low reflect the value of the feature, while the x-axis informs risk, with the positive side indicating no risk (Good) and the negative side indicating risk (Bad). The negative sign contributes to the prediction as ‘0’ or ‘Bad’ and the positive sign contributes to the prediction as ‘1’ or ‘Good’. For example, a higher value of ExternalRiskEstimate indicates that there is no risk (Good), while a lower value indicates that there is a risk (Bad). A higher value of "MSinceMostRecentInqexcl7days" indicates that there is no risk (Good), while the lower value indicates that there is a risk (Bad). The characteristics are ranked according to their importance in the data. In order to make this plot easily understandable, we provide explanations in our web application as shown below in figure 4. 

![image](https://user-images.githubusercontent.com/76246283/225265223-ca2b04d8-3420-4128-bd10-63b4e790dc5c.png)


Figure 4: Feature importance from web app 

Feature Explanation using customised SHAP plot 

Apart from plotting overall feature importance of the model, we explained how and what are the major financial factors or features affecting every individual prediction thus providing a unique or customised plot and explanation for each application or customer. This will help a banker make informed decisions to a customer like ‘The proportion of your revolving balances to total balances is too high (NetFractionRevolvingBurden)’ or ‘you recently opened a new account (MSinceOldestTradeOpen is low)’ and a customer understand what financial factors can be improved to reduce their risk percentage and get the loan approved. We used force plot and waterfall plot in SHAP to provide customised plot and explanation of the financial factors affecting their risk percentage. We also provided a detailed explanation to make the plots more understandable in our web application as shown below in figures 5 and 6. 

![image](https://user-images.githubusercontent.com/76246283/225265322-ee17fb0f-2cd7-4274-95fa-d6995f435c4a.png)
 

Figure 5: Web app screencap of feature explanation 

 
![image](https://user-images.githubusercontent.com/76246283/225265376-d23005e3-79f6-4d6f-a2f5-24c7b596b980.png)

Figure 6:Web app screencap of feature explanation 

## 1.5 Model validation 

The metric we chose for validating our model is ROC_AUC score. The ROC_AUC score helps us visualise how well our model is performing [23]. The ROC_AUC curve is only used for validating the binary classification models and our data also has a target variable which is binary (having only two classes ‘Good’ or ‘Bad’) so, we chose ROC_AUC score for model validation. The ROC_AUC score is a measure of the ability of a model to distinguish between classes and higher the score, the better the performance of the model at distinguishing between the positive and negative classes. This technique is very essential to determine and evaluate the performance of the model and in turn, it helps improve the accuracy and correctness of the model [24]. Our model gained ROC_AUC score of 74% and from the graph below we can clearly summarise that our model is performing well as it can distinguish between both the classes. 

![image](https://user-images.githubusercontent.com/76246283/225265470-8fe017c2-4713-4e1b-98ed-c2fc2cc301a7.png)

Figure 11: Area under ROC curve, showcasing model validation 

## 1.6 Designing web application by deploying our model 

We designed our web application using Stream lit and Ngrok. Stream lit is used in deploying our model and design an application with various features while Ngrok helps in providing a host that helps us to expose our web application on any private network. In deploying our model, we used ‘joblib’ that can save the model which is trained and can be loaded or used in stream lit to deploy our model. After deploying the model using joblib, using different functions in Stream lit, we provided different option to input the applications either as a single application or as batch of applications. Single application can be inputted using the sliders we provided (sliders can be adjusted as per the values of the financial factors in the application). Batch applications can be uploaded using a button that can help you browse the csv files and upload the file as seen in figure 7. 

 
![image](https://user-images.githubusercontent.com/76246283/225265674-bdced067-d610-4576-81e8-4ed6c3027ceb.png)

Figure 7: Web app screencap of input values 

For batch applications, when uploading a csv file, a user should make sure all the columns are in correct order and format and they can also check the correct order and format of the columns using ‘download the image’ button. For predicting the risk and risk percentage, a user can simply click on predict after adjusting sliders for single application or uploading the file and giving an index number for batch applications. 

 
![image](https://user-images.githubusercontent.com/76246283/225265717-ecd80066-7c95-42f3-8647-5b68edc4890a.png)

Figure 8: Predictive capability of web app 

We provided an option to see the financial factors or features contributing the prediction which are unique and customised for each customer/application which helps the users understand what are the financial factors they can improve to reduce their risk percentage and get their loan approved, the users can click on ‘feature explanation’. The option provides the user with the plots and detailed explanations of the plots as shown in the section ‘Feature Explanation using Customised plots’. 

We also provided an option to see the overall feature importance of the model to help the new users understand what are the major financial factors they can focus on to get their loan approved on overall or in general. The users can click on ‘Overall Feature Importance’ and the option provides the users with plot and detailed explanation as shown in the section ‘Feature Explanation’. 

We provided an additional option which allows the users to train the model with their own data which can help fine tune the existing model to improve its correctness. Users can click on ‘Train your own data and predict’ and upload the data, seen in figure 9. 

 
![image](https://user-images.githubusercontent.com/76246283/225265758-52fe28cd-8c3b-4045-bd17-e05cd96ccc56.png)

Figure 9: Web app screencap of model training feature 

The users are also provided with an option to see our model performance and how our model overperforms over other models by clicking on ‘Show performance or evaluation of models’ as seen in figure 10. Our web application is very feasible, interactive and is designed in a way that it can be easily used by a banker or a loaner. 

![image](https://user-images.githubusercontent.com/76246283/225265792-e43112a2-4bc5-47d8-b414-71d260dc106e.png)
 

Figure 10: Performance evaluators in web app 

 

 

# Impact and Significance of Results 

Machine learning models have not yet been able to accurately measure the risk of bank loans based purely on the financial activity of the customer [13], so the loan application process is yet to be automated. However, our XGBoost model is quite accurate, particularly in identifying high and low risk loan applications, which allows it to be deployed as a decision-making tool – in the form of a web-application - for bankers and credit assessors. Our application can provide its assessment instantly for each customer, which would allow for quicker loan approvals for customers deemed ‘low-risk’ by the model, reducing the cost to the financial institution of unnecessary checks for these applications. ‘High-risk’ applications identified by our model can also be flagged for bankers and credit assessors, helping them decrease the amount of ‘high-risk’ files that are approved, and ultimately reducing the default rate among borrowers. 

However, the most significant impact from our solution is the use of SHAP plots to show the most important positive and negative features for each borrower as the model assesses their application. This ensures any decision made by our model is as transparent and understandable as possible, allowing for the implementation of our model in the heavily regulated financial services industry. This also facilitates conversation between bankers and applicants, who will be able to gain an understanding of why they have been placed in a higher or lower risk category by the model, and what aspects of their financial profile can be improved.  

Our solution ultimately allows for the convenience and accuracy of our XGBoost model to be utilized while also leveraging the transparency and explainability requirements in the industry to improve banker-customer relationships and foster a greater understanding in prospective borrowers of their financial profile. The novelty of our solution therefore revolves around the below: 

The extension of the XGBoost classifier model to output probability of default rather than purely classifying ‘good’ or ‘bad’ debt as seen in the literature. 

 

The deployment of our model as a web-application for bankers, given that the accuracy of machine learning models in assessing credit risk is not high enough to justify the complete automation of credit assessment. [3] 

 

The use of SHAP plots to show the most important features for each new data row, giving our model instance-level explainability, allowing each borrower to understand the factors influencing each credit assessment completed by our model. 
