#!/usr/bin/env python
# coding: utf-8

# # _____________________ HR Analytics Project _____________________ 

# ### Introduction

# ### 
# Attrition/Churn/Turnover Rate refers to the rate at which employees leave a company. Employee turnover is a costly problem for organizations. The cost of replacing an employee can be quite large, and a study found that companies typically pay about one-fifth of an employee's salary to replace them. The cost can significantly increase if executives or highest-paid employees are to be replaced. The cost of replacing employees for most employers remains significant. This is due to the amount of time spent to interview and find a replacement, sign-on bonuses, and the loss of productivity for several months while the new employee gets accustomed to the new role.

# ![openhrimage](https://www.bizarmour.co.za/wp-content/uploads/2020/04/bizarmour-functions-of-hr.png)

# ## Purpose:

# ##
# In this project we are going to develop a model that could predict which employees are more likely to quit. We are going to explore the data and then create a model to predict how likely the employee quit the job.

# ### Project Structure

# Data Exploration
# 
# 1.Data Preprocessing
# 
# 2.Exploratory Data Analysis (EDA) - Pandas/Numpy
# 
# 3.Data Modification
# 
# 4.Data Visualization - Seaborn 
# 
# 5.Feature Selection
# 
# 6.Model Selection and Training
# 
# 7.Model Evaluation using ML Algorithms for Attrition Prediction
# 
# 8.Classification Report and Accuracy Explanation
# 
# 9.Conclusion and Sources.

# <!-- 
# SR No.	  Variable Name	             Variable Definition
# 1          EmpId                      Ids of employee
# 2	         Age	                Age of the employee
# 3          Age-Group              Age groups(18-25,26-35,36-45,45-55,55+)
# 4	      Attrition	            Employee who stayed -0 , Employee who leave- 1
# 5	    BusinessTravel	       Traveling frquency
# 6	      DailyRate	          Daily Rate of Employee
# 7	     Department	           Departments existing in the company
# 8	  DistanceFromHome	        Distance from home and workplace
# 7	    Education       	  1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor
# 8	   EmployeeCount	         The count of employee
# 9	   EmployeeNumber	     The number assigned to employee
# 10	  EducationField	      Life Sciences,Other, Medical, Marketing, Technical Degree, Human Resources
# 11	  EnvironmentSatisfaction	  1: Low, 2 :Medium, 3 :High, 4 :Very High
# 12	     Gender	                 Female or Male
# 13	    HourlyRate	          Hourly Rate of Employee
# 14	   JobInvolvement	     1: Low, 2 :Medium, 3 :High, 4 :Very High
# 15	     JobLevel	          Job levels in the company
# 16	      JobRole	         Job Role of the employee
# 17	   JobSatisfaction     	1: Low, 2 :Medium, 3 :High, 4 :Very High
# 18	    MaritalStatus	     Maritial status of the employee
# 19	   MonthlyIncome	     Monthly income of the employee
# 20	    MonthlyRate	         Monthly rate of the employee
# 21	   NumCompaniesWorked	  No.of Companies before the current one
# 22	     Over18	              Employee over 18 years or not
# 23	    OverTime	         Worked over time or not
# 24   PercentSalaryHike	     Percentage of Salary increase between %11-%25
# 25	PerformanceRating	     1 :Low, 2 :Good, 3 :Excellent, 4 :Outstanding
# 26	RelationshipSatisfaction	1: Low, 2 :Medium, 3 :High, 4 :Very High
# 27	StandardHours	         standard work hour for each employee: 80 Hours
# 28	StockOptionLevel	     Indicate the stock level of employee
# 29	TotalWorkingYears	     Employee`s total working years
# 30	TrainingTimesLastYear	  Employee`s training period in the last year
# 31	WorkLifeBalance	         1 :Bad, 2 :Good, 3 :Better, 4 :Best
# 32	YearsAtCompany	         Employess`s total working year at the company
# 33	YearsInCurrentRole	     Employee`s current position at the company in years
# 34	YearsSinceLastPromotion	  Last promotion year of the employee
# 35	YearsWithCurrManager	 Employee working with the current manager in years
#  -->

# ### Variables Present in the Dataset:
# 
# In order to understand our Dataset we have to understand variables present in the dataset properly. In this dataset we are having different parameters including personal and professional information of employee working in the company.
# 
# 
# | Serial No. | Variable Name | Variable Definition |
# | :---: | :---: | :--- |
# | 1 | Age | Age of the employee |
# | 2 | Attrition | Employee who stayed -0 , Employee who leave- 1|
# | 3 | BusinessTravel|Traveling frquency|
# | 4 | DailyRate | Daily Rate of Employee|
# | 5 | Department |Departments existing in the company|
# | 6 | DistanceFromHome| Distance from home and workplace|
# | 7 | Education | 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor|
# | 8 | EmployeeCount | The count of employee|
# | 9 | EmployeeNumber| The number assigned to employee|
# | 10| EducationField | Life Sciences,Other, Medical, Marketing, Technical Degree, Human Resources|
# | 11| EnvironmentSatisfaction |1: Low, 2 :Medium, 3 :High, 4 :Very High|
# | 12| Gender |Female or Male|
# | 13| HourlyRate |Hourly Rate of Employee|
# | 14| JobInvolvement |1: Low, 2 :Medium, 3 :High, 4 :Very High|
# | 15| JobLevel |Job levels in the company|
# | 16| JobRole |Job Role of the employee|
# | 17| JobSatisfaction |1: Low, 2 :Medium, 3 :High, 4 :Very High|
# | 18| MaritalStatus|Maritial status of the employee|
# | 19| MonthlyIncome|Monthly income of the employee|
# | 20| MonthlyRate|Monthly rate of the employee|
# | 21| NumCompaniesWorked|No.of Companies before the current one|
# | 22| Over18 |Employee over 18 years or not|
# | 23| OverTime |Worked over time or not|
# | 24| PercentSalaryHike |Percentage of Salary increase between %11-%25|
# | 25| PerformanceRating |1 :Low, 2 :Good, 3 :Excellent, 4 :Outstanding|
# | 26| RelationshipSatisfaction |1: Low, 2 :Medium, 3 :High, 4 :Very High|
# | 27| StandardHours | standard work hour for each employee: 80 Hours
# | 28| StockOptionLevel |Indicate the stock level of employee|
# | 29| TotalWorkingYears |Employee`s total working years|
# | 30| TrainingTimesLastYear |Employee`s training period in the last year|
# | 31| WorkLifeBalance|1 :Bad, 2 :Good, 3 :Better, 4 :Best|
# | 32| YearsAtCompany|Employess`s total working year at the company|
# | 33| YearsInCurrentRole|Employee`s current position at the company in years|
# | 34| YearsSinceLastPromotion|Last promotion year of the employee|
# | 35| YearsWithCurrManager| Employee working with the current manager in years|
# 

# ## Summary of the Project:

# The dataset contains 1,480 observations with 38 features.
# 
# The target variable is "Attrition", which is a binary variable indicating whether an employee has left the company.
# 
# The goal of the project is to develop a model to predict employee attrition and identify key factors contributing to attrition.
# 
# The project will involve exploratory data analysis, feature engineering, and building and evaluating machine learning models.
# 
# Potential benefits of the project include improved employee retention strategies and cost savings for companies.

# ### Importing Libraries

# In[1]:


import os ,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# pd.set_option('display.max_rows',15000)
# pd.set_option('display.max_columns',500)
# pd.set_option('display.width',15000)         #to display all rows & columns


# ### Loading The Dataset

# In[3]:


df=pd.read_csv('HR_Analytics.csv')


# In[4]:


df.head()


# #### Shape of the dataset

# In[5]:


df.shape


# ##### From above we can have overlook on our dataset. We can see that 1,480 observations with 38 features are present.

# ### Basic information

# In[6]:


df.info()


# ####
# From above information we can say that we have total 12-object,25-int and 1-float columns,in which 'YearsWithCurrManager' has some null values.

# #### Summarry

# In[7]:


df.describe()


# From above we can find out
# 
# 
# *   Min. age is 18 and max. age is 60 years.
# 
# *   Distance from home to office is min 1 KM and max. is 29 KM 
# 
# *   Similarly we can check for all the **minimum, maximum, mean, standard deviation, 25% , 50%, 75% values** of the all numerical columns.
# *   It will help us to understand our data better.
# 
# 
# 
# 
# 

# #### Summary for Categorical columns

# Similarly, we can use df.describe(include = 'object') for categorical columns.

# In[8]:


df.describe(include='object')


# *   **'count'** will give us the no of unique rows. i.e. 1480
# *   **'unique'** is the unique values present in the each column.
# 
# *   **'top'** is the unique value which appeared more frequnt and **'freq'** is the no of times 'top' value appeared.

# #### No. of unique values in dataset

# In[9]:


df.nunique()


# In[10]:


#to explore unique values in the categorical columns
for i in df.columns:
  if df[i].dtype == object:
    print(str(i) + ":" + str(df[i].unique()))
    print(df[i].value_counts())
    print("---------------------------------------------------------------------")


# #### Dropping the not-required column

# We notice that 'EmployeeCount', 'Over18' and 'StandardHours' have only one unique values and 'EmployeeNumber' has 1470 unique values.Also 'AgeGroup','EMPID','SalarySlab','Education','JobLevel' these features aren't useful for us, so we are going to drop those columns.

# In[11]:


df=df.drop(columns=['Over18','AgeGroup','EmpID','StandardHours','SalarySlab','Education','JobLevel','EmployeeCount','EmployeeNumber'],axis=1)


# In[12]:


df.head()


# ## Null Vallues Imputations

# ##### 
# Missing and Duplicate Records can cause problem and bias in our machine learning module. We will check for the any missing values and duplicate records present in our dataset.If present we can drop or impute according to requirements.

# In[13]:


df['BusinessTravel'].value_counts()


# In[14]:


df['BusinessTravel'].replace({'TravelRarely':'Travel_Rarely'},inplace=True)


# In[15]:


df.isnull().sum()/len(df)*100


# In[16]:


df['YearsWithCurrManager'].unique()


# In[17]:


df['YearsWithCurrManager']=df['YearsWithCurrManager'].fillna(df['YearsWithCurrManager'].mode()[0])


# In[18]:


df.isnull().sum()


# #### Dropping Duplicate Values

# In[19]:


df.duplicated().sum()


# In[20]:


df=df.drop_duplicates()


# In[21]:


df.duplicated().sum()


# #### Outliers

# In[22]:


outlier = df[['Age','DistanceFromHome','EnvironmentSatisfaction','JobSatisfaction','StockOptionLevel','YearsInCurrentRole','YearsSinceLastPromotion','TotalWorkingYears','YearsWithCurrManager']]


# In[23]:


plt.figure(figsize=(30,10))#sets the figure size
sns.boxplot(data=outlier, palette='rainbow')


# In[24]:


out = df[['MonthlyIncome','DailyRate']]
plt.figure(figsize=(25,7))                      
sns.boxplot(data=out, palette='rainbow') 


# From above boxplot we can find out:
# 
# * We can see the five point summary of the each attributes i.e. minimum, first quartile [Q1], median, third quartile [Q3] and maximum.
# 
# * We can see, there are outliers present 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'TotalWorkingYears', 'YearsWithCurrManager' columns. But it will not affect our analysis as it is very less.
# 
# * It suggest that few employees are working in current role and with the same manager for more than 15 years also few employees have not got any promotion from 8 years. It can be the reason for increase in rate of attrition.
# 
# * In MonthlyIncome we see that there are few employees who are getting more than maximum salary offered. We can also see that avg salary range is 5000. There are no outliers for DailyRate.
# 
# We can ignore this outliers as it will not affect our analysis and prediction.

# ### Checking the distribution of data

# In[25]:


def distplot(col):
    sns.distplot(df[col])
    plt.show()
    
for i in list(df.select_dtypes(exclude=['object','category']).columns)[0:]:
    distplot(i)


# In[26]:


df.hist(bins=25,figsize=(20,30),color='blue')
plt.show()


# In[27]:


sns.pairplot(data=df,hue='Attrition')


# A few observations can be made based on above plots for numerical features:
# 
# * Many histograms & distribution plots are tail-heavy; indeed several distributions are right-skewed (e.g. MonthlyIncome DistanceFromHome, JobLevel,YearsAtCompany).
# 
# * 'Education','EnvironmentSatisfaction','JobInvolvement', 'JobRole', 'JobSatisfaction', 'RelationshipSatisfaction' are left-slewed which suggest that most of the empolyees are well educated and satisfied with their job role and work environment.
# 
# * Age distribution is a slightly right-skewed normal distribution with the bulk of the staff between 25 and 45 years old.
# 
# * 'MonthlyIncome', 'JobLevel', 'NumCompaniesWorked','PercentSalaryHike', 'StockOptionLevel','TotalWorkingYears', 'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager' are positivley-skewed distribution wich suggest that most of the compensation related attributes values are relatively close to the lower bound.

# ### Gender Wise Exploration of Categorical Features
# We will use Seaborn and Matplotlib liabries for the visualization. Lets use subplots and countplot to represnt the visuals.

# In[28]:


fig = plt.figure(figsize=(42,30))

#  subplot #1
plt.subplot(2,3,1)
plt.title('Attrition', fontsize=35)
a_x=sns.countplot(data = df, y = 'Attrition', hue= "Gender", palette='rainbow_r', orient="h")
a_x.legend(fontsize='large')

#  subplot #2
plt.subplot(2,3,2)
plt.title('BusinessTravel', fontsize=35)
b_x=sns.countplot(data = df, y = 'BusinessTravel', hue= 'Gender', palette='rainbow', orient="h")
b_x.legend(fontsize='large')

#  subplot #3
plt.subplot(2,3,3)
plt.title('Employees per department', fontsize=35)
c_x=sns.countplot(data = df, y = 'Department', hue = 'Gender', palette='rainbow', orient="h")
c_x.legend(fontsize='large')

#  subplot #4
plt.subplot(2,3,4)
plt.title('Education Field', fontsize=35)
d_x=sns.countplot(data = df, y = 'EducationField', hue = 'Gender', palette='rainbow', orient="h")
d_x.legend(fontsize='large')

#  subplot #5
plt.subplot(2,3,5)
plt.title('JobRole', fontsize=35)
e_x=sns.countplot(data = df, y = 'JobRole', hue = 'Gender', palette='rainbow', orient="h")
e_x.legend(fontsize='large')

#  subplot #6
plt.subplot(2,3,6)
plt.title('MaritalStatus', fontsize=35)
f_x=sns.countplot(data = df, y = 'MaritalStatus', hue = 'Gender', palette='rainbow', orient="h")
f_x.legend(fontsize='large')

# Adjust plot spacing
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.show()


# From above countplot we can observe:
# 
# * We can see Gender-wise countplot data for all the categorical columns.
# 
# * We can get Idea various categorical sub-values and their frequency with reference to Gender of employees.
# 
# * Attrition rate is greater in Male than Female employees.
# 
# We can also see that males are leading in every countplot which suggest that female employees are struggling or lagging behind somewhat in the corporate world.

# ## Gender Wise Exploration of Numerical Features

# In[29]:


fig, axes = plt.subplots(4, 2, figsize=(20, 20))

sns.violinplot(ax=axes[0, 0], data=df, x='Age', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[0, 1], data=df, x='DistanceFromHome', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[1, 0], data=df, x='EnvironmentSatisfaction', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[1, 1], data=df, x='JobSatisfaction', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[2, 0], data=df, x='MonthlyIncome', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[2, 1], data=df, x='YearsInCurrentRole', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[3, 0], data=df, x='YearsSinceLastPromotion', y='Gender', palette='rainbow')
sns.violinplot(ax=axes[3, 1], data=df, x='YearsWithCurrManager', y='Gender', palette='rainbow')

plt.tight_layout()
plt.show()


# From above plots we can easily see **the distribution of various numerical attrinbutes within the ranges**. 
# 
# Volinplot is easy to interpret and to get insights from numerical columns. It can also detect any outliers present in Data Set.
# 
# We can observe:
# 
# * For all attributes for both Genders distribution is mostly even. 
# 
# * Avg Age of emplyees lies between 30-40 years. Distance from home for the most of the employees lies within 2-10 KMs.
# 
# * We can see the distribution of monthly income lies between 2500-5000
# 
# * Similary we can interpret the plot for other attributes present.

# ### Correlation

# In[30]:


plt.figure(figsize=(20,15),dpi=100)
sns.heatmap(df.corr(),annot=True,cmap='rainbow')


# Few **key Learnings** from the correlation vizualisation heatmap above:
# 
# 
#  Age has a 68% correlation with total working years, which means that as we continue to work at an organization, we tend to get older, which is a logical conclusion.
# 
# 
#  Monthly income has a 77% correlation with total working years. It indicates **as we work longer, we are likely to receive higher salaries.*
# 
# 

# # Encoding

# In[31]:


#lable encoder
df['Attrition']=df['Attrition'].astype('category')
df['Attrition']=df['Attrition'].cat.codes

df['OverTime']=df['OverTime'].astype('category')
df['OverTime']=df['OverTime'].cat.codes

df['Gender']=df['Gender'].astype('category')
df['Gender']=df['Gender'].cat.codes


# In[32]:


#one hot encoding
df = pd.get_dummies(df, columns=['Department','BusinessTravel','EducationField','JobRole','MaritalStatus'])


# In[33]:


df.head()


# In[34]:


df=df.drop(columns=['Department_Sales','BusinessTravel_Travel_Rarely','EducationField_Other','JobRole_Sales Representative','MaritalStatus_Single'],axis=1)


# In[35]:


df.head()


# In[36]:


x=df.drop(['Attrition'],axis=1)
y=df['Attrition']


# In[37]:


x.head()


# In[38]:


y.head()


# # Scaling

# In[39]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_sc=sc.fit_transform(x)
x_sc


# # Model Building

# ### Train-Test Split

# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_sc,y,test_size=0.25,random_state=42)


# ## Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[43]:


print(classification_report(y_test,y_pred_test))


# In[44]:


print(confusion_matrix(y_test,y_pred_test))


# In[45]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test))


# * As we can see the accuracy is 87% for this model using random forest algorithm. The model correctly identified 87% of the employees that left the company.
# * But also looking at training & testing accuracy we can say that this model is giving some variance,indicating the model has overfitting.
# * To get rid of this we can use further more techniques & models.

# ### Let’s see what the model thinks are the important features/variables.

# In[46]:


rf.feature_importances_


# In[47]:


imp=pd.DataFrame(index=x.columns,data=rf.feature_importances_,columns=['Feature Importance'])
imp=imp.sort_values('Feature Importance',ascending=False)
imp


# In[48]:


imp.plot(kind='bar',color='blue',figsize=(10,6))
plt.show()


#  Monthly income appears to be the most important feature followed by the persons age, daily rate, and Monthly rate. Seeing this result makes me want to see if people with a higher income are less likely to leave than someone with a lower income.
# 

# ## Post-Prunning

# In[49]:


rf2=RandomForestClassifier(n_estimators=1000,max_depth=6,oob_score=True)
rf2.fit(x_train,y_train)
y_pred_train_rf2=rf2.predict(x_train)
y_pred_test_rf2=rf2.predict(x_test)


# In[50]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_rf2))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_rf2))


# * As we can see the accuracy is 86% for this model using **random forest algorithm post prunning**. The model correctly identified 86% of the employees that left the company.
# * Also their is no variance in model.

# ## Logistic Regression

# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_train_lr=lr.predict(x_train)
y_pred_test_lr=lr.predict(x_test)


# In[53]:


print(classification_report(y_test,y_pred_test_lr))


# In[54]:


print(confusion_matrix(y_test,y_pred_test_lr))


# In[55]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_lr))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_lr))


# As we can see the accuracy is 89% for this model using **Logistic Regression** algorithm. The model correctly identified 89% of the employees that left the company.

# ## PCA

# In[56]:


from sklearn.decomposition import PCA


# In[57]:


pca=PCA(0.95)
x_train_1=pca.fit(x_train)
x_test_1=pca.fit(x_test)


# In[58]:


explained_var= pca.explained_variance_ratio_   


# In[59]:


explained_var


# In[60]:


pca=PCA(n_components=15)
x_train_2 = pca.fit_transform(x_train)
x_test_2 = pca.fit_transform(x_test)


# ### PCA-logit

# In[61]:


logit=LogisticRegression()
logit.fit(x_train_2,y_train)
y_pred_train_pca=logit.predict(x_train_2)
y_pred_test_pca=logit.predict(x_test_2)


# In[62]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_pca))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_pca))


# As we can see the accuracy is 82% for this model using **PCA-Logit** algorithm. The model correctly identified 82% of the employees that left the company.

# ### PCA-Randoom Forest

# In[63]:


rf1=RandomForestClassifier(n_estimators=10, criterion='entropy')
rf1.fit(x_train_2,y_train)
y_pred_train_pca1=rf1.predict(x_train_2)
y_pred_test_pca1=rf1.predict(x_test_2)


# In[64]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_pca1))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_pca1))


# * As we can see the accuracy is 85% for this model using **PCA-Random Forest** algorithm. The model correctly identified 83% of the employees that left the company.
# * This model also gives some variance.

# ## Decision Tree

# In[65]:


from sklearn.tree import DecisionTreeClassifier
dt1=DecisionTreeClassifier(criterion='gini')
dt1.fit(x_train,y_train)
y_pred_train_dt1=dt1.predict(x_train)
y_pred_test_dt1=dt1.predict(x_test)


# In[66]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_dt1))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_dt1))


# In[67]:


dt2=DecisionTreeClassifier(criterion='entropy')
dt2.fit(x_train,y_train)
y_pred_train_dt2=dt2.predict(x_train)
y_pred_test_dt2=dt2.predict(x_test)


# In[68]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_dt2))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_dt2))


# * As we can see the accuracy is 77-78% for  model using **Decision Tree-gini** and **Decision Tree-entropy** algorithm. The model correctly identified 77-78% of the employees that left the company.
# * both the model gives high variance.

# In[69]:


dt1.feature_importances_


# ### Post-Prunning Decision Tree

# In[70]:


dt3=DecisionTreeClassifier(criterion='gini',max_depth=3)
dt3.fit(x_train,y_train)
y_pred_train_dt3=dt3.predict(x_train)
y_pred_test_dt3=dt3.predict(x_test)


# In[71]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_dt3))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_dt3))


# In[72]:


dt4=DecisionTreeClassifier(criterion='entropy',max_depth=3)
dt4.fit(x_train,y_train)
y_pred_train_dt4=dt4.predict(x_train)
y_pred_test_dt4=dt4.predict(x_test)


# In[73]:


print('Training Accuracy',accuracy_score(y_train,y_pred_train_dt4))
print('**********************************')
print('Testing Accuracy',accuracy_score(y_test,y_pred_test_dt4))


# * As we can see the accuracy is 87% for this model using **DecisionTree-gini post prunning** & **DecisionTree-entropy post prunning**. The model correctly identified 86% of the employees that left the company.
# * Also their is no variance in model.

# ### Models Summary

# - We developed few models :
#      - 1-Random Forest  [achieved an accuracy of **87%(With variance)**]
#      - 2-Random Forest-post prunning  [achieved an accuracy of **86%**]
#      - 3-Logistic Regression [achieved an accuracy of **89%**]  
#      - 4-PCA-Logit  [achieved an accuracy of **82%**]   
#      - 5-PCA-Random Forest [achieved an accuracy of **85%(With variance)**]
#      - 6-Decision Tree  [achieved an accuracy of **77-78%(With variance)**]
#      - 7-Decision Tree-post prunning  [achieved an accuracy of **87%**]
#         
# -   These models indicating that **'Logistic regression'** & **'Random Forest post prunning' and 'Decision Tree post prunning'**  model has good predictive power.

# ---
# ### Conclusion: 
# 
# Based on our analysis of the HR Analytics Employee Attrition & Performance dataset, we can draw the following conclusions:
# 
# - The developed model achieved an accuracy of 87%, indicating that it has good predictive power.
# 
# 
# - Factors such as job level, monthly income, and age were found to be important predictors of employee attrition.
# 
# 
# - The company can use the model to identify employees who are at high risk of leaving and take proactive measures to retain them.
# 
# 
# - Possible strategies for improving employee retention include offering competitive compensation and benefits packages, providing opportunities for career growth and development, and providing a positive work environment.
# 
# Overall, our analysis highlights the importance of leveraging HR analytics to gain insights into workforce trends and patterns, and ultimately to make strategic decisions that can improve employee retention and reduce the costs associated with employee turnover.
# 

# # Thank You.....

# In[ ]:




