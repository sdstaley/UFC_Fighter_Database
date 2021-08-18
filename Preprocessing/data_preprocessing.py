#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[9]:


#import dataset
df = pd.read_csv ("C://Users//ZAFAR IQBAL//PyProg//datamining//ufcdata.csv")

#df = pd.read_csv ('ufcdata.csv')


# In[10]:


#drop nominal attributes
df=df.drop(['R_fighter','B_fighter', 'Referee','date','location' ], axis=1)
#drop rows having atleast 45 nan values
df=df.dropna(thresh=100)
print(len(df))


# In[11]:


# pd.set_option('display.max_rows', None)#toshow all the rows
# df.isnull().sum()


# In[12]:


#fill in missing values using linear regression

linreg = LinearRegression()
#choose four attributes with highest correlation to R_age 
data = df[['R_losses','R_wins','R_total_rounds_fought','R_win_by_KO/TKO','R_age']]
#Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.
x_train = data[data['R_age'].notnull()].drop(columns='R_age')
y_train = data[data['R_age'].notnull()]['R_age']
x_test = data[data['R_age'].isnull()].drop(columns='R_age')
y_test = data[data['R_age'].isnull()]['R_age']
#Step-2: Train the machine learning algorithm
linreg.fit(x_train, y_train)
#Step-3: Predict the missing values in the attribute of the test data.
predicted = linreg.predict(x_test)
#Step-4: Let’s obtain the complete dataset by combining with the target attribute.
df.R_age[df.R_age.isnull()] = predicted

################B_age###########
#choose four attributes with highest correlation to R_age
data = df[['B_losses','B_wins','B_total_rounds_fought','B_win_by_KO/TKO','B_age']]
#Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.
x_train = data[data['B_age'].notnull()].drop(columns='B_age')
y_train = data[data['B_age'].notnull()]['B_age']
x_test = data[data['B_age'].isnull()].drop(columns='B_age')
y_test = data[data['B_age'].isnull()]['B_age']
#Step-2: Train the machine learning algorithm
linreg.fit(x_train, y_train)
#Step-3: Predict the missing values in the attribute of the test data.
predicted = linreg.predict(x_test)
#Step-4: Let’s obtain the complete dataset by combining with the target attribute.
df.B_age[df.B_age.isnull()] = predicted

################################


# In[13]:


#replace nan with most frequent item in the column
df.R_Stance.describe()
mode=df.R_Stance.mode()
df.R_Stance.fillna(mode[0], inplace=True)


# In[14]:


#replace nan with most frequent item in the column
mode1=df.B_Stance.mode()
df.B_Stance.fillna(mode1[0], inplace=True)
df.B_Stance.describe()


# In[15]:


df.isna().sum()


# In[82]:


df.R_Reach_cms.describe()
df.B_Reach_cms.describe()


# In[16]:


#replace nan for both the columns with mean value
#as B_height_cms has one missing value,remove the entire row

df.R_Reach_cms.fillna(df.R_Reach_cms.mean(),inplace=True)
df.B_Reach_cms.fillna(df.B_Reach_cms.mean(),inplace=True)
df.dropna(inplace=True)
df.isna().sum().sum()


# In[17]:


df_B=df[['B_avg_BODY_att','B_avg_BODY_landed','B_avg_CLINCH_att','B_avg_CLINCH_landed','B_avg_DISTANCE_att','B_avg_DISTANCE_landed','B_avg_GROUND_att','B_avg_GROUND_landed','B_avg_HEAD_att','B_avg_HEAD_landed','B_avg_KD','B_avg_LEG_att','B_avg_LEG_landed','B_avg_PASS','B_avg_REV','B_avg_SIG_STR_att','B_avg_SIG_STR_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_att','B_avg_TD_landed','B_avg_TD_pct','B_avg_TOTAL_STR_att','B_avg_TOTAL_STR_landed','B_avg_opp_BODY_att','B_avg_opp_BODY_landed','B_avg_opp_CLINCH_att','B_avg_opp_CLINCH_landed','B_avg_opp_DISTANCE_att','B_avg_opp_DISTANCE_landed','B_avg_opp_GROUND_att','B_avg_opp_GROUND_landed','B_avg_opp_HEAD_att','B_avg_opp_HEAD_landed','B_avg_opp_KD','B_avg_opp_LEG_att','B_avg_opp_LEG_landed','B_avg_opp_PASS','B_avg_opp_REV','B_avg_opp_SIG_STR_att','B_avg_opp_SIG_STR_landed','B_avg_opp_SIG_STR_pct','B_avg_opp_SUB_ATT','B_avg_opp_TD_att','B_avg_opp_TD_landed','B_avg_opp_TD_pct','B_avg_opp_TOTAL_STR_att','B_avg_opp_TOTAL_STR_landed','B_total_rounds_fought','B_total_time_fought(seconds)','B_Height_cms','B_Reach_cms','B_Weight_lbs','B_age']]
#'B_avg_CLINCH_landed','B_avg_DISTANCE_att','B_avg_DISTANCE_landed','B_avg_GROUND_att','B_avg_GROUND_landed','B_avg_HEAD_att','B_avg_HEAD_landed','B_avg_KD','B_avg_LEG_att','B_avg_LEG_landed','B_avg_PASS','B_avg_REV','B_avg_SIG_STR_att','B_avg_SIG_STR_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_att','B_avg_TD_landed','B_avg_TD_pct','B_avg_TOTAL_STR_att','B_avg_TOTAL_STR_landed','B_avg_opp_BODY_att','B_avg_opp_BODY_landed','B_avg_opp_CLINCH_att','B_avg_opp_CLINCH_landed','B_avg_opp_DISTANCE_att','B_avg_opp_DISTANCE_landed','B_avg_opp_GROUND_att','B_avg_opp_GROUND_landed','B_avg_opp_HEAD_att','B_avg_opp_HEAD_landed','B_avg_opp_KD','B_avg_opp_LEG_att','B_avg_opp_LEG_landed','B_avg_opp_PASS','B_avg_opp_REV','B_avg_opp_SIG_STR_att','B_avg_opp_SIG_STR_landed','B_avg_opp_SIG_STR_pct	B_avg_opp_SUB_ATT','B_avg_opp_TD_att','B_avg_opp_TD_landed','B_avg_opp_TD_pct','B_avg_opp_TOTAL_STR_att','B_avg_opp_TOTAL_STR_landed','B_total_rounds_fought','B_total_time_fought(seconds)'
len(df_B)


# In[18]:


#drop attributes with correlation value greater than 0.80
corr_matrix=df_B.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
#len(to_drop)  #24 dropped
df_B.drop(df[to_drop], axis=1, inplace=True)
df_B.describe()


# In[20]:


#similar process for R
df_R=df[['R_avg_BODY_att','R_avg_BODY_landed','R_avg_CLINCH_att','R_avg_CLINCH_landed','R_avg_DISTANCE_att','R_avg_DISTANCE_landed','R_avg_GROUND_att','R_avg_GROUND_landed','R_avg_HEAD_att','R_avg_HEAD_landed','R_avg_KD','R_avg_LEG_att','R_avg_LEG_landed','R_avg_PASS','R_avg_REV','R_avg_SIG_STR_att','R_avg_SIG_STR_landed','R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_att','R_avg_TD_landed','R_avg_TD_pct','R_avg_TOTAL_STR_att','R_avg_TOTAL_STR_landed','R_avg_opp_BODY_att','R_avg_opp_BODY_landed','R_avg_opp_CLINCH_att','R_avg_opp_CLINCH_landed','R_avg_opp_DISTANCE_att','R_avg_opp_DISTANCE_landed','R_avg_opp_GROUND_att','R_avg_opp_GROUND_landed','R_avg_opp_HEAD_att','R_avg_opp_HEAD_landed','R_avg_opp_KD','R_avg_opp_LEG_att','R_avg_opp_LEG_landed','R_avg_opp_PASS','R_avg_opp_REV','R_avg_opp_SIG_STR_att','R_avg_opp_SIG_STR_landed','R_avg_opp_SIG_STR_pct','R_avg_opp_SUB_ATT','R_avg_opp_TD_att','R_avg_opp_TD_landed','R_avg_opp_TD_pct','R_avg_opp_TOTAL_STR_att','R_avg_opp_TOTAL_STR_landed','R_total_rounds_fought','R_total_time_fought(seconds)','R_Height_cms','R_Reach_cms','R_Weight_lbs','R_age']]
df_R.describe()
#drop attributes with correlation value greater than 0.80
corr_matrix2=df_R.corr().abs()
# Select upper triangle of correlation matrix
upper2 = corr_matrix2.where(np.triu(np.ones(corr_matrix2.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop2 = [column for column in upper2.columns if any(upper2[column] > 0.80)]
#len(to_drop2)  #24 dropped
df_R.drop(df[to_drop2], axis=1, inplace=True)


# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Standardizing the features
df_B = StandardScaler().fit_transform(df_B)


# In[23]:


pca = PCA(.85)
principalComponents = pca.fit_transform(df_B)
variance=pca.explained_variance_ratio_
variance   #10 PCA'S have 99% of informaiton


# In[24]:


#pcs for B
index = np.arange(len(variance))
plt.bar(index, variance)
plt.xlabel('Principal components')
plt.ylabel('variance')
plt.show()


# In[25]:


df_R = StandardScaler().fit_transform(df_R)
pca2 = PCA(.85)
principalComponents2 = pca2.fit_transform(df_R)
variance2=pca2.explained_variance_ratio_
variance2   #10 PCA'S have 99% of informaiton


# In[26]:


#variance for R
index2 = np.arange(len(variance2))
plt.bar(index2, variance2)
plt.xlabel('Principal components')
plt.ylabel('variance')
plt.show()


# In[28]:


pca3 = PCA(n_components=17)
principalComponents3 = pca3.fit_transform(df_B)
df_B = pd.DataFrame(data = principalComponents3
             , columns = ['pcB1', 'pcB2', 'pcB3','pcB4', 'pcB5', 'pcB6','pcB7', 'pcB8', 'pcB9','pcB10','pcB11', 'pcB12', 'pcB13','pcB14', 'pcB15', 'pcB16','pcB17'])

pca4 = PCA(n_components=16)
principalComponents4 = pca4.fit_transform(df_R)
df_R = pd.DataFrame(data = principalComponents4
             , columns = ['pcR1', 'pcR2', 'pcR3','pcR4', 'pcR5', 'pcR6','pcR7', 'pcR8', 'pcR9','pcR10','pcR11', 'pcR12', 'pcR13','pcR14', 'pcR15', 'pcR16'])


# In[29]:


#COMBINE FEATURES AND TARGET INTO ONE DATAFRAME
dd=df.drop(['B_avg_BODY_att','B_avg_BODY_landed','B_avg_CLINCH_att','B_avg_CLINCH_landed','B_avg_DISTANCE_att','B_avg_DISTANCE_landed','B_avg_GROUND_att','B_avg_GROUND_landed','B_avg_HEAD_att','B_avg_HEAD_landed','B_avg_KD','B_avg_LEG_att','B_avg_LEG_landed','B_avg_PASS','B_avg_REV','B_avg_SIG_STR_att','B_avg_SIG_STR_landed','B_avg_SIG_STR_pct','B_avg_SUB_ATT','B_avg_TD_att','B_avg_TD_landed','B_avg_TD_pct','B_avg_TOTAL_STR_att','B_avg_TOTAL_STR_landed','B_avg_opp_BODY_att','B_avg_opp_BODY_landed','B_avg_opp_CLINCH_att','B_avg_opp_CLINCH_landed','B_avg_opp_DISTANCE_att','B_avg_opp_DISTANCE_landed','B_avg_opp_GROUND_att','B_avg_opp_GROUND_landed','B_avg_opp_HEAD_att','B_avg_opp_HEAD_landed','B_avg_opp_KD','B_avg_opp_LEG_att','B_avg_opp_LEG_landed','B_avg_opp_PASS','B_avg_opp_REV','B_avg_opp_SIG_STR_att','B_avg_opp_SIG_STR_landed','B_avg_opp_SIG_STR_pct','B_avg_opp_SUB_ATT','B_avg_opp_TD_att','B_avg_opp_TD_landed','B_avg_opp_TD_pct','B_avg_opp_TOTAL_STR_att','B_avg_opp_TOTAL_STR_landed','B_total_rounds_fought','B_total_time_fought(seconds)','R_avg_BODY_att','R_avg_BODY_landed','R_avg_CLINCH_att','R_avg_CLINCH_landed','R_avg_DISTANCE_att','R_avg_DISTANCE_landed','R_avg_GROUND_att','R_avg_GROUND_landed','R_avg_HEAD_att','R_avg_HEAD_landed','R_avg_KD','R_avg_LEG_att','R_avg_LEG_landed','R_avg_PASS','R_avg_REV','R_avg_SIG_STR_att','R_avg_SIG_STR_landed','R_avg_SIG_STR_pct','R_avg_SUB_ATT','R_avg_TD_att','R_avg_TD_landed','R_avg_TD_pct','R_avg_TOTAL_STR_att','R_avg_TOTAL_STR_landed','R_avg_opp_BODY_att','R_avg_opp_BODY_landed','R_avg_opp_CLINCH_att','R_avg_opp_CLINCH_landed','R_avg_opp_DISTANCE_att','R_avg_opp_DISTANCE_landed','R_avg_opp_GROUND_att','R_avg_opp_GROUND_landed','R_avg_opp_HEAD_att','R_avg_opp_HEAD_landed','R_avg_opp_KD','R_avg_opp_LEG_att','R_avg_opp_LEG_landed','R_avg_opp_PASS','R_avg_opp_REV','R_avg_opp_SIG_STR_att','R_avg_opp_SIG_STR_landed','R_avg_opp_SIG_STR_pct','R_avg_opp_SUB_ATT','R_avg_opp_TD_att','R_avg_opp_TD_landed','R_avg_opp_TD_pct','R_avg_opp_TOTAL_STR_att','R_avg_opp_TOTAL_STR_landed','R_total_rounds_fought','R_total_time_fought(seconds)','B_Height_cms','B_Reach_cms','B_Weight_lbs','R_Height_cms','R_Reach_cms','R_Weight_lbs','B_age','R_age'],axis=1)
processed_df=[]
processed_df = pd.concat([df_B.reset_index(drop=True),df_R.reset_index(drop=True),dd.reset_index(drop=True)], axis=1)
len(processed_df)


# In[30]:


#export datframe in csv
processed_df.to_csv(r'D:\\processed_data.csv')
print("dataset exported to D:\\processed_data.csv")
