
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('iris')
# dataset
print(df)
print(df.shape)
print(df.columns)

#check the null value
print(df.isna().sum()) # data is not null
#check the duplicate value
print("Duplicate",df.duplicated().sum()) # here 1 duplicate value is present which is not affect that much
#number of unique value
print(df.nunique())
# information about dataset
print(df.info())
#To display the stats of data
print(df.describe())

# Here species is a object data type and it is out output feature
print(df['species'].value_counts()) # This is a balanced dataset

# correlation with each other
print(df[df.dtypes[df.dtypes=='float64'].index].corr())
sns.heatmap(df[df.dtypes[df.dtypes=='float64'].index].corr(),annot=True)
plt.show()

#*** Graphical **********
fig,axis=plt.subplots(2,2,figsize=(10,6))
plt.subplot(221)
sns.histplot(data=df,x='sepal_length',kde=True) # nearly gaussian Distribution but std is little high
plt.subplot(222)
sns.histplot(data=df,x='sepal_width',kde=True) # gaussian Distibution
plt.subplot(223)
sns.histplot(data=df,x='petal_length',kde=True)
plt.subplot(224)
sns.histplot(data=df,x='petal_width',kde=True)
plt.show()

fig,axis=plt.subplots(2,2,figsize=(10,6))
plt.subplot(221)
sns.histplot(data=df,x='sepal_length',kde=True,hue='species') # generally sepel_length --> virginica > versicolor > setosa
plt.subplot(222)
sns.histplot(data=df,x='sepal_width',kde=True,hue='species') # generally sepal_width ---> setosa > (virginica ~= versicolor)
plt.subplot(223)
sns.histplot(data=df,x='petal_length',kde=True,hue='species')  # petal_length --->  virginica > versicolor > setosa
plt.subplot(224)
sns.histplot(data=df,x='petal_width',kde=True,hue='species') # petal_width --->  virginica > versicolor > setosa
plt.show() 

# Scatter plot
fig,axis=plt.subplots(2,2,figsize=(10,6))
plt.subplot(221)
sns.scatterplot(data=df,x='sepal_length',y='sepal_width',hue='species')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.subplot(222)
sns.scatterplot(data=df,x='petal_length',y='petal_width',hue='species')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.legend()
plt.subplot(223)
sns.scatterplot(data=df,x='sepal_length',y='petal_length',hue='species')
plt.ylabel('petal_length')
plt.xlabel('sepal_length')
plt.legend()
plt.subplot(224)
sns.scatterplot(data=df,x='sepal_width',y='petal_width',hue='species')
plt.xlabel('sepal_width')
plt.ylabel('petal_width')
plt.legend()
plt.show()

#**** Pair plot *************
sns.pairplot(df,hue='species',height=2)
plt.show() 

# ********* Box plot (for outliers)*****
fig,axis=plt.subplots(2,2,figsize=(10,6))
plt.subplot(221)
sns.boxplot(data=df,x='sepal_length',hue='species') 
plt.xlabel('sepal_length')
plt.legend() 
plt.subplot(222)
sns.boxplot(data=df,x='petal_length',hue='species')
plt.xlabel('petal_length')
plt.legend()
plt.subplot(223)
sns.boxplot(data=df,x='sepal_length',hue='species')
plt.xlabel('sepal_length')
plt.legend()
plt.subplot(224)
sns.boxplot(data=df,x='sepal_width',hue='species')
plt.xlabel('sepal_width')
plt.legend()
plt.show() 






 
















 