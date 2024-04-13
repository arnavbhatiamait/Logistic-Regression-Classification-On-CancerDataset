# %% [markdown]
# Logistic regression 

# %% [markdown]
# Import the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
df=pd.read_csv("breast_cancer.csv")
df.head()

# %%
x=df.iloc[:,1:-1].values
x

# %%
y=df.iloc[:,-1].values
y

# %% [markdown]
# Splitting the dataset into Training and Testing Set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0, test_size=0.2)

# %%
x_train

# %%
x_test

# %%
y_test

# %%
y_train

# %% [markdown]
# Training The Logistic Regression Model on the Training Set

# %%
from sklearn.linear_model import LogisticRegression
classification=LogisticRegression(random_state=0)
classification.fit(x_train,y_train)

# %% [markdown]
# Prediting the y test data and comparing it to the test result

# %%
y_pred=classification.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Confussion Matrix

# %%
import seaborn as sns
from sklearn.metrics import confusion_matrix ,accuracy_score,  classification_report
cm= confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,fmt='g',annot=True)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Confussion matrix")
plt.savefig("Confussion matrix.png")
plt.show()


# %% [markdown]
# Accuracy Score

# %%
accu=accuracy_score(y_test,y_pred)
print(accuracy_score(y_test,y_pred))

# %% [markdown]
# Classification Report

# %%
print(classification_report(y_test,y_pred))

# %% [markdown]
# Computing the accuracy using with k-Fold Cross Validation

# %%
from sklearn.model_selection import cross_val_score
# ! cv = no of cross validartion eastimator is where we want to estimate and x_train and y_train are data 
accuracies=cross_val_score(estimator=classification,X=x_train,y=y_train,cv=10)
print("Accuracy: {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f}%".format(accuracies.std()*100))


