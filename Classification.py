# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # DOWN SYNDROME IN MICE

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df = pd.read_excel("Assg_1\Data_Cortex_Nuclear.xls") ## The Original Data


# %%
df.head()

# %% [markdown]
# ## PREDICTING THE GENEOTYPE

# %%
df.info()

# %% [markdown]
# #### Obseravtions: Here all the data is not numeric (MouseID, Genotype , Treatment, Behavior, Class). Hence, we should use some dummy varaibles in this case. However, for this problem we have to use features from the columns starting from DYRK1A_N to CaNA_N to predict the Genotype. Hence we drop the colums of MouseID, Treatment, Behavior and Class. Hence, we get the data used for this classification problem.
# 

# %%
cols_to_drop = ['MouseID', 'Treatment', 'Behavior', 'class']
df = df.drop(cols_to_drop , axis = 1)


# %%
df.head()


# %%
df.nunique()

# %% [markdown]
# #### Observation: As the target variable takes only two values, we have a binary classification problem. 
# #### Next: We convert the Genotype values to a numeric variable

# %%
## Converting categorical data to numeric values
df['Gene'] = pd.factorize(df['Genotype'])[0]


# %%
df


# %%
## Storing the transformation in dictionary
Gene_dict = {'0': 'Control' , '1' : 'Ts65Dn'}

## droping the categorical variable
df = df.drop('Genotype', axis = 1)


# %%
df.head()

# %% [markdown]
# #### Now, all our data is numeric and we can continue further with data pre processing
# 
# ### Filling missing values

# %%
# Check for which numeric columns have null values
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

# %% [markdown]
# #### All these columns have missing values. We fill them with the median of the column.

# %%
# Fill numeric rows with the median
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Fill missing numeric values with median since it's more robust than the mean
            df[label] = content.fillna(content.median())

# %% [markdown]
# #### Now, let's see if our data has any missing values

# %%
df.isna().sum()

# %% [markdown]
# #### Obseravtions: Good! Now, the data is in the required format and we can proceed with further pre-processign of the data and it's visualisation
# %% [markdown]
# #### For this problem, Gene is the target variable and rest variables are the features. For moving further, we split the given data into train and test datasets.
# 
# ## Train-Test Splitting

# %%
## Separating the features and the target variables
X = df.drop('Gene', axis =1)
y = df['Gene']


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2) # We keep 20 percent of the data for testing

# %% [markdown]
# ## Data Pre-Processing and Visulisation

# %%
for col in X_train.columns.values:
    list_vals = pd.unique(X_train[col])
    print("This feature", col,"is of the type", str(X_train[col].dtypes),"and has", str(len(list_vals)), "unique values and", str(np.isnan(list_vals).sum()), "null values")

# %% [markdown]
# #### As none of the feature have null values in them, we do not need any further pre-processing
# #### Next: We look at features which have low diversity

# %%
## Check if any feature has low diversity
for col in X_train.columns.values:
    list_vals = pd.unique(X_train[col])
    if (len(list_vals)<10): # If the feature has less than 10 unique values, then print them
        print(col)

# %% [markdown]
# #### Observations: 
# #### 1) None of the columns have null entries
# #### 2) All the features except quality have high diversity.
# #### Next: Plot the features to visualize the problem. 

# %%
# Plot the histogram of the features.

for col in X_train.columns.values:
    plt.hist(X_train[col])
    plt.xlabel(col)
    plt.show()

# %% [markdown]
# #### Observation: All the variables are spread well
# #### Next: We have a look at the corrtelation matrix

# %%
corrMatrix =X_train.corr(method = "spearman")
fig , ax = plt.subplots(figsize = (20,16))
sns.heatmap(abs(corrMatrix) , annot= False) # Show absolute value
plt.show()

# %% [markdown]
# #### There isn't much correlation in the features as can be seen above in the correlation matrix. Hence, we continue with all the features
# %% [markdown]
# ## Prepare data

# %%
# Normalize data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler() # For data normalization

scaler.fit(X_train)    #Compute mean and std
train_X = pd.DataFrame(scaler.transform(X_train)) # Use mean and deviation
train_Y = pd.DataFrame(y_train)
display("train_X")
print(train_X.mean)


# %%
test_X = pd.DataFrame(scaler.transform(X_test))
test_y = pd.DataFrame(y_test)
display(test_X)
print(test_X.mean())
print(test_y)

# %% [markdown]
# ## Rigorous training and validation
# %% [markdown]
# ### 1) Lasso Logistic Regression
# 
# #### In the following Lasso regression model, we use Cross-validation for hyperparameter tuning of the parameter alpha. 
# #### Interpretation of the hyperparameter: 
# #### 1) C : Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# #### 2) solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
# #### Algorithm to use in the optimization problem.
# 
# #### For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
# 
# #### For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# 
# #### ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
# 
# #### ‘liblinear’ and ‘saga’ also handle L1 penalty
# 
# #### ‘saga’ also supports ‘elasticnet’ penalty
# #### We use GridSearchCV for cross validating. 

# %%
from sklearn.model_selection import GridSearchCV

scoring = 'f1'

from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
hyperparameters = {'C':[0.1 , 1 , 10,100], 'solver':['newton-cg','lbfgs', 'liblinear','sag','saga']}
log = LogisticRegression(penalty='l1')
clf = GridSearchCV(log, param_grid = hyperparameters , scoring= scoring)
clf.fit(train_X, train_Y)
print('Best parameters: '+str(clf.best_params_))
print("Best score for "+ scoring+": " +str(clf.best_score_))


# %%
print("Test Classification Report: ")
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

y_true, y_pred = np.squeeze(y_test) , clf.predict(np.array(test_X))
print(classification_report(y_true , y_pred))

print("The ROC-AUC "+str((roc_auc_score(y_true, y_pred))))

# %% [markdown]
# ### SVR (Support Vector Regression)
# #### SVR is a powerful algorithm that allows us to choose how tolerant we are of errors, both through an acceptable error margin(ϵ) and through tuning our tolerance of falling outside that acceptable error rate. SVR gives us the flexibility to define how much error is acceptable in our model and will find an appropriate line (or hyperplane in higher dimensions) to fit the data.
# #### The objective function of SVR is to minimize the coefficients — more specifically, the l2-norm of the coefficient vector — not the squared error. The hyperparametersthat are tuned in SVR in the following model are:
# #### 1) Kernel : The function used to map a lower dimensional data into a higher dimensional data. There are various types of kernels like radial basis function (rbf), linear, gaussian, etc.
# ##### 2) C: The Penalty Parameter. It tells the algorithm how much you care about misclassified points.
# #### 3) Degree: It is the degree of the polynomial kernel function ('poly') and is ignored by all other kernels. The default value is 3. 
# 
# #### We use GridSearchCV for cross validating. 
# #### The scoring function used here is "R squared"
# #### R-squared: R Squared is a measurement that tells you to what extent the proportion of variance in the dependent variable is explained by the variance in the independent variables. In simpler terms, while the coefficients estimate trends, R-squared represents the scatter around the line of best fit. For example, if the R² is 0.80, then 80% of the variation can be explained by the model’s inputs. If the R² is 1.0 or 100%, that means that all movements of the dependent variable can be entirely explained by the movements of the independent variables. 

# %%
## First we will use an automated grid-search over renge of hyperparameters.

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

print("Training SVC using GridSearchCV")
scoring = 'f1'

from sklearn import svm
hyperparameters = {'kernel':('rbf' , 'linear' , 'poly') , 'C':[0.1 , 1 , 10,100], 'degree':[3,5,8]}

svc = svm.SVC()
clf_svr = GridSearchCV(estimator=svc, param_grid = hyperparameters , scoring= scoring)
clf_svr.fit(np.array(train_X) , np.squeeze(train_Y))
print('Best parameters: '+str(clf_svr.best_params_))
print("Best "+ scoring+": " +str(clf_svr.best_score_))


print("Test Classification Report: ")
y_true, y_pred = np.squeeze(y_test) , clf_svr.predict(np.array(test_X))
print(classification_report(y_true , y_pred))

print("The ROC-AUC "+str((roc_auc_score(y_true, y_pred))))

# %% [markdown]
# ## Random Forest Classification
# 
# #### Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.
# #### The hyperparameters that are tuned in the following model are:
# #### 1) max_depth : The max_depth of a tree in Random Forest is defined as the longest path between the root node and the leaf node: Using the max_depth parameter, I can limit up to what depth I want every tree in my random forest to grow
# #### 2) n_estimators : This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower.
# 
# #### We use GridSearchCV for cross validating. 
# #### The scoring function used here is "R squared"
# #### R-squared: R Squared is a measurement that tells you to what extent the proportion of variance in the dependent variable is explained by the variance in the independent variables. In simpler terms, while the coefficients estimate trends, R-squared represents the scatter around the line of best fit. For example, if the R² is 0.80, then 80% of the variation can be explained by the model’s inputs. If the R² is 1.0 or 100%, that means that all movements of the dependent variable can be entirely explained by the movements of the independent variables. 

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

print("Training Random Forest Classifier using GridSearchCV")
scoring = 'f1'

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
hyperparameters = {'max_depth':[2,5,10,20,40,100] , 'n_estimators':[10,30,80,100,130,170,200]}
clf_rfc = GridSearchCV(estimator=rfr, param_grid = hyperparameters , scoring= scoring)
clf_rfc.fit(np.array(train_X) , np.squeeze(train_Y))
print('Best parameters: '+str(clf_rfc.best_params_))
print("Best "+ scoring+": " +str(clf_rfc.best_score_))


print("Test Classification Report: ")
y_true, y_pred = np.squeeze(y_test) , clf_rfc.predict(np.array(test_X))
print(classification_report(y_true , y_pred))

print("The ROC-AUC "+str((roc_auc_score(y_true, y_pred))))

# %% [markdown]
# ## Recursive feature elimination and Cross Validation
# 
# #### Now, we try Lasso regresssion with the features which are important. We determine these important features using RFECV.

# %%
from sklearn.feature_selection import RFECV

clf_lr = LogisticRegression()
rfecv = RFECV(estimator = clf_lr , step = 1 , cv = 5, scoring = 'accuracy')
rfecv = rfecv.fit(train_X , train_Y)
print("Optimal number of features : ", rfecv.n_features_)
print("Best features : ",train_X.columns[rfecv.support_])


# %%
print(rfecv.grid_scores_)


# %%
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1,len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.show()


# %%
#### Now, let us look at the performance of our model with these features
train_X_rfecv = rfecv.transform(train_X)
test_X_rfecv = rfecv.transform(test_X)


# %%
clf_lr_rfecv = clf_lr.fit(train_X_rfecv,train_Y)


# %%
print("Test Classification Report: ")
y_true, y_pred = np.squeeze(test_y) , clf_lr_rfecv.predict(np.array(test_X_rfecv))
print(classification_report(y_true , y_pred))

print("The ROC-AUC "+str((roc_auc_score(y_true, y_pred))))

# %% [markdown]
# #### Observation : Here, we did not observe any improvement in the results compared to the original model with all the features.

