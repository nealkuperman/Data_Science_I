#!/usr/bin/env python
# coding: utf-8

# # Math 610: Homework 2
# **Neal Kuperman**
# 

# In[119]:


#%% Packages
import os

from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns

import pandas as pd
import numpy as np
import statsmodels.api as smf

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, confusion_matrix,
                             roc_auc_score, classification_report, ConfusionMatrixDisplay, 
                             roc_curve, auc)
from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

from IPython.display import display, HTML


RANDOM_STATE = 1
VERBOSE = False
PRINT_LATEX = False
SAVE_FIGS = False

# Set working directory to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ## Problem 1: College Data Analysis
# 
# Consider the `College` data from the ISLP package. Details about the data is described on page 65 of the ISLP textbook for this class ([https://islp.readthedocs.io/en/latest/datasets/College.html](https://islp.readthedocs.io/en/latest/datasets/College.html)).
# 
# We would like to *predict* the **number of applications** received using the other variables.
# 
# 80% of the data (randomly generated) will be treated as training data. The rest will be the test data.
# 
# ---
# 
# **a)** Fit a linear model using least squares and report the estimate of the test error.
# 
# **b)** Fit a tree to the data. Summarize the results. Unless the number of terminal nodes is large, display the tree graphically. Report its MSE.
# 
# **c)** Use Cross validation to determine whether pruning is helpful and determine the optimal size for the pruned tree. Compare the pruned and un-pruned trees. Report MSE for the pruned tree. Which predictors seem to be the most important?
# 
# **d)** Use a bagging approach to analyze the data with B = 500 and B = 1000. Compute the MSE. Which predictors seem to be the most important?
# 
# **e)** Repeat (d) with a random forest approach with B = 500 and B = 1000, and m ≈ p = 3.
# 
# **f)** Compare the results from the various methods. Which method would you recommend?

# In[120]:


college_original = load_data("College")

if VERBOSE:
    display(college_original.head())
    display(college_original.describe().round(3))

if PRINT_LATEX:
    desc = college_original.describe()

    # Number of columns per table
    cols_per_table = 6

    # Get all column names
    all_cols = desc.columns.tolist()

    # Split columns into chunks of 5
    for i in range(0, len(all_cols), cols_per_table):
        cols_chunk = all_cols[i:i+cols_per_table]
        desc_subset = desc[cols_chunk].style.format("{:.2f}")

        print(f"\n# Table {i//cols_per_table + 1}: Columns {i+1}-{min(i+cols_per_table, len(all_cols))}")
        print(desc_subset.to_latex())
        print("\n" + "="*80 + "\n")



# Before answering any of the questions, we need to process the data
# 1) split into 80/20 train/test split
# 2) Convert the Private variable to a numeric value using one-hot encoding or dummy in Pandas
# 3) Scale numeric independent variables using the standard scalar transform from scikit-learn 
# 4) Add intercept column to train and test dfs which is needed when passing a dataframe into statsmodel OLS function
# 

# In[121]:


y = college_original["Apps"]
X_original = college_original.drop(columns=["Apps"])
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X_original, y, test_size=0.3, random_state=RANDOM_STATE
)

#============================
# One-hot encoding code
#============================
num_cols = X_original.select_dtypes(include=['number']).columns
cat_cols = X_original.select_dtypes(include=['category']).columns
transformer = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop='first'), cat_cols),
    ],
    remainder="passthrough" 
)


X_train_transformed = transformer.fit_transform(X_train_original)
X_test_transformed = transformer.transform(X_test_original)
feature_names = [name.split("__", 1)[1] for name in transformer.get_feature_names_out()]
X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
X_train.insert(0, 'Intercept', 1.0)
X_test.insert(0, 'Intercept', 1.0)

if VERBOSE:
    display(X_train.head())


# **a)** Fit a Linear Model using least squares and report the estimate of the test error.

# In[122]:


# Linear model and error report
model = smf.OLS(y_train.values, X_train).fit()


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
mse_train = mean_squared_error(y_train.values, y_train_pred)
mse_test = mean_squared_error(y_test.values, y_test_pred)
r2_train = r2_score(y_train.values, y_train_pred)
r2_test = r2_score(y_test.values, y_test_pred)

if VERBOSE:
    print(model.summary())

    print(f"Train MSE: {mse_train}")
    print(f"Train R2: {r2_train}")
    print(f"\n\nTest MSE: {mse_test}")
    print(f"Test R2: {r2_test}")


    plt.plot(y_test.values, y_test_pred, 'o')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('CollegeActual vs Predicted')
    plt.show()

if PRINT_LATEX:
    print(model.summary().as_latex())



# **b)** Fit a *regression* tree to the data. Summarize the results. Unless the number of terminal nodes is large, display the tree graphically. Report its MSE.

# In[123]:


def fit_and_plot_tree(X_train, y_train, X_test, y_test, max_depth, save_fig=False):
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE, criterion="squared_error")
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    y_pred_train = tree.predict(X_train)
    train_mse_tree = mean_squared_error(y_train, y_pred_train)
    train_r2_tree = r2_score(y_train, y_pred_train)
    test_mse_tree = mean_squared_error(y_test, y_pred_tree)
    test_r2_tree = r2_score(y_test, y_pred_tree)

    if VERBOSE:
        print("\n")
        print(f"Tree w/ max depth = {max_depth}")
        print("="*60)   

        if max_depth < 5:
            fig, ax = plt.subplots(figsize=(18, 10))
            plot_tree(tree, max_depth=max_depth, feature_names=pd.get_dummies(X_train).columns, ax=ax, fontsize=10, filled=True)
            ax.set_title(f"Tree w/ max depth = {max_depth}")
            if save_fig and SAVE_FIGS:
                plt.savefig(f"../images/HW_2/prob_1_tree_max_depth_{max_depth}.png", dpi=300)
            plt.show()

        print(f"training MSE: {train_mse_tree}")
        print(f"training R2: {train_r2_tree}")
        print(f"test MSE: {test_mse_tree}")
        print(f"test R2: {test_r2_tree}")
        print("="*60)

    return tree

tree_depth_3 = fit_and_plot_tree(X_train, y_train, X_test, y_test, 3)
tree_depth_10 = fit_and_plot_tree(X_train, y_train, X_test, y_test, 10)

importances_3 = pd.Series(
    tree_depth_3.feature_importances_,
    index=pd.get_dummies(X_train).columns
).sort_values(ascending=False)

importances_10 = pd.Series(
    tree_depth_10.feature_importances_,
    index=pd.get_dummies(X_train).columns
).sort_values(ascending=False)

if VERBOSE:
    print(importances_3.head(10))
    print(importances_10.head(10))


# **c)** Use Cross validation to determine whether pruning is helpful and determine the optimal size for the pruned tree. Compare the pruned and un-pruned trees. Report MSE for the pruned tree. Which predictors seem to be the most important?
# 
# We will use the `cost_complexity_pruning_path()` method of the `DecisionTreeRegressor` class to extract cost-complexity values. We will use the tree with a max depth of 10 to see how much reduction in complexity the pruning can achieve

# In[124]:


path = tree_depth_10.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

kfold = KFold(10,
              random_state=RANDOM_STATE,
              shuffle=True)

grid = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE, criterion="squared_error"),
                        {'ccp_alpha': alphas},
                        refit=True,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

best_alpha = grid.best_params_["ccp_alpha"]
best_cv_mse = -grid.best_score_

if VERBOSE:
    print(f"Best alpha: {best_alpha}")
    print(f"Best CV MSE: {best_cv_mse}")
    # print(f"Best Score: {grid.best_score_}")


# In[125]:


best_tree = grid.best_estimator_

y_train_pred = best_tree.predict(X_train)
y_test_pred = best_tree.predict(X_test)

test_mse = mean_squared_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)  
test_r2 = r2_score(y_test, y_test_pred)

if VERBOSE:
    print(f"Best Tree Training MSE: {train_mse}")
    print(f"Best Tree Training R2: {train_r2}")
    print(f"Best Tree Test MSE: {test_mse}")
    print(f"Best Tree Test R2: {test_r2}")
    print(f"Best Tree Leaves: {best_tree.get_n_leaves()}")
    print(f"Best Tree Depth: {best_tree.get_depth()}")
    print(f"Best Tree R2: {r2_score(y_test, y_test_pred)}")

    plt.plot(y_test.values, y_test_pred, 'o', label='best')
    plt.plot(y_test.values, tree_depth_10.predict(X_test), 'o', label = "10")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    ax = subplots(figsize=(12, 12))[1]
    plot_tree(best_tree,
            feature_names=feature_names,
            ax=ax,
            filled=True);
    plt.title(f'Pruned Best Tree \nalpha = {best_alpha:.3f}')
    if SAVE_FIGS:
        plt.savefig(f'../images/HW_2/pruned_best_tree_alpha_{best_alpha:.3f}.png', dpi=300)
    plt.show()


# In[126]:


importances = pd.Series(
    best_tree.feature_importances_,
    index=pd.get_dummies(X_train).columns
).sort_values(ascending=False)

if VERBOSE:
    print(importances.head(10))
if PRINT_LATEX:
    print(importances.to_latex())


# **d)** Use a bagging approach to analyze the data with B = 500 and B = 1000. Compute the MSE. Which predictors seem to be the most important?
# 

# In[127]:


def bagging_regressor(X_train, y_train, X_test, y_test, n_estimators, save_fig=False):

    regressors = []

    if VERBOSE:
        fig, axes = plt.subplots(1, len(n_estimators), figsize=(12, 5))

    for i, n in enumerate(n_estimators):
        bag_n = BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=n,
            random_state=RANDOM_STATE
        )

        regressors.append(bag_n)
        bag_n.fit(X_train, y_train)

        if VERBOSE:
            y_train_pred_n = bag_n.predict(X_train)     
            y_test_pred_n = bag_n.predict(X_test)
            train_mse_bag_n = mean_squared_error(y_train, y_train_pred_n)
            train_r2_bag_n = r2_score(y_train, y_train_pred_n)
            test_mse_bag_n = mean_squared_error(y_test, y_test_pred_n)
            test_r2_bag_n = r2_score(y_test, y_test_pred_n)
            print(f"Bagging {n} Training MSE: {train_mse_bag_n}")
            print(f"Bagging {n} Training R2: {train_r2_bag_n}")
            print(f"Bagging {n} Test MSE: {test_mse_bag_n}")
            print(f"Bagging {n} Test R2: {test_r2_bag_n}")

            axes[i].scatter(y_test, y_test_pred_n, alpha=0.6, edgecolors='k', linewidths=0.5)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual - Test Data')
            axes[i].set_ylabel('Predicted - Test Data')
            axes[i].set_title(f'Bagging (B={n})\nR$^2$: {r2_score(y_test, y_test_pred_n):.2f}, MSE: {mean_squared_error(y_test, y_test_pred_n):.2f}') 

    if VERBOSE:
        plt.tight_layout()

    if save_fig and VERBOSE:
        estimator_names = str.join("_", [f"B={n}" for n in n_estimators])
        if SAVE_FIGS:
            plt.savefig(f'../images/HW_2/bagging_regressor_n_{estimator_names}.png', dpi=300)

    if VERBOSE:
        plt.show()

    return regressors

regressors = bagging_regressor(X_train, y_train, X_test, y_test, [500, 1000], save_fig=True)
bag_500, bag_1000 = regressors[0], regressors[1]




# In[128]:


def print_feature_importance(regressor):
    imp = np.mean([
        tree.feature_importances_
        for tree in regressor.estimators_
    ], axis=0)
    display(pd.Series(imp, index=pd.get_dummies(X_train).columns)\
      .sort_values(ascending=False)\
      .head(10))

if VERBOSE:
    print("Feature Importance for Bagging (B=500)")
    print("="*60)
    print_feature_importance(bag_500)

    print("\n")
    print("Feature Importance for Bagging (B=1000)")
    print("="*60)
    print_feature_importance(bag_1000)


# **e)** Repeat (d) with a random forest approach with B = 500 and B = 1000, and m ≈ p = 3.
# 

# In[129]:


def RF_regressor(X_train, y_train, X_test, y_test, n_estimators, save_fig=False):

    regressors = []

    if VERBOSE:
        fig, axes = plt.subplots(1, len(n_estimators), figsize=(12, 5))

    for i, n in enumerate(n_estimators):
        RF_n = RandomForestRegressor(
            n_estimators=n,
            max_features=3,
            random_state=RANDOM_STATE
        )

        regressors.append(RF_n)

        RF_n.fit(pd.get_dummies(X_train), y_train)

        if VERBOSE:
            y_train_pred_n = RF_n.predict(pd.get_dummies(X_train))     
            y_test_pred_n = RF_n.predict(pd.get_dummies(X_test))
            train_mse_RF_n = mean_squared_error(y_train, y_train_pred_n)
            train_r2_RF_n = r2_score(y_train, y_train_pred_n)
            test_mse_RF_n = mean_squared_error(y_test, y_test_pred_n)
            test_r2_RF_n = r2_score(y_test, y_test_pred_n)
            print(f"RF w {n} estimators Training MSE: {train_mse_RF_n}")
            print(f"RF w {n} estimators Training R2: {train_r2_RF_n}")
            print(f"RF w {n} estimators Test MSE: {test_mse_RF_n}")
            print(f"RF w {n} estimators Test R2: {test_r2_RF_n}")

            axes[i].scatter(y_test, y_test_pred_n, alpha=0.6, edgecolors='k', linewidths=0.5)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual - Test Data')
            axes[i].set_ylabel('Predicted - Test Data')
            axes[i].set_title(f'RF (num estimators={n})\nR$^2$: {r2_score(y_test, y_test_pred_n):.2f}, MSE: {mean_squared_error(y_test, y_test_pred_n):.2f}') 

    if VERBOSE:
        plt.tight_layout()

    if save_fig:
        estimator_names = str.join("_", [f"estimators={n}" for n in n_estimators])
        if SAVE_FIGS:
            plt.savefig(f'../images/HW_2/RF_regressor_n_{estimator_names}.png', dpi=300)

    if VERBOSE:
        plt.show()


    return regressors

regressors = RF_regressor(X_train, y_train, X_test, y_test, [500, 1000], save_fig=False)
RF_500, RF_1000 = regressors[0], regressors[1]





# In[130]:


def print_feature_importance(regressor):
    imp = np.mean([
        tree.feature_importances_
        for tree in regressor.estimators_
    ], axis=0)
    display(pd.Series(imp, index=pd.get_dummies(X_train).columns)\
      .sort_values(ascending=False)\
      .head(10))

if VERBOSE:
    print("Feature Importance for Bagging (B=500)")
    print("="*60)
    print_feature_importance(RF_500)

    print("\n")
    print("Feature Importance for Bagging (B=1000)")
    print("="*60)
    print_feature_importance(RF_1000)


# **f)** Compare the results from the various methods. Which method would you recommend?
# 
# I would recommend the random forest with bagging classifier. Although the linear regression model has a similar MSE for the test data, the random forest will work with non-linear relationships making it a more robust classifier. The major benefit to the linear regression model is the interpretability of the coefficients.

# ## Problem 2: College Data Analysis
# 
# Consider the business school admission data available in the admission.csv. The admission officer of a business school has used an *“index”* of undergraduate grade point average (GPA,𝑋1) and graduate management aptitude test (GMAT,𝑋2) scores to help decide which applicants should be admitted to the school’s graduate programs. This index is used to categorize each applicant into one of three groups – admit (group 1), do not admit (group 2), and borderline (group 3). We will take the last ***four*** observations in **<u>each category</u>** as test data and the remaining observations as training data.
# 
# 
# 
# **a)** Perform an exploratory analysis of the training data by examining appropriate plots and comment on how helpful these predictors may be in predicting response.
# 
# **b)** Perform an LDA using the training data. Superimpose the decision boundary on an appropriate display of the data. Does the decision boundary seem sensible? In addition, compute the confusion matrix and overall misclassification rate based on both training and test data. What do you observe?
# 
# **c)** Repeat (b) using QDA.
# 
# **d)** Fit a KNN with K chosen optimally using test error rate. Report error rate, sensitivity, specificity, and AUC for the optimal KNN based on the training data. Also, report its estimated test error rate.
# 
# **e)** Compare the results in (b), (c) and (d). Which classifier would you recommend? Justify your conclusions.
# 
# ---

# </br>
# </br>
# 
# **a)** Perform an exploratory analysis of the training data by examining appropriate plots and comment on how helpful these predictors may be in predicting response.
# 

# In[131]:


admin_data = pd.read_csv("./admission.csv")

if VERBOSE:
    display(HTML('<h3>Head of Admission Data</h3>'))
    display(admin_data.head())
    display(HTML('<h3>Description of Admission Data</h3>'))
    display(admin_data.describe())
    # print(admin_data.describe().to_latex())

    # Check for missing values
    display(HTML('<h3>Missing Values in Admission Data</h3>'))
    display(admin_data.isnull().sum())

# Split the data into training and test sets
train_data = admin_data[admin_data.groupby('Group').cumcount(ascending=False) >= 4]
test_data = admin_data.groupby('Group').tail(4)
# train_data = admin_data.iloc[:-4]
# test_data = admin_data.iloc[-4:]
y_train = train_data['Group']
y_test = test_data['Group']
X_train = train_data[['GPA', 'GMAT']]
X_test = test_data[['GPA', 'GMAT']]

# display(HTML(admin_data.to_html(max_rows=None)))


# In[132]:


#EDA

if VERBOSE:
    # Check for missing values
    missing_values = admin_data.isnull().sum()
    print("Missing values in adminission data:")
    print(missing_values)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Bar plot of Group counts
    train_data['Group'].value_counts().plot(kind='bar', ax=axes[0], edgecolor='k')
    axes[0].set_title('Group Distribution')
    axes[0].set_xlabel('Group')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    # Histogram for GPA
    axes[1].hist(train_data['GPA'], bins=20, edgecolor='k', alpha=0.7)
    axes[1].set_title('GPA Distribution')
    axes[1].set_xlabel('GPA')
    axes[1].set_ylabel('Frequency')

    # Histogram for GMAT
    axes[2].hist(train_data['GMAT'], bins=20, edgecolor='k', alpha=0.7)
    axes[2].set_title('GMAT Distribution')
    axes[2].set_xlabel('GMAT')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    if SAVE_FIGS:   
        plt.savefig('../images/HW_2/admission_data_EDA_histograms.png', dpi=300)
    plt.show()

    sns.scatterplot(data=admin_data, x='GPA', y='GMAT', hue='Group', palette='viridis')
    plt.title('Scatter plot of GPA vs GMAT')
    if SAVE_FIGS:
        plt.savefig('../images/HW_2/admission_data_EDA_scatterplot.png', dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=admin_data, x='Group', y='GPA', ax=axes[0])
    axes[0].set_title('GPA Distribution by Group')

    sns.boxplot(data=admin_data, x='Group', y='GMAT', ax=axes[1])
    axes[1].set_title('GMAT Distribution by Group')

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig('../images/HW_2/admission_data_EDA_boxplots.png', dpi=300)
    plt.show()


# **comment on how helpful these predictors may be in predicting response.**

# **b)** Perform an LDA using the training data. Superimpose the decision boundary on an appropriate display of the data. Does the decision boundary seem sensible? In addition, compute the confusion matrix and overall misclassification rate based on both training and test data. What do you observe?
# 

# In[133]:


# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
y_pred_test = lda.predict(X_test)
y_pred_train = lda.predict(X_train)

lda.classes_

if VERBOSE:
    display(HTML('<h3>Confusion Matrix for Training Data</h3>'))
    display(confusion_table(y_pred_train, y_train.values, lda.classes_))

    if PRINT_LATEX:
        print(confusion_table(y_pred_train, y_train.values, lda.classes_).to_latex())

    display(HTML('<h3>Confusion Matrix for Test Data</h3>'))
    display(confusion_table(y_pred_test, y_test.values, lda.classes_))

    if PRINT_LATEX:
        print(confusion_table(y_pred_test, y_test.values, lda.classes_).to_latex())

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    # Misclassification rate = (Total Samples - Correct Predictions) / Total Samples
    misclass_rate_test = (cm_test.sum() - cm_test.diagonal().sum()) / cm_test.sum() 
    misclass_rate_train = (cm_train.sum() - cm_train.diagonal().sum()) / cm_train.sum() 
    misclass_rate_total = (cm_test.sum() - cm_test.diagonal().sum() + cm_train.sum() - cm_train.diagonal().sum()) / (cm_test.sum() + cm_train.sum())
    print(f"Misclassification rate for test data: {misclass_rate_test*100:.2f}%")
    print(f"Misclassification rate for train data: {misclass_rate_train*100:.2f}%")
    print(f"Misclassification rate for total data: {misclass_rate_total*100:.2f}%")


# In[134]:


if VERBOSE:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the data points
    scatter = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
                        c=y_train, cmap='viridis', edgecolors='k', s=50)

    # Get GPA range for plotting lines
    gpa_range = np.linspace(X_train.iloc[:, 0].min() - 0.1, 
                            X_train.iloc[:, 0].max() + 0.1, 100)

    # Plot decision boundary between each pair of classes
    classes = lda.classes_
    colors = ['red', 'blue', 'green']
    pairs = [(0, 1), (0, 2), (1, 2)]  # class index pairs

    for (i, j), color in zip(pairs, colors):
        # Coefficients for the boundary line
        coef_diff = lda.coef_[i] - lda.coef_[j]
        intercept_diff = lda.intercept_[i] - lda.intercept_[j]

        # Solve for GMAT: coef_diff[0]*GPA + coef_diff[1]*GMAT + intercept_diff = 0
        # GMAT = -(coef_diff[0]*GPA + intercept_diff) / coef_diff[1]
        gmat_boundary = -(coef_diff[0] * gpa_range + intercept_diff) / coef_diff[1]

        ax.plot(gpa_range, gmat_boundary, color=color, linestyle='--', linewidth=2,
                label=f'Boundary {classes[i]} vs {classes[j]}')

    ax.set_xlim(X_train.iloc[:, 0].min() - 0.1, X_train.iloc[:, 0].max() + 0.1)
    ax.set_ylim(X_train.iloc[:, 1].min() - 20, X_train.iloc[:, 1].max() + 20)
    ax.set_xlabel('GPA')
    ax.set_ylabel('GMAT')
    ax.set_title('LDA Decision Boundaries (Analytical)')
    ax.legend()
    plt.show()


# In[135]:


if VERBOSE:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot decision boundaries (one line!)
    disp = DecisionBoundaryDisplay.from_estimator(
        lda, 
        X_train, 
        response_method="predict",
        alpha=0.5,
        ax=ax,
        xlabel='GPA',
        ylabel='GMAT'
    )

    # Overlay the actual data points
    scatter_train = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
                        c=y_train, edgecolor="k", cmap='viridis')
    scatter_test = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], 
                        c=y_test, edgecolor="k", cmap='viridis', marker='D', s=75)

    # Create invisible scatter points just for legend
    train_handle = ax.scatter([], [], marker='o', c='gray', edgecolor='k', label='Train')
    test_handle = ax.scatter([], [], marker='D', c='gray', edgecolor='k', label='Test')

    # Legend 1: Groups
    legend1 = ax.legend(*scatter_train.legend_elements(), title="Group", loc='upper left')
    ax.add_artist(legend1)

    # Legend 2: Markers
    ax.legend(handles=[train_handle, test_handle], title="Data", loc='upper right')

    ax.set_title('LDA Decision Boundaries')
    if SAVE_FIGS:
        plt.savefig('../images/HW_2/problem_2_lda_decision_boundaries.png', dpi=300)
    plt.show()


# <!-- Perform an QDA using the training data. Superimpose the decision boundary on an appropriate display of the data. Does the decision boundary seem sensible? In addition, compute the confusion matrix and overall misclassification rate based on both training and test data. What do you observe? -->
# 
# **c)** Repeat (b) using QDA.

# In[136]:


# Apply Linear Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_test = qda.predict(X_test)
y_pred_train = qda.predict(X_train)

qda.classes_

if VERBOSE:
    display(HTML('<h3>Confusion Matrix for Training Data</h3>'))
    display(confusion_table(y_pred_train, y_train.values, lda.classes_))
    print(confusion_table(y_pred_train, y_train.values, lda.classes_).to_latex())

    display(HTML('<h3>Confusion Matrix for Test Data</h3>'))
    display(confusion_table(y_pred_test, y_test.values, lda.classes_))
    print(confusion_table(y_pred_test, y_test.values, lda.classes_).to_latex())

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    # Misclassification rate = (Total Samples - Correct Predictions) / Total Samples
    misclass_rate_test = (cm_test.sum() - cm_test.diagonal().sum()) / cm_test.sum() 
    misclass_rate_train = (cm_train.sum() - cm_train.diagonal().sum()) / cm_train.sum() 
    misclass_rate_total = (cm_test.sum() - cm_test.diagonal().sum() + cm_train.sum() - cm_train.diagonal().sum()) / (cm_test.sum() + cm_train.sum())
    print(f"Misclassification rate for test data: {misclass_rate_test*100:.2f}%")
    print(f"Misclassification rate for train data: {misclass_rate_train*100:.2f}%")
    print(f"Misclassification rate for all data: {misclass_rate_total*100:.2f}%")


# In[137]:


if VERBOSE:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot decision boundaries (one line!)
    disp = DecisionBoundaryDisplay.from_estimator(
        qda, 
        X_train, 
        response_method="predict",
        alpha=0.5,
        ax=ax,
        xlabel='GPA',
        ylabel='GMAT'
    )

    # Overlay the actual data points
    scatter_train = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
                        c=y_train, edgecolor="k", cmap='viridis')
    scatter_test = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], 
                        c=y_test, edgecolor="k", cmap='viridis', marker='D', s=75)

    # Create invisible scatter points just for legend
    train_handle = ax.scatter([], [], marker='o', c='gray', edgecolor='k', label='Train')
    test_handle = ax.scatter([], [], marker='D', c='gray', edgecolor='k', label='Test')

    # Legend 1: Groups
    legend1 = ax.legend(*scatter_train.legend_elements(), title="Group", loc='upper left')
    ax.add_artist(legend1)

    # Legend 2: Markers
    ax.legend(handles=[train_handle, test_handle], title="Data", loc='upper right')

    ax.set_title('QDA Decision Boundaries')
    if SAVE_FIGS:
        plt.savefig('../images/HW_2/problem_2_qda_decision_boundaries.png', dpi=300)
    plt.show()


# **d)** Fit a KNN with K chosen optimally using test error rate. Report error rate, sensitivity, specificity, and AUC for the optimal KNN based on the training data. Also, report its estimated test error rate.
# 
# </br>
# </br>
# 
# **Error rate**: percentage misclassification
# $$\frac{\text{Misclassified}}{\text{Total}} = 1 - \text{Accuracy}$$
# **Sensitivity** (*true positive rate*): measures the proportion of actual positives correctly identified as positive 
# $$\frac{\text{Number of true positives}}{\text{Number of true positives} + \text{Number of false negatives}}$$
# **Specificity** (*true negative rate*): measures the proportion of actual negatives correctly identified. NOTE: Also called recall
# $$\frac{\text{Number of true negatives}}{\text{Number of true negatives} + \text{Number of false positives}}$$
# 
# **Precision**: Ratio of the correctly predicted class to the total predicted class
# $$\frac{\text{Number of true negatives}}{\text{Number of true positives} + \text{Number of false positives}}$$
# **F1 Score**: A harmonic means between the Precision and Recall score
# $$2 \times \frac{\text{Precision} \times \text{Specificity}}{\text{Precision} + \text{Specificity}}$$
# 
# 
# ROC Curve: A Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system by plotting the true-positive rate (sensitivity) against the false-positive rate (1-specificity) across various decision thresholds $^{[1]}$. For a KNN classifier, the predicted probability for a class is the proportion of the K neighbors belonging to that class (i.e., $P(\text{class K}|x) = \frac{\text{number of neighbors in class}}{K}$). The ROC curve is generated by varying the probability threshold required to classify a point as positive. When building the ROC curve, we vary t, where $\text{t} \in [0, 1]$ with the decision rule being predict Class_n if $P(\text{Class\_n}|x) \geq t$
# 
# For a multiclass classification problem, we can no longer generate a single ROC curve since it is based ona binary classifier. We can use the one vs all scheme, which compares each class against all the others (assumed as one) [(scikit learn tutorial)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
# 
# 
# 
# **NOTE**: Since KNN is a distance based method, we want to make sure that GMAT and GRE are on similar scales. We are going to use the standard scaling method from scikit learn on our training and test data.
# 
# 
# [1] https://en.wikipedia.org/wiki/Receiver_operating_characteristic
# 

# In[138]:


# Create a pipeline with scaler + KNN
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Optimal KNN Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different K values to find optimal
k_range = range(1, len(X_train_scaled)-1)
train_errors = []
test_errors = []

for k in k_range:
    knn = knn_pipeline.set_params(knn__n_neighbors=k)
    # knn.fit(X_train_scaled, y_train)

    knn.fit(X_train, y_train)

    train_errors.append(1 - accuracy_score(y_train, knn_pipeline.predict(X_train)))
    test_errors.append(1 - accuracy_score(y_test, knn_pipeline.predict(X_test)))

# Find optimal K (lowest test error)
optimal_k = k_range[np.argmin(test_errors)]

if VERBOSE:
    print(f"Optimal K: {optimal_k}")

    # Plot K vs Error Rate
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, train_errors, label='Train Error', marker='o')
    plt.plot(k_range, test_errors, label='Test Error', marker='s')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.title('KNN: K vs Error Rate')
    plt.legend()

    if SAVE_FIGS:
        plt.savefig('../images/HW_2/problem_2_knn_k_vs_error_rate.png', dpi=300)
    plt.show()

# Fit optimal KNN
knn_opt = knn_pipeline.set_params(knn__n_neighbors=optimal_k)
knn_opt.fit(X_train, y_train)

y_train_pred = knn_opt.predict(X_train)
y_test_pred = knn_opt.predict(X_test)

if VERBOSE:
    # Metrics on TRAINING data
    print("\n=== Training Data Metrics ===")
    print(f"Training Error Rate: {1 - accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")

    # Confusion matrix for training
    cm_train = confusion_matrix(y_train, y_train_pred)
    print(f"\nConfusion Matrix (Train):\n{cm_train}")

    # Per-class sensitivity (recall) and specificity
    print("\nPer-Class Metrics (Training):")
    print(classification_report(y_train, y_train_pred))


# In[139]:


# Calculate sensitivity & specificity for each class (one-vs-rest)
classes = knn_opt.classes_
for i, _cls in enumerate(classes):
    # Binary: this class vs all others
    y_binary = (y_train == _cls).astype(int)
    pred_binary = (y_train_pred == _cls).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_binary, pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    if VERBOSE:
        print(f"Class {_cls}: Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")

# AUC (multiclass: one-vs-rest)
if hasattr(knn_opt, 'predict_proba'):
    y_train_proba = knn_opt.predict_proba(X_train)
    auc_train = roc_auc_score(y_train, y_train_proba, multi_class='ovr')

    if VERBOSE:
        print(f"\nAUC (Training, OvR): {auc_train:.4f}")

if VERBOSE:
    # Metrics on TEST data
    print("\n=== Test Data Metrics ===")
    print(f"Test Error Rate: {1 - accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    cm_test = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix (Test):\n{cm_test}")

    # Confusion matrix display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(knn_opt, X_train, y_train, ax=axes[0], cmap='Blues')
    axes[0].set_title(f'KNN (K={optimal_k}) - Training')
    ConfusionMatrixDisplay.from_estimator(knn_opt, X_test, y_test, ax=axes[1], cmap='Blues')
    axes[1].set_title(f'KNN (K={optimal_k}) - Test')
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig('../images/HW_2/problem_2_knn_confusion_matrix.png', dpi=300)
    plt.show()

if VERBOSE:

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot decision boundaries with HIGHER resolution for smoother appearance
    disp = DecisionBoundaryDisplay.from_estimator(
        knn_opt, 
        X_train, 
        response_method="predict",
        alpha=0.5,
        ax=ax,
        xlabel='GPA',
        ylabel='GMAT',
        grid_resolution=200,  # Increased from default 100 for smoother boundaries
        eps=0.1  # Slight padding
    )

    # Overlay the actual data points
    scatter_train = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
                        c=y_train, edgecolor="k", cmap='viridis', s=50)
    scatter_test = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], 
                        c=y_test, edgecolor="k", cmap='viridis', marker='D', s=75)

    # Create invisible scatter points just for legend
    train_handle = ax.scatter([], [], marker='o', c='gray', edgecolor='k', label='Train')
    test_handle = ax.scatter([], [], marker='D', c='gray', edgecolor='k', label='Test')

    # Legend 1: Groups
    legend1 = ax.legend(*scatter_train.legend_elements(), title="Group", loc='upper left')
    ax.add_artist(legend1)

    # Legend 2: Markers
    ax.legend(handles=[train_handle, test_handle], title="Data", loc='upper right')

    ax.set_title(f'KNN Decision Boundaries (K={optimal_k})')  # Fixed title
    plt.savefig('../images/HW_2/problem_2_knn_decision_boundaries.png', dpi=300)
    plt.show()



# In[140]:


# Binarize labels for multiclass
y_train_bin = label_binarize(y_train, classes=knn_opt.classes_)
y_train_pred_prob = knn_opt.predict_proba(X_train)

if VERBOSE:
    # Plot ROC for each class
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(knn_opt.classes_):
        fpr, tpr, _ = roc_curve(y_train_bin[:, i], y_train_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Group {cls} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Multiclass (One-vs-Rest)')
    plt.legend() 
    plt.savefig('../images/HW_2/problem_2_knn_roc_curve.png', dpi=300)
    plt.show()

    auc_train = roc_auc_score(y_train, y_train_pred_prob, multi_class="ovr", average="macro")
    print("KNN train AUC (macro OVR):", auc_train)
    # y_test_pred_prob


# **e)** Compare the results in (b), (c) and (d). Which classifier would you recommend? Justify your conclusions.
# 
#  We can see that the KNN classifier outperforms the other two classifiers, with the KNN having a misclassification rate for the test data of 8.3\% compared to 16.7\% and 25\% for QDA and LDA respectively. Given the lower test error rate and overall error rate, I would recommend the KNN classifier.
