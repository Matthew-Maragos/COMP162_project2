Matthew Maragos

# Introduction

**Link:**<br /> https://www.kaggle.com/datasets/atharvasoundankar/global-space-exploration-dataset-2000-2025?resource=download

**Description:**<br />
This dataset contains 3,000 entries and 12 columns, each representing characteristics of space missions conducted globally from 2000 - 2025.

**Key Variables:**

- **Country** – The country responsible for the mission.
- **Year** – The launch year of the mission.
- **Mission Name** – Name or label of the mission.
- **Mission Type** – Whether the mission is Manned or Unmanned.
- **Launch Site** – The location from which the mission was launched.
- **Satellite Type** – The type of satellite used (e.g., Communication, Research).
- **Budget (in Billion $)** – The budget allocated for the mission.
- **Success Rate (%)** – A percentage value representing mission success.
- **Technology Used** – The main technology or propulsion system used.
- **Environmental Impact** – Level of environmental impact (Low, Medium, High).
- **Collaborating Countries** – Countries that collaborated on the mission.
- **Duration (in Days)** – Total mission duration in days.

**Data Preparation**<br />
As of now, no modifications have been made to the dataset.

# Hypothesis Generation and Exploration

**Questions:**<br />
1. **Does a higher budget correlate with higher mission success rates?**
2. **How has the distribution of manned vs unmanned missions changed over the years?**

**This is where all the imports will be:**


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
```


```python
Space = pd.read_csv("Global_Space_Exploration_Dataset.csv")
```


```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=Space, x='Budget (in Billion $)', y='Success Rate (%)', hue='Environmental Impact', alpha=0.7)
plt.title('Budget vs. Success Rate by Environmental Impact')
plt.xlabel('Budget (in Billion $)')
plt.ylabel('Success Rate (%)')
plt.legend(title='Environmental Impact')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_8_0.png)
    


## Budget vs. Success Rate by Environmental Impact Analysis

The scatter plot shows that the success rate of a mission isn't determined by the budget.  It also shows that any mission(low budget, low success, and etc.) can be impactful to the environment.  So, there are more factors into these missions outside of budget that impacts the success rate and the environmental impact they have.


```python
plt.figure(figsize=(14, 6))
sns.countplot(data=Space, x='Year', hue='Mission Type')
plt.title('Number of Missions per Year by Mission Type')
plt.xlabel('Year')
plt.ylabel('Number of Missions')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Mission Type')
plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    


## Number of Missions per Year by Mission Type Analysis

This plot illustrates how the number of manned and unmanned missions fluctuate over the years.  Overall, manned missions appear to be dominant more frequently than unmanned missions.  It seems that in the early 2000s that unmanned missions were dominant, suggesting a growing of reliance of robotics and satellites during those times.  However, manned missions become dominant pass 2010, which could correspond to significant international launches or space station activity.

# Machine Learning

### Regression: Predicting Missions Success Rate <b />
I'll be using  a Random Forest Regressor to predict the Success Rate (%) of a mission based on other features in this dataset.

 ### Target Varaible (Y):
 - Success Rate (%)

### Features (X): 
- Budget (in Billion $)
- Duration (in Days)
- Environmental Impact (encoded)
- Satellite Type, Mission Type, and Technology Used (one-hot encoded)


```python
# Selecting revelant columns
regression_Space = Space[['Budget (in Billion $)', 'Duration (in Days)', 'Environmental Impact',
                    'Satellite Type', 'Mission Type', 'Technology Used', 'Success Rate (%)']].dropna()

# Encode categorical variables
regression_Space = pd.get_dummies(regression_Space, columns=['Environmental Impact', 'Satellite Type',
                                                        'Mission Type', 'Technology Used'])
```


```python
# Split into features and target
X = regression_Space.drop('Success Rate (%)', axis=1)
y = regression_Space['Success Rate (%)']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Training a Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Predicting and evaluating
y_pred = rf_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R2 score =", r2)
print("RMSE =", rmse)
```

    R2 score = -0.08161833800900298
    RMSE = 15.812245518373834



```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
plt.title('Actual vs. Predicted Success Rate')
plt.xlabel('Actual Success Rate (%)')
plt.ylabel('Predicted Success Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_21_0.png)
    


### Regression Evaluation
- **R2 Score:** -0.082
- **RMSE:** 15.812

**Interpretation**<br />
Looking at our R2 score and RMSE, the Random Forest Regressor performs poorly on this task.

- A negative R2 score means that the model performs worse than simply predicting the mean of the target variable for all observations.
- The high RMSE suggests that the prediction errors are large.<br />

This likely means that the model is underfitting the data - it's not capturing the underlying relationship between the features and the success rate.  I believe that the main reason for this is that these features don't have a strong relationship with Success Rate (%).  I have an example of this in the first graph of Budget vs. Success Rate by Environmental Impact, where I concluded that Budget didn't have a linear relationship with Success Rate (%).<br />
<br />
Actual vs. Predicted Success Rate shows how inaccurate the prediction was. There are low amounts of actual correct predictions, however, they are outweighed by the majority of wrong predictions.<br />
<br />
I will conclude this evaluation by saying that I don't recommend using this model for future prediction.  The fact that none of the features contribute to Success Rate (%) make it hard to use any type of regression model.

### Classification: Predicting Mission Type 
I will be using a Random Forest Classifier to predict whether a mission is going to be manned or unmanned.

### Target Variable (Y)
- Mission Type (Binary: Manned / Unmanned)

### Features (X)
- Budget (in Billion $)
- Duration (in Days)
- Environmental Impact (endcoded)
- Satellite Type, Technology Used (one-hot encoded)


```python
# Filter and encode
clf_Space = Space[["Budget (in Billion $)", "Duration (in Days)", "Environmental Impact", 
                   "Satellite Type", "Technology Used", "Mission Type"]].dropna()

# Encode categorical variables
clf_Space = pd.get_dummies(clf_Space, columns=["Environmental Impact", "Satellite Type", "Technology Used"])

# Label encode the target
le = LabelEncoder()
# 0 = Manned | 1 = Unmanned
clf_Space["Mission Type"] = le.fit_transform(clf_Space["Mission Type"])
#print(clf_Space.columns.tolist())
```


```python
# Splitting the features and target
X_clf = clf_Space.drop("Mission Type", axis=1)
y_clf = clf_Space["Mission Type"]

# Training, Testing, Splitting
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.5, random_state=356)
```


```python
# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=356)
rf_clf.fit(Xc_train, yc_train)

# Predict and evaluate
yc_pred = rf_clf.predict(Xc_test)
acc = accuracy_score(yc_test, yc_pred)
cm = confusion_matrix(yc_test, yc_pred)

acc, cm
```




    (0.506,
     array([[423, 344],
            [397, 336]]))




```python
# Plotting the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot()
plt.title("Confusion Matrix: Mission Type Prediction")
plt.show()
```


    
![png](output_29_0.png)
    


### Classification Interpretation
- **acc** = 0.506
- **cm** = the chart above

**Interpretation**<br />
The model was able to predict about 50.6% (accuracy score = acc = 0.506) correctly, which is not very exciting.  This shows that the confusion matrix shows a fairly balanced distribution of false positives and false negatives, meaning that the model isn't clearly biased toward one class.  This suggests that the model is underfitting, meaning (as said in the Regression) that the model doesn't have that good of data from the features to make accurate predicitons.

In conclusion, Random Forest is not a reliable prediction model for this dataset.  The main problem being that the features in this data don't contribute much to distinguish between manned and unmanned missions.

# Ethical Considerations

### Respect for Persons:
- Dataset appears to be focused on missions and technologies, nothing in here includes any personal or identifiable information about individuals.
- Thus, there are no direct privacy concerns or risks to indiviual autonomy present in this data.

### Beneficence
- Space exploration can bring substantial benefits to humanity, however analyzing and interpreting data on missions types and technologies should be done with care to avoid promoting harmul technologies (such as weaponization or spy satellites).

### Respect for Law and Public Interest
- All information in the dataset seems to be publicly available and doesn't violate intellectual property or national security.
- This dataset demonstrates how reponsible sharing of findings can promote transparency and public understanding of space missions aligns with this principle.

# Final Conclusions

Overall, I think this was an interesting dataset to go through. Finding if the budget of the mission actually impacted the success rate of missions is an interesting idea at first, but looking at the scatter plot I made and thinking of outside possibilities that can affect mission success, it makes sense why the budget didn't really matter — there are too many outside occurrences that can happen and affect the mission itself.<br />

Then, seeing if any of the other variables affected the mission type is also a very interesting idea. Again, just like mission success, there are too many other variables that are not in this data that affect whether or not we man or unman the mission itself. Since all the variables didn't really affect each other, that just meant predicting was going to be messy.<br />

And behold, concluding that both the regression and classification models used didn't predict well. So, even though this dataset was interesting at first, the more and more I analyzed it, the more and more I realized how kinda useless analyzing this was.
