#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


# In[2]:


df=pd.read_csv('sepsis.csv')
df


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.isnull().sum()


# In[10]:


df.isnull().sum()[df.isnull().sum()>0]


# In[11]:


for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())


# In[12]:


df.isnull().sum()[df.isnull().sum()>0]


# In[13]:


df.duplicated().sum()


# In[14]:


df[df.duplicated()]


# In[15]:


df['gender'].unique()


# In[16]:


df['gender'] = df['gender'].replace({
    'M': 'Male',
    'F': 'Female',
    'Mael': 'Male'
})


# In[17]:


plt.figure()
sns.countplot(x='sepsis_label', data=df)
plt.title("Sepsis Distribution (0 = No Sepsis, 1 = Sepsis)")
plt.xlabel("Sepsis Label")
plt.ylabel("Count")
plt.show()


# In[18]:


df['insurance'].value_counts()


# In[19]:


df = pd.get_dummies(df, columns=['insurance','gender'], drop_first=True)


# In[20]:


df


# In[21]:


plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), 
            cmap="coolwarm", 
            annot=False, # avoid clutter
            linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[22]:


# Top Correlated Features
corr = df.corr()['sepsis_label'].abs().sort_values(ascending=False)
top_features = corr[1:11]
plt.figure(figsize=(8,8))
plt.pie(top_features, 
        labels=top_features.index, 
        autopct='%1.1f%%', 
        startangle=90)
plt.title("Top 10 Features Correlated with Sepsis")
plt.show()


# In[23]:


cols_to_drop = [
    # leakage
    'sofa_score', 'apache_iv', 'qsofa', 'sirs_criteria','pao2_fio2_ratio',
    'lactate_mmol', 'creatinine', 'ph_arterial','gcs_total'
    
    # treatment
    'vasopressors_flag', 'mechanical_ventilation',
    'vasopressor_dose_mcg_kg_min', 'antibiotics_24h',
    'insulin_infusion_flag', 'fluids_ml_24h', 'sedation_score',
    
    # time/future
    'icu_los_hours', 'icu_admit_time_hour', 'readmission_30day',
]


# In[24]:


df.drop(cols_to_drop, axis=1, inplace=True,errors='ignore')


# In[25]:


df.columns


# In[28]:


# 4. SPLIT FEATURES & TARGET
# split features and target
X = df.drop('sepsis_label', axis=1)
y = df['sepsis_label']

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[29]:


# 5. FEATURE SCALING
# =====================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


# DEFINE ALL ALGORITHMS
# =====================================================
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier())
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier())
    ]),

    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier())
    ]),

    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingClassifier())
    ]),

    "Naive Bayes": Pipeline([
        ("model", GaussianNB())
    ])
}


# In[31]:


# 6. TRAIN ALL MODELS
# =====================================================
results = []

for name, model in models.items():
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

    results.append([name, accuracy, cv_score])


results_df = pd.DataFrame(
    results,
    columns=["Model", "Test Accuracy", "CV Score"]
)

print("\n================ MODEL COMPARISON ================\n")
print(results_df)


# In[32]:


plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Test Accuracy", data=results_df)
plt.title("Test Accuracy Comparison")
plt.xticks(rotation=30)
plt.show()


# In[33]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

tuned_results = []
best_models = {}

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
rf_pred = best_rf.predict(X_test)

tuned_results.append([
    "Random Forest",
    accuracy_score(y_test, rf_pred),
    grid_rf.best_score_,
    grid_rf.best_params_
])

best_models["Random Forest"] = best_rf


# ---------------- Gradient Boosting ----------------
gb = GradientBoostingClassifier(random_state=42)

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

grid_gb = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_gb.fit(X_train, y_train)

best_gb = grid_gb.best_estimator_
gb_pred = best_gb.predict(X_test)

tuned_results.append([
    "Gradient Boosting",
    accuracy_score(y_test, gb_pred),
    grid_gb.best_score_,
    grid_gb.best_params_
])

best_models["Gradient Boosting"] = best_gb


# ---------------- AdaBoost ----------------
ab = AdaBoostClassifier(random_state=42)

ab_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}

grid_ab = GridSearchCV(ab, ab_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_ab.fit(X_train, y_train)

best_ab = grid_ab.best_estimator_
ab_pred = best_ab.predict(X_test)

tuned_results.append([
    "AdaBoost",
    accuracy_score(y_test, ab_pred),
    grid_ab.best_score_,
    grid_ab.best_params_
])

best_models["AdaBoost"] = best_ab


# ---------------- Final Comparison ----------------
tuned_df = pd.DataFrame(
    tuned_results,
    columns=["Model", "Test Accuracy", "Best CV Score", "Best Parameters"]
)

print("\n=========== TUNED MODEL COMPARISON ===========\n")
print(tuned_df.sort_values(by="Test Accuracy", ascending=False))


# In[34]:


plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Best CV Score", data=tuned_df)
plt.title("Tuned Model Comparison")
plt.xticks(rotation=30)
plt.show()


# In[35]:


# 7. SELECT BEST MODEL
# =====================================================

best_name = results_df.iloc[0]["Model"]   # Model with highest Test Accuracy
best_model = models[best_name]            # Get the model object

print("\nBest Model Selected:", best_name)


# In[36]:


# 8. FINAL EVALUATION
# =====================================================

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on Test Set
y_pred = best_model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, best_model.predict(X_train))
test_acc  = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

# Confusion Matrix
print("\n================ CONFUSION MATRIX ================\n")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\n================ CLASSIFICATION REPORT ================\n")
print(classification_report(y_test, y_pred))


# In[37]:


# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm")
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[38]:


from sklearn.metrics import *
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability for positive class
else:
    y_prob = best_model.decision_function(X_test)    # For models like SVM

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='blue', linewidth=2)
plt.plot([0,1], [0,1], 'k--', linewidth=1)  # Diagonal line
plt.title(f"ROC Curve - {best_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# In[39]:


import joblib
joblib.dump(best_model, "sepsis_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved successfully")


# In[40]:


import joblib

# remove target
X = df.drop("sepsis_label", axis=1)

# save columns
joblib.dump(X.columns, "columns.pkl")

print("Columns saved")


# In[41]:


import os
print(os.getcwd())


# In[ ]:




