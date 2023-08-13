'''
To complete this task i divided the task in 7 setps:

1. Load the data.
2. preprocess of the data: Missing values, Encoding Categorical Variables and Feature Scaling.
3. Handling Class Imbalance
4. Data Splitting
5. Model Training
6. Model Evaluation
7. Model Packaging

'''


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from joblib import dump, load
import pickle


# ------------------- step 1: Load data ---------------------
file_path = 'creditcard.csv'

df = pd.read_csv(file_path)


# ------------------- step 2: Preprocessing  ------------------


# Check is there is missing values
missing_values = df.isnull().sum()
print(missing_values)

# If there are missing values, we can choose to either delete them or impute them based on the needs
# In this example for simplicity the values are deleted.
df.dropna(inplace=True)

# Scale the 'Time' and 'Amount' features
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

X = df.drop('Class', axis=1)
y = df['Class']


# -------------- step 3: Handling Class Imbalance -------------------

# classes separtion
df_majority = df[df.Class==0]
df_minority = df[df.Class==1]

# Undersample the majority class
df_majority_downsampled = resample(df_majority,
                                 replace=False,
                                 n_samples=len(df_minority),
                                 random_state=42)

# Combine the minor class with the under-sampled class majority
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# New class counts
print(df_downsampled.Class.value_counts())

# Undersample the minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

# Combine the mayority class with the under-sampled class minority
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# New class counts
print(df_upsampled.Class.value_counts())


# SMOTE Apply
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# See class counts
print(y_res.value_counts())


# ------------------ step 4: Data Splitting ----------------

# Separate the features and the label.
X = df.drop('Class', axis=1)
y = df['Class']

# Split in training and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# The 'stratify=y' option ensures that the split maintains the same proportion of the minority class in both sets.

print('Training dimensions group:', X_train.shape)
print('Test dimensions group:', X_test.shape)


# --------------------- step 5: Model Training -----------------------

model = LogisticRegression(class_weight='balanced', random_state=42)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
print('Training group score:', train_score)

# Save the model
joblib.dump(model, 'logistic_regression_model.pkl')

# ------------------ step 6: Model Evaluation ---------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Precision: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Score F1: {f1}')
print(f'Area under the ROC curve: {roc_auc}')


# ------------------- step 7: Model Packaging --------------


dump(model, 'credit_card_fraud_model.joblib')

model = load('credit_card_fraud_model.joblib')

with open('credit_card_fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('credit_card_fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)
