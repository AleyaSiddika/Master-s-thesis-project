import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_curve, auc)
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('data.csv')

factors = {
    'Mental_Stress': ['MS1','MS2','MS3','MS4'],
    'Physical_Health': ['PH1','PH2','PH3','PH4'],
    'Time_Management': ['TM1','TM2','TM3'],
    'Technological_Impact': ['TI1','TI2','TI3','TI4'],
    'Emotional_Aspect': ['EA1','EA2','EA3','EA4'],
    'Effort': ['ET1','ET2','ET3'],
    'Performance_Concern': ['PC1','PC2','PC3','PC4'],
    'Frustration': ['FT1','FT2','FT3','FT4','FT5'],
    'Social_Environment': ['SE1','SE2'],
    'Overall_Mental_Workload': ['OM1','OM2','OM3','OM4','OM5','OM6']
}

for factor, questions in factors.items():
    data[factor] = data[questions].mean(axis=1)

factor_columns = list(factors.keys())
data_factors = data[factor_columns]
low_threshold = 2.5
high_threshold = 3.6

def classify_workload(x):
    if x <= low_threshold:
        return 'Low'
    elif low_threshold < x <= high_threshold:
        return 'Medium'
    else:
        return 'High'

data['Workload_Label_Ternary'] = data['Overall_Mental_Workload'].apply(classify_workload)

label_mapping_ternary = {'Low': 0, 'Medium': 1, 'High': 2}
data['Workload_Label_Ternary_Num'] = data['Workload_Label_Ternary'].map(label_mapping_ternary)

X = data[factor_columns]
y = data['Workload_Label_Ternary_Num']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=factor_columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


svm_model = SVC(probability=True)

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=param_grid_svm,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search_svm.fit(X_train, y_train)
y_pred_svm = grid_search_svm.best_estimator_.predict(X_test)


def evaluate_model(y_test, y_pred, model_name):

    unique_classes = np.unique(y_test)
    target_names = [str(cls) for cls in unique_classes]

    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    if model_name == 'SVM':
        y_prob = svm_model.predict_proba(X_test)

    for i in range(len(unique_classes)):
        fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


evaluate_model(y_test, y_pred_svm, 'SVM')


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_svm = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='f1_macro') 
metrics = pd.DataFrame(
    {
        "Model": ["SVM"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_svm),
        ],
        "Precision": [
            precision_score(y_test, y_pred_svm, average="macro"),
        ],
        "Recall": [
            recall_score(y_test, y_pred_svm, average="macro"),
        ],
        "F1 Score": [
            f1_score(y_test, y_pred_svm, average="macro"),
        ],
    }
)


from sklearn.inspection import permutation_importance

perm_importance_svm = permutation_importance(
    svm_model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_macro"
)

importance_df_svm = pd.DataFrame(
    {
        "Factor": X_test.columns,
        "Importance": perm_importance_svm.importances_mean,
    }
)

importance_df_svm["Importance"] = importance_df_svm["Importance"].abs()

importance_df_svm = importance_df_svm.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(
    x="Importance", y="Factor", data=importance_df_svm, palette="viridis"
)

# Add value labels on bars
for p in bar_plot.patches:
    width = p.get_width()
    bar_plot.text(
        width + 0.02,
        p.get_y() + p.get_height() / 2,
        f"{width:.4f}",
        ha="center",
        va="center",
        color="black",
    )

plt.title("SVM Feature Importance (Permutation Importance)", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Factor", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'MLP Classifier': MLPClassifier(max_iter=10000)
}

results = {}

for model_name, model in models.items():
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall,
        'Macro F1 Score': macro_f1
    }

results_df = pd.DataFrame(results).T

results_df.plot(kind='bar', figsize=(8, 6))
plt.title('Comparative Performance of Models')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()


custom_palette = {'Low': '#77DD77', 
                  'Medium': '#89CFF0',  
                  'High': '#FF6961'}

plt.figure(figsize=(8, 6))
sns.countplot(x='Workload_Label_Ternary', data=data, palette=custom_palette)
plt.title('Distribution of Workload Classification (Low, Medium, High)')
plt.xlabel('Workload Classification')
plt.ylabel('Count')

plt.show()


age_mapping = {
    1: '18-24 years old',
    2: '25-34 years old',
    3: '35-44 years old',
    4: '45-54 years old',
    5: '55-64 years old'
}

data['Age_Label'] = data['Age'].map(age_mapping)

import matplotlib.pyplot as plt
import seaborn as sns

workload_palette = {
    'Low': '#77DD77',    
    'Medium': '#89CFF0', 
    'High': '#FF6961' 
}

plt.figure(figsize=(8,6))
sns.countplot(data=data, x='Age_Label', hue='Workload_Label_Ternary', palette=workload_palette)

plt.title('Distribution of Mental Workload by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Mental Workload', title_fontsize='13', fontsize='11', loc='upper right')
plt.xticks()  
plt.show()


for factor in factors.keys():
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=data,
        x="Workload_Label_Ternary",
        y=factor,
        inner="quartile",
        palette=custom_palette,
        linewidth=1.5,
    )

    sns.boxplot(
        data=data,
        x="Workload_Label_Ternary",
        y=factor,
        whis=1.5,
        width=0.01,
        boxprops={"zorder": 2, "alpha": 0.6, "color": "#fff"}, 
        showcaps=False,
        whiskerprops={"linewidth": 1.5, "color": "#aaa"}, 
        medianprops={"color": "red", "linewidth": 2}, 
        showfliers=False,
    )

    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    sns.despine(left=False, bottom=False)

    plt.title(
        f"{factor} Distribution by Workload Classification",
        fontsize=14,
    )
    plt.xlabel("Workload Classification", fontsize=14)
    plt.ylabel(factor, fontsize=14)

    plt.show()
    

import scipy.stats as stats


independent_factors = [
    'Mental_Stress', 'Physical_Health', 'Time_Management', 'Technological_Impact',
    'Emotional_Aspect', 'Effort', 'Performance_Concern', 'Frustration', 'Social_Environment'
]


results = pd.DataFrame(columns=['Independent Factor', 'Normality Test Statistic', 'Normality P-value', 
                                'Kruskal-Wallis Statistic', 'Kruskal-Wallis P-value'])


alpha = 0.05


for factor in independent_factors:
    
    shapiro_stat, shapiro_p = stats.shapiro(data_factors[factor])
    
 
    grouped = pd.qcut(data_factors[factor], q=4, labels=False, duplicates='drop')  
   
    unique_groups = grouped.nunique()
    
    if unique_groups > 1:  
        kruskal_stat, kruskal_p = stats.kruskal(*[data_factors['Overall_Mental_Workload'][grouped == i] for i in range(unique_groups)])
    else:
        kruskal_stat, kruskal_p = (None, None)  

    new_row = {
        'Independent Factor': factor,
        'Normality Test Statistic': round(shapiro_stat, 3),
        'Normality P-value': format(shapiro_p, '.10f'),  
        'Kruskal-Wallis Statistic': round(kruskal_stat, 3) if kruskal_stat is not None else 'N/A',
        'Kruskal-Wallis P-value': format(kruskal_p, '.10f') if kruskal_p is not None else 'N/A'  
    }
    
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

print(results)

significant_factors = results[results['Kruskal-Wallis P-value'].apply(lambda x: float(x) if x != 'N/A' else 1) < alpha]
print("\nSignificant Factors (p < 0.05):")
print(significant_factors)




