#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import warnings


warnings.filterwarnings('ignore')
#%%
file_path = 'suicide_rate.csv'
suicide_data = pd.read_csv(file_path)
print(suicide_data.tail())
print(suicide_data.shape)
#%%
suicide_data = suicide_data.drop_duplicates()
numerical_columns = suicide_data.select_dtypes(include=['float64', 'int64']).columns
suicide_data[numerical_columns] = suicide_data[numerical_columns].interpolate(method='linear', limit_direction='both')
bins = [-0.01, 50, 200, np.inf]
labels = ['Low', 'Medium', 'High']
suicide_data['RiskLevel'] = pd.cut(suicide_data['SuicideCount'], bins=bins, labels=labels, include_lowest=True)
suicide_data['RiskLevel'] = suicide_data['RiskLevel'].replace({'Medium': 'High'})
suicide_data = suicide_data[suicide_data['RiskLevel'] != 'Medium']
print(suicide_data[['SuicideCount', 'RiskLevel']].head())
#%%
# Encode Categorical Variables
categorical_columns = suicide_data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
suicide_data['RiskLevel'] = encoder.fit_transform(suicide_data['RiskLevel'])
for col in categorical_columns:
    suicide_data[col] = encoder.fit_transform(suicide_data[col])

#%%
# Feature Reduction (reducing uneessary features)
suicide_data = suicide_data.drop(columns=['RegionCode'])
suicide_data['EconomicIndex'] = suicide_data[['GDP', 'GrossNationalIncome']].mean(axis=1)
suicide_data = suicide_data.drop(columns=['GDP', 'GrossNationalIncome'])

#%%
X = suicide_data.drop(['RiskLevel','SuicideCount'], axis=1)
y = suicide_data['RiskLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5805, stratify=y)
print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")
#%% VIF
vif_data = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values(by='VIF', ascending=False)
print(vif_data)

#%%
# Random Forest

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

rf = RandomForestRegressor()
rf.fit(X_train, y_train_encoded)
importance = rf.feature_importances_
features = X_train.columns


df = pd.DataFrame({'Feature': features, 'Importance': importance})
df = df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(df['Feature'], df['Importance'], color='green')
plt.gca().invert_yaxis()
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

threshold = 0.01
selected_rf = df[df['Importance'] > threshold]['Feature'].tolist()
eliminated_rf = df[df['Importance'] <= threshold]['Feature'].tolist()
print("RF Selected:", selected_rf)
print("RF Eliminated:", eliminated_rf)

#%%
# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

num_features_95 = np.argmax(cumulative_variance >= 0.95)

print(f'Explained variance by each component: {explained_variance}')
print(f'Number of components chosen: {pca.n_components_}')
print(f"Number of features needed to explain more than 95% of the variance: {num_features_95}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title("Cumulative Explained Variance vs Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axvline(x=num_features_95, color='g', linestyle='-')
plt.text(num_features_95 + 0.5, 0.95, f'{num_features_95} features', color='green')
plt.show()

#%%
# SVD
U, singular_values, Vt = np.linalg.svd(X_scaled, full_matrices=False)
variance_explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
threshold = 0.95
num_components = np.argmax(variance_explained >= threshold) + 1

print(f"Number of components to retain 95% variance: {num_components}")

Vt_df = pd.DataFrame(Vt[:num_components].T, index=X.columns, columns=[f'Singular Vector {i+1}' for i in range(num_components)])

selected_features = set()
num_top_features = 3
for i in range(num_components):
    top_features = Vt_df[f'Singular Vector {i+1}'].abs().nlargest(num_top_features).index
    selected_features.update(top_features)

selected_features = list(selected_features)
print(f"Selected features based on top contributions: {selected_features}")

X_reduced = X[selected_features]
print("Reduced feature dataset:")
print(X_reduced.head().to_string())

#%%
# Step 5: Outlier Detection
iso_forest = IsolationForest(contamination=0.01, random_state=5805)
outliers = iso_forest.fit_predict(X)

X['Outlier'] = outliers
cleaned_data = X[X['Outlier'] == 1].drop(columns=['Outlier']).reset_index(drop=True)
cleaned_data['RiskLevel'] = y.loc[X[X['Outlier'] == 1].index].reset_index(drop=True)
original_size = len(X)
cleaned_size = len(cleaned_data)
removed_size = original_size - cleaned_size

sizes = [cleaned_size, removed_size]
labels = ['Cleaned Data', 'Removed Data']
colors = ['#66c2a5', '#fc8d62']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
plt.title('Percentage of Removed Data vs Cleaned Data')
plt.show()
#%%
# Covariance Matrix Heatmap
cov_matrix = np.cov(X_scaled, rowvar=False)
plt.figure(figsize=(14, 14))
sns.heatmap(cov_matrix, annot=False, cmap="coolwarm", xticklabels=X.columns, yticklabels=X.columns)
plt.title("Covariance Matrix Heatmap")
plt.show()

#%%
# Correlation Matrix Heatmap
corr_matrix = np.corrcoef(X_scaled, rowvar=False)
plt.figure(figsize=(14, 14))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", xticklabels=X.columns, yticklabels=X.columns)
plt.title("Correlation Matrix Heatmap")
plt.show()


#%% phase II
#regression

cleaned_data_adjusted = resample(cleaned_data,
                                 replace=False,
                                 n_samples=100,
                                 random_state=5805)
X_train = cleaned_data_adjusted.drop('RiskLevel', axis=1)
y_train = cleaned_data_adjusted['RiskLevel']
X_test = cleaned_data_adjusted.drop('RiskLevel', axis=1)
y_test = cleaned_data_adjusted['RiskLevel']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()

mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
n, p = X_test.shape
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
aic = ols_model.aic
bic = ols_model.bic

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred_test, label="Predicted", linestyle='dashed', marker='x')
plt.title("Actual vs. Predicted: Test Set")
plt.xlabel("Observations")
plt.ylabel("RiskLevel Encoded")
plt.legend()
plt.show()

t_test_results = ols_model.t_test(np.eye(len(ols_model.params)))
f_test_results = ols_model.f_test(np.identity(len(ols_model.params)))
confidence_intervals = ols_model.conf_int()

print("Regression Model Summary:")
print(ols_model.summary())
print("\nMetrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Adjusted R-squared: {adjusted_r2:.2f}")
print(f"AIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")
print("\nT-Test Results:")
print(t_test_results)
print("\nF-Test Results:")
print(f_test_results)
print("\nConfidence Intervals:")
print(confidence_intervals)

#%%
# backward stepwise regresion
scaler = StandardScaler()

X_train = cleaned_data.drop('RiskLevel', axis=1)
y_train = cleaned_data['RiskLevel']
X_test = cleaned_data.drop('RiskLevel', axis=1)
y_test = cleaned_data['RiskLevel']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

eliminated = []
process = []
def backward_elimination(X, y, threshold=0.01):
    features = list(X.columns)
    while len(features) > 0:
        X_subset = X[features]
        model = sm.OLS(y, sm.add_constant(X_subset)).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        worst = p_values.idxmax()
        process.append({
            "Eliminated Feature": worst if max_p_value > threshold else None,
            "Adjusted R2": model.rsquared_adj,
            "AIC": model.aic,
            "BIC": model.bic,
            "P-value": max_p_value
        })
        if max_p_value > threshold:
            features.remove(worst)
            eliminated.append(worst)
        else:
            break
        table = PrettyTable()
        table.field_names = ["Eliminated Feature", "Adjusted R2", "AIC", "BIC", "P-value"]
        for step in process:
            table.add_row([step["Eliminated Feature"],
                           round(step["Adjusted R2"], 2),
                           round(step["AIC"], 2),
                           round(step["BIC"], 2),
                           round(step["P-value"], 2)])
        print(table)
    return model, features

final_model, selected = backward_elimination(X_train_scaled, y_train)
print("Eliminated Features:", eliminated)
print("Final Selected Features:", selected)
print('Final model', final_model.summary())

#%% final model
X_train_final = X_train_scaled[selected]
X_test_final = X_test_scaled[selected]
final_model = sm.OLS(y_train, sm.add_constant(X_train_final)).fit()

y_pred = final_model.predict(sm.add_constant(X_test_final))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFinal Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {final_model.rsquared_adj:.4f}")

print("\nFinal Model Summary:")
print(final_model.summary())

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", linestyle='dashed', marker='x')
plt.title("Actual vs Predicted: Final Model")
plt.xlabel("Observations")
plt.ylabel("Encoded RiskLevel")
plt.legend()
plt.show()

#%% phase III stratified k-fold crossval
def stratified_kfold_cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Stratified K-Fold Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.2f}")

def evaluate_model(model, X_test, y_test, y_pred, y_proba=None):
    confusion_matrix = pd.crosstab(
        y_test,
        y_pred,
        rownames=['Actual'],
        colnames=['Predicted'],
        dropna=False
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    print(confusion_matrix)

    # Extracting values with safeguards
    TP = confusion_matrix.loc[1, 1] if 1 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
    TN = confusion_matrix.loc[0, 0] if 0 in confusion_matrix.index and 0 in confusion_matrix.columns else 0
    FP = confusion_matrix.loc[0, 1] if 0 in confusion_matrix.index and 1 in confusion_matrix.columns else 0
    FN = confusion_matrix.loc[1, 0] if 1 in confusion_matrix.index and 0 in confusion_matrix.columns else 0

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F = 2 / (1 / recall + 1 / precision) if recall > 0 and precision > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("Specificity:", round(specificity, 2))
    print("F-score:", round(F, 2))


def plot_roc_curves(models, X_test, y_test, model_names):
    plt.figure(figsize=(12, 8))

    for model, name in zip(models, model_names):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.50)")
    plt.title("ROC Curves for Classification Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


#%% phase III
# decision tree classfier pre purning
dt = DecisionTreeClassifier(random_state=5805)
dt.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, dt.predict(X_train))
test_accuracy = accuracy_score(y_test, dt.predict(X_test))

print("Training Accuracy:", round(train_accuracy,2))
print("Test Accuracy:", round(test_accuracy,2))

print("Tree Parameters:", dt.get_params())

plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Low', 'High'])
plt.show()

#%%
# grid search to find best params
dt1 = DecisionTreeClassifier(random_state=5805)
tuned_parameters = [{
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2']
}]

grid_search = GridSearchCV(dt1, tuned_parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:\n", grid_search.best_params_)

best_dt = grid_search.best_estimator_
best_dt.fit(X_train, y_train)

train_accuracy_pruned = accuracy_score(y_train, best_dt.predict(X_train))
test_accuracy_pruned = accuracy_score(y_test, best_dt.predict(X_test))

print("Pruned Tree Training Accuracy:", round(train_accuracy_pruned,2))
print("Pruned Tree Test Accuracy:", round(test_accuracy_pruned,2))

plt.figure(figsize=(20, 10))
plot_tree(best_dt, filled=True, feature_names=X.columns, class_names=['Low', 'High'])
plt.show()

#%%
# dt post-purning
dt2 = DecisionTreeClassifier(random_state=5805)
dt2.fit(X_train, y_train)

path = dt2.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=5805, ccp_alpha=ccp_alpha)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt.predict(X_test)))

optimal_alpha = ccp_alphas[np.argmax(test_scores)] + 0.001

print("Optimal ccp_alpha:", optimal_alpha)

pruned_dt = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimal_alpha, class_weight='balanced')
pruned_dt.fit(X_train, y_train)

train_accuracy_pruned = accuracy_score(y_train, pruned_dt.predict(X_train))
test_accuracy_pruned = accuracy_score(y_test, pruned_dt.predict(X_test))

print("Post-Pruned Tree Training Accuracy:", round(train_accuracy_pruned,2))
print("Post-Pruned Tree Test Accuracy:", round(test_accuracy_pruned,2))

plt.figure(figsize=(20, 10))
plot_tree(pruned_dt, filled=True, feature_names=X.columns, class_names=['Low', 'High'])
plt.title("Pruned Decision Tree")
plt.show()

# evaluating the post pruning tree
y_pred_pruned_dt = pruned_dt.predict(X_test)
y_proba_pruned_dt = pruned_dt.predict_proba(X_test)[:, 1]

print('Decision tree evaluation:')
evaluate_model(pruned_dt, X_test, y_test, y_pred_pruned_dt, y_proba_pruned_dt)
stratified_kfold_cv(pruned_dt, X, y)
plt.title("Pruned Decision Tree")
plt.title("Pruned Decision Tree")
#%% Logistic regression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model_logistic = LogisticRegression(random_state=5805)
model_logistic.fit(X_train_scaled,y_train)
y_pred_logistic = model_logistic.predict(X_test_scaled)
y_pred_logistic_prob= model_logistic.predict_proba(X_test_scaled)[:, 1]

print("Logistic regression evaluation:")
evaluate_model(model_logistic,X_test_scaled,y_test,y_pred_logistic,y_pred_logistic_prob)
stratified_kfold_cv(model_logistic, X, y)
#%% KNN
accuracy_scores = []
k_values = range(2, 31)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

optimal_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print(f"Optimal K: {optimal_k}")

knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train, y_train)

y_pred_knn = knn_final.predict(X_test)
y_proba_knn = knn_final.predict_proba(X_test)[:, 1]

print("KNN Evaluation:")
evaluate_model(knn_final, X_test, y_test, y_pred_knn, y_proba_knn)
stratified_kfold_cv(knn, X, y)

#%% SVM
#svm takes a while to run
cleaned_data_resampled = resample(cleaned_data,
                                 replace=False,
                                 n_samples=60000,
                                 random_state=5805)
X_train_re = cleaned_data_resampled.drop('RiskLevel', axis=1)
y_train_re = cleaned_data_resampled['RiskLevel']
X_test_re = cleaned_data_resampled.drop('RiskLevel', axis=1)
y_test_re = cleaned_data_resampled['RiskLevel']

model_SVC = SVC(kernel='rbf', random_state=5805,probability=True)
#model_SVC.fit(X_train_re,y_train_re)
#y_pred_svm = model_SVC.predict(X_test_re)
#y_proba_svm = model_SVC.predict_proba(X_test_re)[:, 1]

#print("SVM evaluation:")
#evaluate_model(model_SVC, X_test_re, y_test_re, y_pred_svm, y_proba_svm)
#stratified_kfold_cv(model_SVC, X, y)
#%% Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_proba_nb = nb.predict_proba(X_test)[:, 1]

print("Naive Bayes Evaluation:")
evaluate_model(nb, X_test, y_test, y_pred_nb, y_proba_nb)
stratified_kfold_cv(nb, X, y)

#%% neural net
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=5805)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
y_proba_mlp = mlp.predict_proba(X_test)[:, 1]

print("Neural Network (MLP) Evaluation:")
evaluate_model(mlp, X_test, y_test, y_pred_mlp, y_proba_mlp)
stratified_kfold_cv(mlp, X, y)

#%% ROC AUC plot
models = [pruned_dt, model_logistic, knn_final, nb, mlp]
model_names = ['Decision Tree', 'Logsitic Regresion', 'KNN', 'Naive Bayes', 'Neural Net']
plot_roc_curves(models, X_test, y_test, model_names)

#%% phase IV clustering and associate rule mining
X_clustering = cleaned_data.drop(columns=['RiskLevel'], errors='ignore')

silhouette_scores = []
within_cluster_variations = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_clustering)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_clustering, labels))
    within_cluster_variations.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-', label="Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(k_range, within_cluster_variations, marker='o', linestyle='-', label="Within-Cluster Variation")
plt.title("Within-Cluster Variation for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Variation (Inertia)")
plt.grid()
plt.legend()
plt.show()

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans_final.fit(X_clustering)
final_labels = kmeans_final.labels_

cleaned_data['Cluster'] = final_labels
print(cleaned_data[['Cluster']].value_counts())
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add DBSCAN labels to the dataset
cleaned_data['DBSCAN_Cluster'] = dbscan_labels
print(cleaned_data[['DBSCAN_Cluster']].value_counts())

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"Number of clusters found: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise}")
#%%
categorical_data = cleaned_data[['RiskLevel', 'Cluster']].astype(str)
transactions = categorical_data.values.tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("Association Rules:")
print(rules)