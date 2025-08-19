#%% phase I
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot
import warnings
warnings.filterwarnings('ignore')

file_path = 'suicide_rate.csv'
data = pd.read_csv(file_path)
data = data.drop_duplicates()
data['SuicideRatePer100k'] = data['SuicideCount'] / data['Population'] * 100000
data = data.dropna()
numeric_cols = ['GDP', 'GDPPerCapita', 'GNIPerCapita', 'SuicideRatePer100k', 'DeathRatePer100K']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#%% Line plot 1
print("Line plot of suicide rates over time by region.")
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Year", y="SuicideRatePer100k", hue="RegionName", ci=None)
plt.title("Suicide Rates Per 100k Population Over Time by Region", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Suicide Rate Per 100k", fontsize=12)
plt.legend(title="Region", fontsize=10)
plt.grid(True)
plt.show()
print("This line plot shows how suicide rates vary over time by region. Europe stands out with consistently higher rates, while other regions show more stable trends.")

#%% Bar plot
print("Bar plot showing suicide rates by age group and sex.")
plt.figure(figsize=(14, 6))
sns.barplot(data=data, x="AgeGroup", y="SuicideRatePer100k", hue="Sex", ci=None)
plt.title("Suicide Rates by Age Group and Sex", fontsize=14)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Suicide Rate Per 100k", fontsize=12)
plt.legend(title="Sex", fontsize=10)
plt.grid(True)
plt.show()
print("Middle-aged males exhibit the highest suicide rates across all age groups, indicating a vulnerable demographic requiring focused interventions.")

#%% Pie chart
print("Pie chart showing data distribution by region.")
data['RegionName'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), colors=sns.color_palette("pastel"))
plt.title("Data Distribution by Region", fontsize=14)
plt.ylabel('')
plt.show()
print("The pie chart shows that Europe and Asia have the highest representation in the dataset, which might influence overall trends.")

#%% Dist plot
print("Distribution of suicide rates per 100k population.")
sns.displot(data['SuicideRatePer100k'], kde=True, height=6, aspect=1.5)
plt.title("Distribution of Suicide Rates Per 100k Population", fontsize=14)
plt.xlabel("Suicide Rate Per 100k", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.show()
print("Most suicide rates cluster at lower values, but there are significant outliers, highlighting regions or countries with extreme rates.")

#%% Pair plot
print("Plot: Pair plot showing relationships between socioeconomic factors and suicide rates.")
sns.pairplot(data[['GDP', 'GDPPerCapita', 'SuicideRatePer100k']], diag_kind="kde")
plt.show()
print("The pair plot reveals weak relationships between GDP-related factors and suicide rates, suggesting that economic factors alone don't fully explain suicide trends.")

#%% Heatmap
print("Heatmap showing correlations between socioeconomic factors and suicide rates.")
plt.figure(figsize=(10, 8))
corr_matrix = data[['GDP', 'GDPPerCapita', 'SuicideRatePer100k']].corr()
sns.heatmap(corr_matrix, annot=True, cbar=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Socioeconomic Factors", fontsize=14)
plt.show()
print("The heatmap indicates weak correlations between GDP-related factors and suicide rates, with the highest being between GDP and GDP per capita.")

#%% Histogram with KDE
print("Histogram with KDE showing the distribution of suicide rates.")
sns.histplot(data['SuicideRatePer100k'], kde=True, bins=20, color="blue")
plt.title("Histogram with KDE of Suicide Rates", fontsize=14)
plt.xlabel("Suicide Rate Per 100k", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.show()
print("The distribution shows a heavy tail on the right, indicating a few countries with exceptionally high suicide rates.")

#%% QQ-plot
print("QQ-plot of suicide rates to check normality.")
probplot(data['SuicideRatePer100k'].dropna(), dist="norm", plot=plt)
plt.title("QQ-Plot of Suicide Rates", fontsize=14)
plt.grid(True)
plt.show()
print("The QQ-plot shows significant deviations from normality, especially in the tail regions, reflecting the skewed nature of the data.")

#%% KDE plot
print("KDE plot showing the density relationship between GDP and suicide rates.")
sns.kdeplot(data=data, x="GDP", y="SuicideRatePer100k", fill=True, alpha=0.6, linewidth=1.5, cmap="coolwarm")
plt.title("KDE Plot of GDP vs Suicide Rates", fontsize=14)
plt.xlabel("GDP", fontsize=12)
plt.ylabel("Suicide Rate Per 100k", fontsize=12)
plt.grid(True)
plt.show()
print("This KDE plot shows areas of high density where suicide rates and GDP cluster, indicating potential economic thresholds.")

#%% Regression plot
print("Plot: Regression plot showing the relationship between GDP and suicide rates.")
sns.regplot(data=data, x="GDP", y="SuicideRatePer100k", scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Regression Plot of GDP vs Suicide Rates", fontsize=14)
plt.xlabel("GDP", fontsize=12)
plt.ylabel("Suicide Rate Per 100k", fontsize=12)
plt.grid(True)
plt.show()
print("The regression plot confirms a weak positive correlation between GDP and suicide rates, with significant variability.")

#%%
# Box plot
# Suicide rates by age group and sex
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x="AgeGroup", y="SuicideCount", hue="Sex")
plt.title("Suicide Counts by Age Group and Sex")
plt.ylabel("Suicide Count")
plt.xlabel("Age Group")
plt.xticks(rotation=45)
plt.show()
#%%
# Violin plot
sns.violinplot(data=data, x="Sex", y="SuicideRatePer100k", hue="RegionName", split=True)
plt.title("Violin Plot of Suicide Rates by Sex and Region")
plt.show()

#%%
# Joint plot
sns.jointplot(data=data, x="GDP", y="SuicideRatePer100k", kind="kde", fill=True)
plt.title("Joint KDE Plot of GDP vs Suicide Rates")
plt.show()

#%%
# Rug plot
sns.rugplot(data=data, x="SuicideRatePer100k", height=0.1)
plt.title("Rug Plot of Suicide Rates")
plt.show()
#%%
# Area plot
data.groupby("Year")['SuicideRatePer100k'].mean().plot(kind='area', figsize=(10, 6))
plt.title("Area Plot of Suicide Rates Over Time")
plt.show()

#%% 3D Scatter Plot
print("3D scatter plot of GDP, GNI per capita, and suicide rates.")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['GDP'], data['GNIPerCapita'], data['SuicideRatePer100k'], c=data['SuicideRatePer100k'], cmap='viridis', alpha=0.6)
ax.set_xlabel("GDP")
ax.set_ylabel("GNI Per Capita")
ax.set_zlabel("Suicide Rate Per 100k")
plt.colorbar(scatter, ax=ax, shrink=0.5)
plt.title("3D Scatter Plot: GDP, GNI Per Capita, Suicide Rates", fontsize=14)
plt.show()
print("The 3D plot provides a spatial representation of economic factors and their relation to suicide rates, revealing clusters and outliers.")

#%% Cluster map
print("Cluster map of socioeconomic factors and suicide rates.")
sns.clustermap(data[['GDP', 'GDPPerCapita', 'SuicideRatePer100k']].corr(), cmap="coolwarm", annot=True, figsize=(10, 10))
plt.title("Cluster Map of Socioeconomic Factors", fontsize=14)
plt.show()
print("The cluster map highlights weak correlations between GDP-related factors and suicide rates, suggesting other variables like mental health support may play a stronger role.")

#%%
# Hexbin plot
data.plot.hexbin(x='GDP', y='SuicideRatePer100k', gridsize=30, cmap='Blues')
plt.title("Hexbin Plot of GDP vs Suicide Rates")
plt.show()

#%%
# Strip plot
sns.stripplot(data=data, x="Sex", y="SuicideRatePer100k", hue="RegionName", jitter=True, alpha=0.5)
plt.title("Strip Plot of Suicide Rates by Sex and Region")
plt.show()

#%%
# Swarm plot
sns.swarmplot(data=data, x="Sex", y="SuicideRatePer100k", hue="RegionName", alpha=0.5)
plt.title("Swarm Plot of Suicide Rates by Sex and Region")
plt.show()

#%%
# Exploring the relationship between GDP and suicide counts
# Scatterplot for GDP vs Suicide Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="GDP", y="SuicideCount", alpha=0.6, hue="RegionName")
plt.title("Correlation Between GDP and Suicide Counts")
plt.xlabel("GDP")
plt.ylabel("Suicide Count")
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%% Subplot: Regional suicide rates across years
print("Regional suicide rates across years using bar plots.")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Regional Suicide Rates", fontsize=16)

# Bar plot 1: Average suicide rates by region
region_avg = data.groupby("RegionName")["SuicideRatePer100k"].mean().sort_values()
region_avg.plot(kind="bar", ax=axes[0, 0], color="lightcoral", edgecolor="black")
axes[0, 0].set_title("Average Suicide Rates by Region", fontsize=14)
axes[0, 0].set_xlabel("Region", fontsize=12)
axes[0, 0].set_ylabel("Suicide Rate Per 100k", fontsize=12)
axes[0, 0].grid(True)
# Bar plot 2: Stacked bar plot
region_year = data.groupby(["RegionName", "Year"])["SuicideRatePer100k"].mean().unstack()
region_year.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="viridis", edgecolor="black")
axes[0, 1].set_title("Stacked Suicide Rates by Region and Year", fontsize=14)
axes[0, 1].set_xlabel("Region", fontsize=12)
axes[0, 1].set_ylabel("Suicide Rate Per 100k", fontsize=12)
axes[0, 1].grid(True)

# Pie chart
data['RegionName'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1, 0], colors=sns.color_palette("pastel"))
axes[1, 0].set_title("Record Distribution by Region", fontsize=14)
axes[1, 0].set_ylabel("")
# Bar plot 3: Gender comparison
sns.barplot(data=data, x="Sex", y="SuicideRatePer100k", ax=axes[1, 1], palette="muted", ci=None)
axes[1, 1].set_title("Suicide Rates by Gender", fontsize=14)
axes[1, 1].set_xlabel("Gender", fontsize=12)
axes[1, 1].set_ylabel("Suicide Rate Per 100k", fontsize=12)
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
print("The subplots show the global picture of suicide rates:")
print("- Europe consistently has higher rates compared to other regions (subplot 1).")
print("- Suicide rates have fluctuated across years, with noticeable peaks in certain regions (subplot 2).")
print("- Record distribution highlights data dominance from specific regions (subplot 3).")
print("- Gender disparities show males have significantly higher rates than females globally (subplot 4).")