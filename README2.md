## Data preparation and visualization -  Efektet e përdorimit të telefonit në stresin dhe shëndetin mendor

## Introductory Information

#### University of Prishtina - Faculty of Computer and Software Engineering  
#### Master’s Program in Computer and Software Engineering  
#### Professor: Dr.Sc. Mërgim H. HOTI

---

**Group:** 10  

**Team Members:**  
- Erza Bërbatovci  
- Leotrim Halimi  
- Rinor Ukshini  

---

### Dataset:  
[Global Mobile Phone Addiction Dataset](https://www.kaggle.com/datasets/khushikyad001/global-mobile-phone-addiction-dataset)  
Source: Kaggle — dataset containing mobile phone addiction survey data with demographic and behavioral information.  

---


### FAZA 2:
Detektimi i përjashtuesit.
Mënjanimi i zbulimeve jo të sakta
Eksplorimi i te dhënave: statistika përmbledhëse, multivariante.

# Detektimi i përjashtuesve (Outlier Detection)

Detektimi i outlier-ve është thelbësor për të identifikuar vlerat që devijojnë shumë nga pjesa tjetër e të dhënave dhe që mund të ndikojnë në mënyrë të pabarabartë në analizat statistike dhe modelet. Në këtë projekt janë përdorur katër metoda të ndryshme për një analizë të plotë multivariate:

## 1. Isolation Forest

### Isolation Forest izoloi outlier-ët duke i trajtuar si vlera që ndahen më shpejt në një model pyjor binar.
```python
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_data = df[numeric_columns]

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomali'] = iso_forest.fit_predict(numeric_data)

normal_data = df[df['Anomali'] == 1]
anomalies = df[df['Anomali'] == -1]
```

➤ 1 = normal, -1 = outlier

## 2. DBSCAN Clustering

DBSCAN identifikon outlier-at si pika që nuk përkasin në asnjë kluster.
```python
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)

df['Cluster'] = clusters
  ```

➤ Cluster = -1 konsiderohet outlier

Shumë efikas për detektim të anomalive multivariate

### 3. K-Means Distance Outliers

Outliers identifikohen sipas distancës nga qendrat e klusterëve.
```python
distances = kmeans.transform(scaled_features)
min_distance = np.min(distances, axis=1)

threshold = np.percentile(min_distance, 90)
df['Outlier'] = min_distance > threshold
```

➤ Instancat më larg qendrës së klusterit shënohen si outlier

### 4. Z-Score Statistical Outliers

Z-Score është metodë statistike klasike:
```python
z_scores = np.abs(stats.zscore(data_numeric, nan_policy='omit'))
outliers_mask = (z_scores > 3).any(axis=1)

df_clean = df[~outliers_mask].reset_index(drop=True)
```

➤ |Z| > 3 konsiderohet vlerë ekstreme

➤ Këtu u krijua dataset-i i pastruar df_clean

# Mënjanimi i zbulimeve jo të sakta (Noise Removal)

Dataset-i përmbante zhurmë (noise), të cilat janë trajtuar si vijon:
```python
### 1. Heqja e outlier-ve me Z-Score
df_clean = df[~outliers_mask].reset_index(drop=True)

```
Heq rreshtat ekstreme që prishin statistikat.

### 2. Trajtimi i vlerave të munguar

Kategorike → plotësim me modalen

Numerike → plotësim me medianën
```python
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
```
### 3. Heqja e vlerave negative

Vlerat negative nuk kanë kuptim për kolona të orëve të përdorimit:
```python
df[time_columns] = df[time_columns].clip(lower=0)
```
### 4. Heqja e duplikateve
df = df.drop_duplicates()

# Eksplorimi i të dhënave (EDA)

EDA u krye për të kuptuar shpërndarjen, varësitë dhe strukturën multivariate të dataset-it.

### 1. Statistikat përmbledhëse
```python
numeric_data.describe().transpose()
```

Përfshin:

Mesatare

Medianë

Devijim standard

Percentilat (25%, 50%, 75%)

### 2. Histogramet + Normal Distribution Fit

Për secilën kolonë:
```python
data.hist(bins=30)
mu, std = norm.fit(data)
```

Vizualizim i shpërndarjes dhe tendencës.

### 3. Heatmap i korrelacioneve
```python
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)
```

Shfaq varësitë midis variablave numerikë.

### 4. PCA – Projectimi në 2 dimensione

Për analizë vizuale:
```python
scaled_data = scaler.fit_transform(numeric_data.fillna(0))
pca_result = PCA(n_components=2).fit_transform(scaled_data)

plt.scatter(pca_result[:,0], pca_result[:,1])
```
### 5. Clustering Hierarkik (Dendrogram)
```python
linkage_matrix = linkage(scaled_data, method='ward')
dendrogram(linkage_matrix)
```

Hierarkia e grupimeve vizualizohet në formën e pemës.

# Balancimi i të dhënave (SMOTE)

Dataset-i ishte i pabalancuar për kolonën Stress_Category, prandaj u përdor SMOTE:
```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

Kjo rrit numrin e mostrave minoritare dhe e balancon datasetin.

# Ruajtja e fajllit final të procesuar
```python
df.to_csv('dataset/mobile_addiction_data_processed2.csv', index=False)
```
