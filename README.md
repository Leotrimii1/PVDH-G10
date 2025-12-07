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

## Project Setup

Follow these steps to run the project and the Jupyter Notebook.


###  Create directory
mkdir PVDH-G10
cd PVDH-G10

### Clone the repository
git clone https://github.com/Leotrimii1/PVDH-G10.git


### Create a virtual environment
python3 -m venv ./venv

### Activate the virtual environment
### macOS/Linux
source venv/bin/activate
### Windows (PowerShell)
venv\Scripts\activate

### Install project requirements
pip install -r requirements.txt

#### Open your .ipynb notebook file. When you attempt to run a cell,
#### VS Code will ask you to select a Python environment.
#### Select the interpreter that points to your virtual environment:
#### ./venv/bin/python



---
This project focuses on **data pre-processing for preparing data for analysis**.  
It covers the main stages of cleaning, transforming, and selecting data features before building machine learning models.


The following steps were completed in this project:

 **Data Collection**  

 **Data Type Definition and Quality Check**  
  

 **Data Integration, Aggregation, and Sampling**  
  

 **Data Cleaning and Handling Missing Values**  
   

 **Dimensionality Reduction and Feature Subset Selection**  
   

 **Feature Creation (Feature Engineering)**  
   

 **Discretization and Binarization**  
  

 **Data Transformation**  
   

---
## Technologies Used

- Python 3.12  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

# Pre-procesimi i të dhënave

Pre-procesimi i të dhënave është një hap i domosdoshëm në analizën e të dhënave, sepse të dhënat e papërpunuara zakonisht janë të pasakta, të paplota ose të pa-strukturuara. Ky proces përfshin trajtimin e vlerave të humbura, normalizimin ose standardizimin e vlerave numerike, kodimin e variablave kategorikë dhe eliminimin e anomalive ose outlier-eve. Pa një pre-procesim të kujdesshëm, modelet mund të japin rezultate të pasakta

## FAZA 1

### Libraritë

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

```

### Leximi i datasetit
Për leximin e datasetit është përdorur libraria pandas e cila është e njohur për përpunimin dhe analizën e të dhënave.

```python
df = pd.read_csv('dataset/mobile_addiction_data.csv')
 ```
### Forma e datasetit dhe llojet e atributeve
Ky hap ndihmon për të kuptuar strukturën e datasetit: sa rreshta dhe kolona përmban, si dhe cilat kolona janë numerike dhe cilat kategorike. Njohja e tipit të atributeve është e nevojshme për të zgjedhur metodat e duhura të analizës, vizualizimit dhe modelimit, pasi algoritmet trajtojnë ndryshe të dhënat numerike dhe kategorike.

```python
print("Number of rows::",df.shape[0])
print("Number of columns::",df.shape[1])

categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("Atributet kategorike", categorical_columns)
print("Atributet numerike", numerical_columns)
```

### Formatimi i emrave të kolonave
Formatimi i emrave të kolonave është praktikë e mirë sepse siguron konsistencë dhe shmang gabimet gjatë referimit të kolonave në kod. P.sh., hapësirat, shkronjat e mëdha dhe karakteret special mund të shkaktojnë probleme në funksione dhe metoda të ndryshme. Duke standardizuar emrat, bëhet më e lehtë për t’u përdorur më pas në manipulime dhe analizë.

```python
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
```

### Statistika gjenerale të datasetit
Kjo pjesë jep një përmbledhje të përgjithshme të të dhënave, duke treguar numrin e rreshtave të plotë, mungesat e mundshme, llojet e të dhënave dhe shpërndarjen statistikore të variablave.

```python
print("General Statistics::\n")
print(df.info())

print("Summary Statistics::\n")
print(df.describe(include='all'))
```

### Heqja e kolonave jo të nevojshme
Disa kolona mund të jenë të panevojshme për analizën ose modelimin, pasi nuk ofrojnë informacion të rëndësishëm ose janë identifikues unik (p.sh., user_id) që nuk ndihmon në analiza statistikore. Heqja e tyre ndihmon në thjeshtimin e datasetit.

```python
df = df.drop(columns=[
    'user_id',
    'primary_device_brand',
    'internet_connection_type',
    
], errors='ignore')

df.head()
```

### Gjetja e vlerave të zbrazëta
Ky hap është i domosdoshëm për të identifikuar të dhënat që mungojnë (NaN). Vlerat e zbrazëta mund të ndikojnë në analizat statistike. Duke ditur cilat kolona dhe sa rreshta janë pa vlera, mund të vendosim strategjinë më të përshtatshme për trajtimin e tyre.
```python
print("Columns with Missing Values::", df.columns[df.isnull().any()].tolist())
print("Number of rows with Missing Values::", df.isnull().any(axis=1).sum())
print("Sample Indices with missing data::", df.isnull().any(axis=1).to_numpy().nonzero()[0].tolist()[0:10])
```

### Trajtimi i vlerave të zbrazëta
Për kolonën education_level, vlerat e zbrazëta janë zëvendësuar me modalen (vlerën më të shpeshtë). Arsyeja për këtë është që kjo kolonë përdoret në analizë dhe nuk duam të humbasim rreshta të të dhënave vetëm për disa mungesa. Përdorimi i mode është një metodë e thjeshtë për të plotësuar vlerat kategorike.
```python
mode_value = df['education_level'].mode()[0]
df['education_level'].fillna(mode_value, inplace=True)
print(f"Filled missing education_level with mode: {mode_value}")
```

### Trajtimi i vlerave negative
Disa kolona që përmbajnë të dhëna të matshme si orët e përdorimit të telefonit, aktivitetet fizike apo njoftimet push mund të kenë gabimisht vlera negative, të cilat nuk kanë kuptim në kontekstin real. Duke përdorur clip(lower=0), këto vlera negative zëvendësohen me zero, duke siguruar që analiza dhe statistikat të jenë të sakta dhe të interpretuara drejt.
```python
time_columns = [
    "daily_screen_time_hours",
    "phone_unlocks_per_day",
    "social_media_usage_hours",
    "gaming_usage_hours",
    "streaming_usage_hours",
    "messaging_usage_hours",
    "work_related_usage_hours",
    "sleep_hours",
    "physical_activity_hours",
    "time_spent_with_family_hours",
    "online_shopping_hours",
    "monthly_data_usage_gb",
    "push_notifications_per_day"
]

df[time_columns] = df[time_columns].clip(lower=0)
rows_with_negatives = df[negatives_mask.any(axis=1)]

```

### Të dhënat duplikate
Rreshtat duplikatë mund të shfaqen për shkak të mbledhjes së të dhënave nga burime të ndryshme ose gabimeve gjatë importimit. Rreshtat e njëjtë mund të ndikojnë në shpërndarjet statistike dhe modelimin e të dhënave, duke i bërë rezultatet të pasakta. Identifikimi dhe heqja e tyre siguron integritetin dhe saktësinë e datasetit.
```python
duplicate_rows = df[df.duplicated()]

print(f"Number of duplicate rows found: {duplicate_rows.shape[0]}")


if duplicate_rows.shape[0] > 0:
    df = df.drop_duplicates()
    print(" Duplicate rows removed successfully.")
else:
    print(" No duplicate rows found.")


print(f"New dataset shape: {df.shape}")
```

### One hot encoding për të dhënat kategorike
Shumë kolona kategorike nuk mund të përdoren drejtpërdrejt nga algoritmet numerike të mësimit makinerik. Kodimi i këtyre kolonave (p.sh., education_level, relationship_status, gender, urban_or_rural) me numra ose one-hot encoding transformon variablat kategorikë në format të kuptueshëm për algoritmet, duke ruajtur informacionin origjinal dhe duke mundësuar analizë statistikore dhe modelim.
```python
education_type = { 'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4 , np.nan: -1 }
relationship_status_type = { 'Single': 1, 'In Relationship': 2, 'Married': 3, 'Divorced': 4, np.nan: -1 }
df['education_level'] = df['education_level'].replace("Master's", "Master")
df['education_level_encoded'] = df['education_level'].map(education_type)
df['relationship_status_encoded'] = df['relationship_status'].map(relationship_status_type)

urban_or_rural_type = {
    'Urban': 1,
    'Rural': 2,
    np.nan: -1
}

self_reported_addiction_level_type = {
    'Low': 1,
    'Moderate': 2,
    'High': 3,
    'Severe': 4,
    np.nan: -1
}

gender_type = {
    'Male': 1,
    'Female': 2,
    'Other': 3,
    np.nan: -1
}
```

### Agregimi
Agregimi i të dhënave është një metodë për të përmbledhur informacionin dhe për të nxjerrë statistika mbi grupe të ndryshme. P.sh., klasifikimi i orëve të përdorimit të ekranit në bin-e, krijimi i grupeve sipas moshës dhe analizimi mesatar i stresit, shëndetit mendor apo gjumit, ndihmon në identifikimin e modeleve dhe marrëdhënieve në dataset.
```python
bins = [0, 2, 4, 6, 8, 24]
labels = ['0-2','2-4','4-6','6-8','8+']
df['screen_time_bin'] = pd.cut(df['daily_screen_time_hours'], bins=bins, labels=labels)

df.groupby('screen_time_bin')[['mental_health_score','depression_score','sleep_hours']].mean()
age_bins = [0, 25, 35, 45, 60, 100]
age_labels = ['<25','26-35','36-45','46-60','60+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)


avg_screen_and_stress_by_age_gender = df.groupby(['age_group', 'gender']).agg({
    'daily_screen_time_hours': 'mean',
    'stress_level': 'mean'
}).round(2)

avg_social_and_stress_by_area_education = df.groupby(['urban_or_rural', 'education_level']).agg({
    'social_media_usage_hours': 'mean',
    'stress_level': 'mean'
}).round(2)
mental_health_by_activity = df.groupby(['physical_activity_hours']).agg({
    'mental_health_score': 'mean',
    'depression_score': 'mean',
    'anxiety_score': 'mean',
    'sleep_hours': 'mean'
}).round(2)
```
### Mostrimi i të dhënave
Vizualizimi i shpërndarjes së kolonave numerike është kryer për të përshtatur vlerat e mostres me shpërndarjen e datasetit të plotë. Duke marrë një mostër të vogël (p.sh., 30% të të dhënave), krahasimi i histogramëve me datasetin origjinal siguron që mostrat e përdorura në analiza apo modelim pasqyrojnë drejt shpërndarjen reale. Nëse shpërndarja e mostres nuk përputhet, mund të rritet fraksioni i mostres për të pasqyruar më mirë vlerat origjinale.
```python
sample_data = df.sample(frac=0.3, random_state=42)

numeric_cols = ['age', 'income_usd', 'urban_or_rural', 'stress_level', 'daily_screen_time_hours']

for col in numeric_cols:
    plt.figure(figsize=(10,5))
    sns.histplot(df[col], color='blue', label='Full Dataset', kde=True, stat="density", alpha=0.5)
    sns.histplot(sample_data[col], color='orange', label='Sample', kde=True, stat="density", alpha=0.5)
    plt.title(f'Distribution Comparison: {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.show()
```

<img width="844" height="469" alt="Screenshot 2025-12-07 at 8 14 10 PM" src="https://github.com/user-attachments/assets/ae86c14e-f012-4ac4-99e4-6e636aca4879" />

### Trajtimi i outliers
Outlier-et janë vlera ekstreme që mund të ndikojnë në analizat statistike dhe modelimin e të dhënave. Duke përdorur metodën e IQR (Interquartile Range), vendosen kufijtë e sipërm dhe të poshtëm për të identifikuar vlerat jashtëzakonisht të larta ose të ulëta.
```python
Q1 = df["income_usd"].quantile(0.25)
Q2 = df["income_usd"].quantile(0.50)
Q3 = df["income_usd"].quantile(0.75)
Q4 = df["income_usd"].max()

IQR = Q3 - Q1

print(f"Q1 (25th percentile): {Q1}")
print(f"Q2 (Median): {Q2}")
print(f"Q3 (75th percentile): {Q3}")
print(f"Q4 (Max): {Q4}")
print(f"IQR (Q3 - Q1): {IQR}")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["income_usd"] < lower_bound) | (df["income_usd"] > upper_bound)]

count_above_upper = (df["income_usd"] > upper_bound).sum()
count_below_lower = (df["income_usd"] < lower_bound).sum()
```

### Zgjedhja e nënbashkësive
Duke koduar kolonat kategorike dhe përdorur teknikën e korrelacionit me targetin, si dhe metodat SelectKBest dhe RFE me RandomForest, identifikohen veçoritë më të rëndësishme për parashikimin e nivelit të varësisë.
```python
target_column = "self_reported_addiction_level"
X = df.drop(columns=[target_column, "self_reported_addiction_level_encoded"], errors='ignore')

y = df[target_column]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_encoded = pd.get_dummies(X, drop_first=True)

X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce')
X_encoded = X_encoded.fillna(0)

corr = X_encoded.corrwith(pd.Series(y_encoded))
plt.figure(figsize=(12,5))
corr.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature correlation with target")
plt.show()

selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X_encoded, y_encoded)
top_features_f = X_encoded.columns[selector.get_support()]

model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_encoded, y_encoded)
top_features_rfe = X_encoded.columns[rfe.support_]
```
<img width="832" height="564" alt="Screenshot 2025-12-07 at 8 14 42 PM" src="https://github.com/user-attachments/assets/3e5a74be-c84d-4e42-91fd-e0fa871abdf3" />

### Diskretizimi
Diskretizimi i një kolone numerike, si daily_screen_time_hours, është përdorur për të kategorizuar vlerat në grupe kuptimplote (Low, Medium, High, Very High). Kjo e bën më të lehtë interpretimin e të dhënave dhe analizën krahasuese, sidomos kur vlerat janë shumë të shpërndara dhe ka interes për analiza kategoriale ose vizualizime të përmbledhura.

```python
bins = [0, 2, 5, 8, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High']
df['Screen_Time_Category'] = pd.cut(df['daily_screen_time_hours'], bins=bins, labels=labels)
```

### Binarizimi
Kolonat kategorike, si gender, relationship_status, urban_or_rural dhe has_children, janë transformuar në forma binare (0/1) për t’u përdorur nga algoritmet e mësimit makinerik. Ky proces lejon që modelet numerike të kuptojnë informacionin kategorial pa humbur ndonjë të dhënë të rëndësishme dhe lehtëson analizën statistikore të kombinimeve të ndryshme.
```python
#Binarizimi
binarize_cols = [
    'gender', 'relationship_status', 'urban_or_rural', 'has_children'
]

binarized_dfs = []

for col in binarize_cols:
    if col in df.columns:
        
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)  
        df = pd.concat([df, dummies], axis=1)
        binarized_dfs.append(dummies)
        print(f" U krye binarizimi për kolonën: {col}")
    else:
        print(f"Kolona '{col}' nuk ekziston — u anashkalua.")

if binarized_dfs:
    binarized_result = pd.concat(binarized_dfs, axis=1)
    print("\n--- Kolonat e binarizuara (me 0 dhe 1) ---")
    print(binarized_result.head(5))
else:
    print("\n Asnjë kolonë nuk u binarizua — kontrollo emrat e kolonave.")

df.head()
```

### Krijimi i vetive të reja
Vetitë e reja janë krijuar për të përmbledhur informacionin dhe për të ndihmuar analizat komplekse:
```python
df['Total_Entertainment_Hours'] = (
    df['social_media_usage_hours'] +
    df['gaming_usage_hours'] +
    df['streaming_usage_hours'] +
    df['messaging_usage_hours']
)

# Indeksi i shëndetit mendor
df['Overall_Mental_Health_Index'] = (
    df['mental_health_score'] -
    (df['stress_level'] + df['depression_score'] + df['anxiety_score']) / 3
)
```

### SMOOTHING– ndarja në interval me gjerësi të barabartë për income_usd
Smoothing me ndarje në intervale (bins) përdoret për të grupuar vlerat e shpërndara të income_usd në disa kategori të barabarta numerikisht.
```python

if 'income_usd' in X_encoded.columns:
    print("\nZbutja e 'income_usd' (5 intervale me gjerësi të barabartë)…")
    X_encoded['income_usd_smoothed'], bins = pd.cut(
        X_encoded['income_usd'], bins=5, retbins=True, labels=False
    )
    X_encoded['income_usd_smoothed'] = X_encoded['income_usd_smoothed'].astype(float)
    print(f"   Kufijtë e intervaleve: {bins.round(0)}")
else:
    print("'income_usd' nuk gjendet → anashkalohet zbutja.")
```
### Normalizimi
Normalizimi është procesi i transformimit të vlerave numerike në një shkallë të përbashkët pa ndryshuar shpërndarjen e tyre relative. Kjo është e rëndësishme sepse shumica e algoritmeve të mësimit makinerik (si K-Means, DBSCAN, apo regresionet) janë të ndjeshme ndaj magnitudes së të dhënave dhe kolona me vlera më të mëdha mund të dominojnë rezultatet.

Në kodin e dhënë përdoren tre metoda të normalizimit:

#### Min-Max Scaling – Transformon të dhënat në intervalin [0,1] duke përdorur formulën​
 Kjo metodë ruan shpërndarjen origjinale dhe është e dobishme kur duam që të gjitha kolonat të kenë të njëjtën shkallë.

#### Z-Score Standardization
Shndërron vlerat duke përdorur formulën 

#### Decimal Scaling
Redukton magnitudën e vlerave duke i ndarë me një fuqi të dhjetëshit, në mënyrë që vlerat të kenë një maksimum absolut më të vogël se 1.

Në këtë shembull, normalizimi është demonstruar mbi kolonën income_usd dhe vizualizuar për të krahasuar shpërndarjen origjinale me versionet e normalizuara (Min-Max dhe Z-Score).

```python
X_minmax = X_encoded.copy()
X_zscore = X_encoded.copy()
X_decimal = X_encoded.copy()

v = 73600
col = 'income_usd'

print("\n1. Normalizimi Min-Max  v' = (v-min)/(max-min)")

minmax_ct = ColumnTransformer(
    [(c, MinMaxScaler(), [c]) for c in numerical_cols],
    remainder='passthrough'
)

X_minmax[numerical_cols] = minmax_ct.fit_transform(X_encoded[numerical_cols])

if col in X_encoded.columns:
    min_v = X_encoded[col].min()
    max_v = X_encoded[col].max()
    manual = (v - min_v) / (max_v - min_v)
    print(f"   Manuale  ${v:,} → {manual:.4f}")

    match = X_minmax.loc[X_encoded[col] == v, col]
    if not match.empty:
        print(f"   Sklearn (saktë) ${v:,} → {match.iloc[0]:.4f}")
    else:
        scaler_for_col = minmax_ct.named_transformers_[col]
        approx = scaler_for_col.transform([[v]])[0][0]
        print(f"   Sklearn (përafërsim) ${v:,} → {approx:.4f}")

print("\n2. Normalizimi Z-Score  v' = (v-μ)/σ")

z_ct = ColumnTransformer(
    [(c, StandardScaler(), [c]) for c in numerical_cols],
    remainder='passthrough'
)

X_zscore[numerical_cols] = z_ct.fit_transform(X_encoded[numerical_cols])

if col in X_encoded.columns:
    mu    = X_encoded[col].mean()
    sigma = X_encoded[col].std()
    manual_z = (v - mu) / sigma

    match_z = X_zscore.loc[X_encoded[col] == v, col]
    if not match_z.empty:
        print(f"   Sklearn (saktë) ${v:,} → {match_z.iloc[0]:.4f}")
    else:
        scaler_z = z_ct.named_transformers_[col]
        approx_z = scaler_z.transform([[v]])[0][0]

for c in numerical_cols:
    max_abs = X_encoded[c].abs().max()
    j = 0
    while max_abs >= 1 and j < 20:
        max_abs /= 10
        j += 1
    if j:
        X_decimal[c] = X_encoded[c] / (10 ** j)


X_normalized = X_minmax.copy()

if col in X_encoded.columns:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    X_encoded[col].hist(ax=ax[0], bins=30, color='skyblue', edgecolor='black')
    ax[0].set_title('Të ardhurat origjinale')
    X_minmax[col].hist(ax=ax[1], bins=30, color='lightgreen', edgecolor='black')
    ax[1].set_title('Min-Max [0,1]')
    X_zscore[col].hist(ax=ax[2], bins=30, color='salmon', edgecolor='black')
    ax[2].set_title('Z-Score')
    plt.tight_layout()
    plt.show()

X_encoded = X_normalized.copy()
```
<img width="1105" height="292" alt="Screenshot 2025-12-07 at 8 19 55 PM" src="https://github.com/user-attachments/assets/284b4808-958d-494f-9be1-bd9cd8fe0611" />

### Ruajtja e fajllit pas procesimit

```python
df.to_csv('dataset/mobile_addiction_data_processed.csv', index=False)
```


## FAZA 2:
Detektimi i përjashtuesit.
Mënjanimi i zbulimeve jo të sakta
Eksplorimi i te dhënave: statistika përmbledhëse, multivariante.

### Detektimi i përjashtuesve (Outlier Detection)

Detektimi i outlier-ve është thelbësor për të identifikuar vlerat që devijojnë shumë nga pjesa tjetër e të dhënave dhe që mund të ndikojnë në mënyrë të pabarabartë në analizat statistike dhe modelet. Në këtë projekt janë përdorur katër metoda të ndryshme për një analizë të plotë multivariate:

#### 1. Isolation Forest
 Isolation Forest izoloi outlier-ët duke i trajtuar si vlera që ndahen më shpejt në një model pyjor binar.
```python
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_data = df[numeric_columns]

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomali'] = iso_forest.fit_predict(numeric_data)

normal_data = df[df['Anomali'] == 1]
anomalies = df[df['Anomali'] == -1]
```

➤ 1 = normal, -1 = outlier

#### 2. DBSCAN Clustering

DBSCAN identifikon outlier-at si pika që nuk përkasin në asnjë kluster ku kluster = -1 konsiderohet outlier
```python
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)

df['Cluster'] = clusters
  ```

➤ 

###$ 3. K-Means Distance Outliers
Outliers identifikohen sipas distancës nga qendrat e klusterëve ku instancat më larg qendrës së klusterit shënohen si outlier
```python
distances = kmeans.transform(scaled_features)
min_distance = np.min(distances, axis=1)

threshold = np.percentile(min_distance, 90)
df['Outlier'] = min_distance > threshold
```

###$ 4. Z-Score Statistical Outliers

Z-Score është metodë statistike klasike, |Z| > 3 konsiderohet vlerë ekstreme
```python
z_scores = np.abs(stats.zscore(data_numeric, nan_policy='omit'))
outliers_mask = (z_scores > 3).any(axis=1)

df_clean = df[~outliers_mask].reset_index(drop=True)
```

### Mënjanimi i zbulimeve jo të sakta (Noise Removal)

Dataset-i përmbante zhurmë (noise), ku trajtimi është bërë duke hequr rreshtat ekstrem që prishin statistikat.
```python
### 1. Heqja e outlier-ve me Z-Score
df_clean = df[~outliers_mask].reset_index(drop=True)

```

# Eksplorimi i të dhënave (EDA)

EDA u krye për të kuptuar shpërndarjen, varësitë dhe strukturën multivariate të dataset-it.

### 1. Statistikat përmbledhëse
```python
numeric_data.describe().transpose()
```

Përfshin: Mesatare, Medianë, Devijim standard, Percentilat (25%, 50%, 75%)

### 2. Histogramet + Normal Distribution Fit

Për secilën kolonë:
```python
data.hist(bins=30)
mu, std = norm.fit(data)
```

Vizualizim i shpërndarjes dhe tendencës.

### 3. Heatmap i korrelacioneve
Shfaq varësitë midis variablave numerikë.
```python
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)
```

### 4. PCA – Projectimi në 2 dimensione

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

