 # === Import libraries ===
import os
import gdown
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_url = 'https://drive.google.com/uc?export=download&id=1t7lLZZlC_FpveffIDz5tqAdpbawMfM4X'
df1_url = "https://drive.google.com/uc?export=download&id=1M90PGon2io8Bx9NusCvXP1AvFb1X2_yN"
df_weather = "https://drive.google.com/uc?export=download&id=1LpU30HmDTwFyDhJe8CmO8PjHjgkk7-PB"

# === Load data ===
df = pd.read_csv(df_url)
df1 = pd.read_csv(df1_url)
df_weather = pd.read_csv(df_weather)


st.title('Projet de 1.88 millions de feux aux USA entre 1992 à 2015')
st.sidebar.title('Sommaire')
pages = ['Introduction','Exploration','Nettoyage et enrichessment des donées','Data Visualisation','Modèle de prédiction']
page = st.sidebar.radio('Aller vers', pages)
st.sidebar.markdown("Auteurs")
st.sidebar.markdown("Achraf BELKIYAL")
st.sidebar.markdown("Timothy RUSSAC")
st.sidebar.markdown("Julia AUMONT")



if page == pages[0]:
    st.write('### Introduction')
    st.image('https://images.unsplash.com/photo-1634009653379-a97409ee15de?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')

    st.markdown(
    """
    Ce projet a été réalisé dans le cadre de la formation Data Analyst de DataScientest, dans le but de mettre en pratique les compétences acquises en analyse de données, visualisation et machine learning.

    🎯 Objectif principal
       Étudier un jeu de données réel, issu de Kaggle :
       le dataset 1.88 Million US Wildfires, qui recense 1,88 million de feux de forêts survenus aux États-Unis entre 1992 et 2015.



""")
    
    st.markdown('📊Data Source: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/data')
    st.markdown('📷 Photo source: Unsplash')


if page == pages[1]:
    st.markdown("Chargement des données: via Google Colab")
    st.markdown("Langage: Python")
    st.markdown("Librarie: Pandas, Numpy, Matplotlit, Searborn,Scikit Learn")
    st.subheader("Data overview")
    st.write("Extrait du DataFrame: Nous avons les divisé en 3 partie par type de donnée (voir les tableux en bas)")
    st.dataframe(df.head())
    st.write('Data type: INTEGER')
    data_int = { "Columne name" : ["OBJETID","FOD_ID","FIRE_YEAR","DISCOVERY_DOY","STAT_CAUSE_CODE"],
            "Data Type":['INT64','INT64','INT64','INT64','INT64'],
            "Missing Data Rate":[0,0,0,0,0]}
    st.dataframe(pd.DataFrame(data_int))
    st.markdown("Nous contactons en total 5 variables qui contiennent des differentes information:")
    st.markdown("""
                1. OBJETID, FOD_ID et STAT_CODE: identifiant
                2. FIRE_YEAR et DISCOVERY_DOY: information des dates qui nécessite de transformation en type datetime""")

    st.write('Data type: OBJET')
    data_obj = {"Columnn name":["FPA_ID","SOURCE_SYSTEM_TYPE","SOURCE_SYSTEM","NWCG_REPORTING_AGENCY","NWCG_REPORTING_ID",
                                "NWCG_REPORTING_UNIT_NAME","SOURCE_REPORTING_UNI","SOURCE_REPORTING_UNIT_NAME", "LOCAL_FIRE_REPORT_ID",
                                "LOCAL_INCIDENT_ID","FIRE_CODE","FIRE_NAME","ICS_209_INCIDENT_NUMBER","ICS_209_NAME","MTBS_ID","MTBS_FIRES_NAME",
                                "COMPLEX_NAME","STAT_CAUSE_DESCR","FIRE_SIZE_CLASS","OWNER_DESCR","STATE","COUNTY","FIPS_NAME","SHAPE"],
                "Data Type" : ["OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT",
                               "OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT","OBJECT",
                               "OBJECT","OBJECT","OBJECT","OBJECT",],
                "Data Missing Rate" : [0,0,0,0,0,0,0,0, 77.6 , 43.65 , 82.73 , 51.08, 98.63, 98.63, 99.41, 99.41, 99.72, 0,0,0,0, 36.06, 36.06,0]}
    st.dataframe(pd.DataFrame(data_obj))
    st.markdown("""
                Nous avons constaté qu'il y avait beaucoup d'identifiant: ex: nom de l'agence, nom du feu...etc
                1. STAT_CAUSE_DESCR, FIRE_SIZE_CLASS: Ceux sont des cause du feux et la taille du feux que l'on pourrait en utiliser plus tard pour le prédiction
                2. STATE et COUNTY: Idication des lieux que l'on pourrait en utiliser pour Data Visualisation.
                """)

    data_float = {"Column name": ["DISCOVERY_DATE","DISCOVERY_TIME","CONT_DATE","CONT_DOY","CONT_TIME","FIRE_SIZE","LATITUDE","LONGTITUDE",
                                 "OWNER_CODE","FIPS_CODE"],
                "Data Type" : ["FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64"],
                "Data Missing Rate" : [0,46.94,47.41,47.41,51.72,0,0,0,0,36.06]}
    st.write("Data type: FLOAT 64")
    st.dataframe(pd.DataFrame(data_float))
    st.markdown("""
                1. LATITUDE et LONGTITUDE: information géogrphique
                2. DISCOVERY_DATE, DISCOVERY_YEAR, CONT_DATE, CONT_DOY, CONT_TIME: information date qui néccite de transformation en datetime
                3. OWER_CODE et FIS_CODE: Identifiant
                
                """)
              
    st.markdown("""
                REMARQUE:\n
                1. Un dataset volumineux: 39 colonnes et 1,88 million de lignes\n
                2. Le dataset ne contient pas de doublons\n
                3. Valuer NA: certaines colonnes contennent plus de 50 pourcent de valeurs manquantes, dont il sera difficile de les remplacer.\n
                   Cependant au vu du volumn de donnée disponilbe, on en est venu à la conclusion qu'il serait possible de réaliser un apprentissage sur une partie seule du dataset et ainsi pouvoir prédire certaines valuers manquante.
                4. Nous avons remarqué que les dates sont en format julien\n

                """)
    st.subheader("Observation de la corelation pour la colonne DISCOVERY_TIME par rapport aux autre valeurs manquantes")
    valeurs_manquantes = df.isnull()
    pd.set_option("display.max_columns",None)
    nombre_valeurs_manquantes = valeurs_manquantes.groupby(valeurs_manquantes["DISCOVERY_TIME"]).sum()
    st.dataframe(nombre_valeurs_manquantes)
    st.markdown("On observe que la grande partie des donées manquantes pour le DISCOVERY_TIME l'est aussi pour le CONT_TIME et/ou CONT_DOT\n"
    "Inversement cela implique, sous convert que les valurs présentes sont exploitables, que les donées DISCOVERY_TIME et CONT_TIME sont présentes simultanément.")
    


if page == pages[2]:
    st.write('### Nettoyage et enrichessment des donées')
    st.markdown("""
                Nous allons maintenant procéder à la supressions des :\n
                Colonnes que nous jugeons non pertinentes dans la suite des travaux d'apprentissage pour le machine learning :
                """)
    code_01 =''' df.drop(["FOD_ID", "FPA_ID","NWCG_REPORTING_AGENCY", "NWCG_REPORTING_UNIT_ID", "NWCG_REPORTING_UNIT_NAME", "LOCAL_FIRE_REPORT_ID", "LOCAL_INCIDENT_ID", "FIRE_CODE", "FIRE_CODE",
         "ICS_209_INCIDENT_NUMBER", "ICS_209_NAME", "MTBS_ID", "MTBS_FIRE_NAME", "COMPLEX_NAME", "OWNER_CODE", "OWNER_DESCR", "FIRE_NAME", "COUNTY", "FIPS_CODE", "FIPS_NAME"], axis = 1, inplace = True)'''
    st.code(code_01, language="python")
    st.markdown("""Ces colonnes n'apportent qu'une information d'identification du feu en fonction des différents organismes, ainsi aucune caractéristique
                du feu n'est perdu, cela limite le bruit en ne gardant qu'un seul ID""")
    
    # load clean data
    df1 = pd.read_csv("fires_cleaned.csv")

    

    st.subheader("Pré-processing")
    st.write("1. Convertissement des donées")
    df.dtran = {
        "Colomn Name": ["DISCOVERY_DATE","CONT_DATE"],
        "Convertion" : ["Format Julien => Format Georgian","Format Julien => Format Georgian"]
    }

    st.write("La variable FIRE_SIZE est en acres, on va la convertir en Hectares, unité plus parlante pour les lecteurs :")
    code_hectre = ("df['FIRE_SIZE'] = df['FIRE_SIZE']/ 2.471")
    st.code(code_hectre)
    
    st.write("Date type convertissement:")
    # Show python code
    st.dataframe(df.dtran)
    import streamlit as st

    if st.checkbox("Afficher le code: Conversion Date Julienne => Grégorienne"):
        code = """
    from datetime import datetime, timedelta

    def julian_to_gregorian(julian_date):
        julian_start = datetime(2000, 1, 1)  # Date grégorienne correspondant à la date julienne 2451545.0
        julian_start_as_days = 2451545.0  # Valeur en jours de la date julienne de référence

    # Calcul de la différence en jours
        delta_days = julian_date - julian_start_as_days

    # Retourner la date grégorienne
        return julian_start + timedelta(days=delta_days)

# Application de la fonction aux colonnes du DataFrame
    df['DISCOVERY_DATE'] = df['DISCOVERY_DATE'].dropna().apply(julian_to_gregorian)
    df['CONT_DATE'] = df['CONT_DATE'].dropna().apply(julian_to_gregorian)
"""
        st.code(code, language="python")

    st.write("2. Création les nouvelles variable")
    df.newcol = {
        "Colonnes inspirées":["STATE","STAT_CAUSE_DESCP","DISCOVERY/CONT DATE","DISCOVERY_DATE","DISCOVERY_DATE","DISCOVERY_DATE","CONT_DATE","CONT_DATE"],
        "Création Colonnes":["REGIONS","CAUSE_CATEGORY","FIRE_DURATION","DISCOVERY_YEAR","DISCOVERY_MONTH","DISCOVERY_DATE","CONT_YEAR","CONT_MONTH"]
    }
    st.dataframe(df.newcol)
    st.write("3. Traitement des valeurs manquantes")
    data_na = {
        "Colonnes avec un nombre élevé de valeurs manquantes":["DISCOVERY_TIME","CONT_TIME", "FIRE_NAME", "COUNTY", "FIPS_CODE", "FIPS_NAME"],
        "Traitement":["Drop","Drop","Drop","Drop","Drop","Drop"]
    }
    st.dataframe(data_na)
    st.markdown("""
    "DISCOVERY_TIME","CONT_TIME": On observe qu'une majeur partie des données manquante sur Discovery Time est aussi manquante
    sur le Control Time. Par conséquent, un modèle d'apprentissage se basant sur le moment de découverte ne pourra pas être appliqué correctement
    sur le dataset cible avec les cont_date manquante.
                
    "FIRE_NAME","FIPS_CODE", "FIPS_NAME": comme décrit plus bas, la méthodologie trouvée (Geopy) est trop longue pour pouvoir être exploité
     ici. Si la gestion des feux peut avoir un impact selon les county, on suppose que la position GPS aura un impact majoritaire.
   
                """)
    st.write("4. Traitement des valeurs aberrantes")
    st.markdown("""
    Nous avons remarqué que la variable FIRE_SIZE contient des valeurs aberrantes.
    Nous avons décidé de supprimer ces valeurs aberrantes pour éviter qu'elles n'influencent les résultats de notre analyse.
    Nous avons également remarqué que la variable FIRE_DURATION contient des valeurs aberrantes, notamment des feux de plus de 1000 jours.
    Nous avons décidé de supprimer ces valeurs aberrantes pour éviter qu'elles n'influencent les résultats de notre analyse.
    Nous avons également remarqué que la variable CONT_DATE contient des valeurs aberrantes, notamment des feux qui ont été contrôlés avant leur découverte.
    Nous avons décidé de supprimer ces valeurs aberrantes pour éviter qu'elles n'influencent les résultats de notre analyse.
    """)
    st.write("5. Ajout de nouveaux données: Météo")
    st.markdown("""
    Nous avons enrichi notre dataset avec des données météorologiques provenant de la base de données US Wildfire Weather Data.
    Ces données comprennent des informations sur la température, les précipitations et la vitesse du vent, qui sont des facteurs importants dans l'étude des incendies de forêt.
    Nous avons fusionné ces données avec notre dataset principal en utilisant l'OBJECTID comme clé de jointure.
    """)

if page == pages[3]:
 
    @st.cache_data
    def load_data():
     url = 'https://drive.google.com/uc?export=download&id=1M90PGon2io8Bx9NusCvXP1AvFb1X2_yN' 
     df1 = pd.read_csv(url)
     return df1
#---------------------1ST plot--------------------------------------------------------
# Clean up year values
    df1['DISCOVERY_YEAR'] = pd.to_numeric(df1['DISCOVERY_YEAR'], errors='coerce')
    df1 = df1.dropna(subset=['DISCOVERY_YEAR'])
    df1 = df1[df1['DISCOVERY_YEAR'] >= 1992]

# Group by state and year
    fire_counts = df1.groupby(["REGIONS", 'DISCOVERY_YEAR']).size().reset_index(name='FIRE_COUNT')

# Pivot table for plotting
    fire_pivot = fire_counts.pivot(index='DISCOVERY_YEAR', columns='REGIONS', values='FIRE_COUNT').fillna(0)

# Plot
    st.subheader("Nombre de Feux par an par Région")
    fig, ax = plt.subplots(figsize=(14, 7))
    fire_pivot.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de Feux")
    ax.set_title("Nombre de Feux par Mois par Région")
    st.pyplot(fig)
# comments
    st.markdown("""
    Le graphique ci-dessus présente le nombre de feux de forêt par an et par région.
    On y observe que les feux sont plus fréquents dans certaines régions, notamment dans l'Ouest des États-Unis.    
    Cette visualisation met en évidence l'importance de la prévention des feux de forêt dans les régions les plus touchées.
    En outre, on peut constater une tendance à la hausse du nombre de feux au fil des ans, ce qui souligne l'importance de la surveillance et de la gestion des incendies de forêt.
    """)
#-----2ND plot------------------------------------------------------------------------------#
    st.subheader("Nombre de Feux par Mois par Région")
# Group and pivot data
    df_monthly = df1.groupby(['DISCOVERY_MONTH', 'REGIONS']).size().reset_index(name='count')
    df_pivot = df_monthly.pivot(index='DISCOVERY_MONTH', columns='REGIONS', values='count').fillna(0)

# Sort by month if needed
    df_pivot = df_pivot.sort_index()

# Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(20, 10))
    df_pivot.plot(kind='bar', stacked=True, ax=ax)

# Customize axes
    ax.set_xlabel('Mois de Découverte')
    ax.set_ylabel('Nombre de Feux')
    ax.set_title('Nombre de Feux par Mois par Région')
    ax.set_xticks(range(0, 12))
    ax.set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'], rotation=45)
    ax.grid(True)

# Show in Streamlit
    st.pyplot(fig)

# comments
    st.markdown("""
    Le graphique présente le nombre de feux de forêt par mois et par région.
    On y observe que les feux sont plus fréquents pendant les mois d'été, avec un pic en juillet et août.
    Cette visualisation met en évidence l'importance de la prévention des feux de forêt pendant les mois d'été, lorsque les conditions sont les plus propices aux incendies.
    """)
    
# -----------------------3RD plot-------------#
 # Ensure DISCOVERY_MONTH is numeric (1–12) and convert to month names
    df1['MONTH'] = pd.to_datetime(df1['DISCOVERY_MONTH'], format='%m', errors='coerce').dt.month_name()

# Optional: enforce month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
    df1['MONTH'] = pd.Categorical(df1['MONTH'], categories=month_order, ordered=True)

# Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.countplot(data=df1, x='MONTH', hue='CAUSE_CATEGORY', ax=ax)

# Format
    ax.set_title('Wildfire Causes by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Fires')
    plt.xticks(rotation=45)
    plt.tight_layout()

# Streamlit display
    st.pyplot(fig)
# comments
    st.markdown("""
    Le graphique, présente le nombre de feux de forêt par mois et par région.
    On y observe que les feux causés par des activités humaines (en orange) sont plus fréquents pendant les mois d'été, tandis que les feux d'origine naturelle (en bleu) sont plus fréquents au printemps et en automne.
    Cette visualisation met en évidence l'importance de la prévention des feux de forêt, en particulier pendant les mois d'été où les activités humaines sont plus fréquentes.
    """)
#-----------------------------4TH plot----------------------------------------------------


    df_cause_state = df1.groupby(['CAUSE_CATEGORY', 'REGIONS']).size().reset_index(name='count')
    df_pivot = df_cause_state.pivot(index='REGIONS', columns='CAUSE_CATEGORY', values='count').fillna(0)

# Optional: sort states by total fires
    df_pivot['Total'] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values('Total', ascending=False).drop(columns='Total')

# Set color palette
    sns.set_palette("tab20")  # Up to 20 distinct colors

# Plot
    fig, ax = plt.subplots(figsize=(22, 10))
    df_pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8)

# Style and layout
    ax.set_title("Nombre de Feux par Cause et par Région", fontsize=20, weight='bold')
    ax.set_xlabel("État", fontsize=14)
    ax.set_ylabel("Nombre de Feux", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Cause", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)

# comments
    st.markdown("""
    Dans le graphique ci-dessus, nous avons visualisé le nombre de feux de forêt par cause et par région.
    On peut observer que les feux causés par des activités humaines sont prédominants dans la plupart des régions.
    Cependant, il est intéressant de noter que dans certaines régions, les feux d'origine naturelle sont également significatifs.
    Cette visualisation met en évidence l'importance de la prévention des feux de forêt, en particulier dans les zones où les activités humaines sont fréquentes.
    """)

#---------------------5th-------------------------------------------------------------#
# Increase figure size
    fig=plt.figure(figsize=(14, 8))

# Use jitter + transparency to avoid overplotting
    sns.boxplot(x='CAUSE_CATEGORY', y='FIRE_SIZE', data=df1, showcaps=False, fliersize=0, color='lightgray')
    sns.stripplot(x='CAUSE_CATEGORY', y='FIRE_SIZE', data=df1, jitter=True, alpha=0.3)


# Improve labels and layout
    plt.xticks(rotation=45, ha='right')
    plt.title('Taille des Incendies selon la Cause', fontsize=18, weight='bold')
    plt.xlabel('Catégorie de Cause', fontsize=14)
    plt.ylabel('Taille du Feu (hectares)', fontsize=14)
    plt.tight_layout()

# Optional: use log scale for skewed fire sizes
# plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
# comments
    st.markdown(""" 
    Le graphique ci-dessus présente la taille des incendies en fonction de leur cause.
    On y observe que les incendies causés par des activités humaines ont tendance à être plus petits en taille, tandis que les incendies d'origine naturelle sont plus grands.
    Cette visualisation met en évidence l'importance de la prévention des incendies, en particulier pour les causes d'origine naturelle qui peuvent entraîner des incendies de grande taille.
    """)

#-----------------6th plot --------------------------------------------------------------------------#
    avg_size = df1.groupby('STAT_CAUSE_DESCR')['FIRE_SIZE'].mean().sort_values(ascending=False)

# Plot
    fig =plt.figure(figsize=(14, 8))
    sns.barplot(x=avg_size.index, y=avg_size.values, palette="viridis")

    plt.xticks(rotation=45, ha='right')
    plt.title("🔥 Taille Moyenne des Feux par Cause", fontsize=18, weight='bold')
    plt.xlabel("Cause du Feu", fontsize=14)
    plt.ylabel("Taille Moyenne (hectares)", fontsize=14)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
# comments
    st.markdown("""
    Le graphique ci-dessus présente la taille moyenne des feux de forêt par cause.
    On y observe que les feux causés par des activités humaines ont tendance à être plus petits en taille, tandis que les feux d'origine naturelle sont plus grands.
    Cette visualisation met en évidence l'importance de la prévention des feux de forêt, en particulier pour les causes d'origine naturelle qui peuvent entraîner des incendies de grande taille.
    """)

# ----------------------------------------end of 3rd page-------------------------------------------------

if page == pages[4]:
    # import pacakged
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    st.header("🔍 Machine Learning: Prédiction de la Durée des Feux")
    st.markdown("Comparez deux modèles de régression : **Random Forest** et **XGBoost**")

    @st.cache_data
    def load_data():
        return pd.read_csv("fires_cleaned.csv")

    df1 = load_data()
    df_weather = pd.read_csv("US_wildfire_weather_data.csv")
    df_merged = df1.merge(df_weather, on="OBJECTID")

# Encode cyclic features
    def encode_cyclic(df, col, max_val):
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        return df

    df_merged = encode_cyclic(df_merged, "DISCOVERY_MONTH", 12)
    df_merged = encode_cyclic(df_merged, "DISCOVERY_DOW", 7)

    if "FIRE_SIZE_CLASS" in df_merged.columns:
        df_merged = pd.get_dummies(df_merged, columns=["FIRE_SIZE_CLASS"], drop_first=True)

# All available features
    all_features = [
        'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'STAT_CAUSE_CODE',
        'DISCOVERY_MONTH_sin', 'DISCOVERY_MONTH_cos',
        'DISCOVERY_DOW_sin', 'DISCOVERY_DOW_cos',
        'DISCOVERY_YEAR',
        'temp_mean_0', 'temp_mean_10', 'temp_mean_30', 'temp_mean_60', 'temp_mean_180',
        'prcp_sum_0', 'prcp_sum_10', 'prcp_sum_30', 'prcp_sum_60', 'prcp_sum_180',
        'wspd_mean_0', 'wspd_mean_10', 'wspd_mean_20', 'wspd_mean_60', 'wspd_mean_180'
    ]
    all_features = [col for col in all_features if col in df_merged.columns]

# Set default features
    default_features = [
        'LATITUDE', 'LONGITUDE', 'FIRE_SIZE',
        'DISCOVERY_MONTH_sin', 'DISCOVERY_MONTH_cos',
        'DISCOVERY_DOW_sin', 'DISCOVERY_DOW_cos',
        'temp_mean_0', 'prcp_sum_0', 'wspd_mean_0'
    ]

# Initialize session state for feature selection
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = default_features.copy()

# Reset button
    if st.button("🔁 Réinitialiser les variables par défaut"):
        st.session_state.selected_features = default_features.copy()

# Multiselect with session state
    selected_features = st.multiselect(
        "📌 Sélectionnez les variables explicatives :",
        all_features,
        default=st.session_state.selected_features,
        key="selected_features"
    )


# Feature selector with defaults
    selected_features = st.multiselect(
        "📌 Sélectionnez les variables explicatives :",
        all_features,
        default=default_features
    )

# Continue if features selected
    if selected_features:
        df_model = df_merged[selected_features + ["FIRE_DURATION"]].dropna(subset=["FIRE_DURATION"])
        X = df_model.drop(columns=["FIRE_DURATION"])
        y = df_model["FIRE_DURATION"]

    # Preprocessing
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        y_log = np.log1p(y)

        valid_mask = np.isfinite(y_log) & np.all(np.isfinite(X_scaled), axis=1)
        X_filtered = X_scaled[valid_mask]
        y_filtered = y_log[valid_mask]

        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42
        )

        model_choice = st.selectbox("Choisissez un modèle :", ["Random Forest", "XGBoost"])

        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
        else:
            model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)

        model.fit(X_train, y_train_log)

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test_log)

        st.subheader(f"📈 Évaluation du modèle : {model_choice}")
        st.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_true, y_pred):.3f}")

        st.subheader("📊 Top 10 variables les plus importantes")

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color='skyblue')
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 des variables importantes")
        st.pyplot(fig)

        st.subheader("🔍 Aperçu des prédictions")
        df_results = pd.DataFrame({
            "Durée réelle": y_true[:10],
            "Durée prédite": y_pred[:10]
        }).round(2)
        st.dataframe(df_results)

    else:
        st.warning("Veuillez sélectionner au moins une variable explicative.")
