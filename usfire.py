import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('fires.csv')

st.title('Projet de 1.88 millions de feux aux USA entre 1992 à 2015')
st.sidebar.title('Sommaire')
pages = ['Introduction','Exploration','Nettoyage et enrichessment des donées','Data Visualisation','Visualisation Interactive','Modèle de prédition']
page = st.sidebar.radio('Aller vers', pages)
st.sidebar.markdown("Autheurs")
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
    st.write("Extrait du DataFrame")
    st.dataframe(df.head())
    st.write('Data type: INTEGER')
    data_int = { "Columne name" : ["OBJETID","FOD_ID","FIRE_YEAR","DISCOVERY_DOY","STAT_CAUSE_CODE"],
            "Data Type":['INT64','INT64','INT64','INT64','INT64'],
            "Missing Data Rate":[0,0,0,0,0]}
    st.dataframe(pd.DataFrame(data_int))

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

    data_float = {"Column name": ["DISCOVERY_DATE","DISCOVERY_TIME","CONT_DATE","CONT_DOY","CONT_TIME","FIRE_SIZE","LATITUDE","LONGTITUDE",
                                 "OWNER_CODE","FIPS_CODE"],
                "Data Type" : ["FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64","FLOAT64"],
                "Data Missing Rate" : [0,46.94,47.41,47.41,51.72,0,0,0,0,36.06]}
    st.wrtie("Data type: FLOAT 64")
    st.dataframe(pd.DataFrame(data_float))
    st.markdown("""
                REMARQUE:\n
                1. Le dataset ne contient pas de doublons\n
                2. Valuer NA: certaines colonnes contennent plus de 50 pourcent de valeurs manquantes, dont il sera difficile de les remplacer.\n
                   Cependant au vu du volumn de donnée disponilbe, on en est venu à la conclusion qu'il serait possible de réaliser un apprentissage sur une partie seule du dataset et ainsi pouvoir prédire certaines valuers manquante.
                3. Nous avons remarqué que les dates sont en format julian\n

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
    code =''' df.drop(["FOD_ID", "FPA_ID","NWCG_REPORTING_AGENCY", "NWCG_REPORTING_UNIT_ID", "NWCG_REPORTING_UNIT_NAME", "LOCAL_FIRE_REPORT_ID", "LOCAL_INCIDENT_ID", "FIRE_CODE", "FIRE_CODE",
         "ICS_209_INCIDENT_NUMBER", "ICS_209_NAME", "MTBS_ID", "MTBS_FIRE_NAME", "COMPLEX_NAME", "OWNER_CODE", "OWNER_DESCR", "FIRE_NAME", "COUNTY", "FIPS_CODE", "FIPS_NAME"], axis = 1, inplace = True)'''
    st.code(code, language="python")
    st.markdown("""Ces colonnes n'apportent qu'une information d'identification du feu en fonction des différents organismes, ainsi aucune caractéristique
                du feu n'est perdu, cela limite le bruit en ne gardant qu'un seul ID""")
    
    # load clean data
    df1 = pd.read_csv("fires_cleaned.csv")

    

    st.subheader("Pré-processing")
    st.write("1. Convertissement des donées")
    df.dtran = {
        "Colomn Name": ["DISCOVERY_DATE","CONT_DATE"],
        "Convertion" : ["Format Julian => Format Georgian","Format Julian => Format Georgian"]
    }
    st.write("La variable FIRE_SIZE est en acres, on va la convertir en Hectares, unité plus parlante pour les lecteurs :")
    code = ("df['FIRE_SIZE'] = df['FIRE_SIZE']/ 2.471")
    st.code(code)
    
    st.write("Date type convertissement:")
    st.dataframe(df.dtran)
    st.write("2. Création les nouvelles variable")
    df.newcol = {
        "Colonnes inspirées":["STATE","STAT_CAUSE_DESCP","DISCOVERY/CONT DATE","DISCOVERY_DATE","DISCOVERY_DATE","DISCOVERY_DATE","CONT_DATE","CONT_DATE"],
        "Création Colonnes":["REGIONS","CAUSE_CATEGORY","FIRE_DURATION","DISCOVERY_YEAR","DISCOVERY_MONTH","DISCOVERY_DATE","CONT_YEAR","CONT_MONTH"]
    }
    st.dataframe(df.newcol)
    st.write("3. Traitement des valeurs manquantes")
    st.markdown("""





                """)
    

if page == pages[3]:
      
    @st.cache_data
    def load_data():
        return pd.read_csv("fires_cleaned.csv")
    df1 = load_data()
#------1st plot------------------------------------------------------------------------------#
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
    st.markdown("""
                




                """)
# -----------------------"2nd plot-------------#

    df_cause_state = df1.groupby(['CAUSE_CATEGORY', 'STATE']).size().reset_index(name='count')
    df_pivot = df_cause_state.pivot(index='STATE', columns='CAUSE_CATEGORY', values='count').fillna(0)

# Optional: sort states by total fires
    df_pivot['Total'] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values('Total', ascending=False).drop(columns='Total')

# Set color palette
    sns.set_palette("tab20")  # Up to 20 distinct colors

# Plot
    fig, ax = plt.subplots(figsize=(22, 10))
    df_pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8)

# Style and layout
    ax.set_title("Nombre de Feux par Cause et par État", fontsize=20, weight='bold')
    ax.set_xlabel("État", fontsize=14)
    ax.set_ylabel("Nombre de Feux", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Cause", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)

#---------------------3rd-------------------------------------------------------------#
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

#-----------------4th plot --------------------------------------------------------------------------#
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

if page == pages[4]:
    # --- PAGE 4: Visualisation Interactive ---
    st.title("Visualisation Interactive : Taille des Feux par Cause, État, Année et Saison")
    
    @st.cache_data
    def load_data():
        return pd.read_csv("fires_cleaned.csv")  

    df1 = load_data()

    if 'DISCOVERY_DATE' not in df1.columns:
        st.error("La colonne 'DISCOVERY_DATE' est absente.")
    else:
        df1['DISCOVERY_DATE'] = pd.to_datetime(df1['DISCOVERY_DATE'], errors='coerce')
        df1['YEAR'] = df1['DISCOVERY_DATE'].dt.year
        df1['SEASON'] = df1['DISCOVERY_DATE'].dt.month.apply(lambda m: (
            'Winter' if m in [12, 1, 2] else
            'Spring' if m in [3, 4, 5] else
            'Summer' if m in [6, 7, 8] else
            'Fall'
        ))

        # --- Inline Filters ---
        st.subheader("🧰 Filtres")

        col1, col2 = st.columns(2)
        with col1:
            selected_states = st.multiselect("États", sorted(df1['STATE'].dropna().unique()), default=sorted(df1['STATE'].dropna().unique())[:5])
            selected_years = st.multiselect("Années", sorted(df1['YEAR'].dropna().unique()), default=sorted(df1['YEAR'].dropna().unique())[-5:])
        with col2:
            selected_causes = st.multiselect("Causes", sorted(df1['STAT_CAUSE_DESCR'].dropna().unique()), default=sorted(df1['STAT_CAUSE_DESCR'].dropna().unique())[:5])
            selected_seasons = st.multiselect("Saisons", ['Winter', 'Spring', 'Summer', 'Fall'], default=['Spring', 'Summer', 'Fall'])

        st.markdown("### 📊 Type d'Agrégation")
        agg_type = st.radio(
            "Choisissez ce que vous voulez visualiser :",
            ["Moyenne (taille moyenne)", "Totale (taille totale)", "Nombre (nombre d'incendies)"]
        )

        # Filter data
        filtered_df = df1[
            df1['STATE'].isin(selected_states) &
            df1['STAT_CAUSE_DESCR'].isin(selected_causes) &
            df1['YEAR'].isin(selected_years) &
            df1['SEASON'].isin(selected_seasons)
        ]

        if filtered_df.empty:
            st.warning("Aucune donnée ne correspond à vos filtres.")
        else:
            # Aggregation
            if agg_type == "Moyenne (taille moyenne)":
                agg_df = filtered_df.groupby(['STAT_CAUSE_DESCR', 'STATE'])['FIRE_SIZE'].mean().reset_index()
                y_label = "Taille moyenne (ha)"
            elif agg_type == "Totale (taille totale)":
                agg_df = filtered_df.groupby(['STAT_CAUSE_DESCR', 'STATE'])['FIRE_SIZE'].sum().reset_index()
                y_label = "Taille totale (ha)"
            else:  # Nombre
                agg_df = filtered_df.groupby(['STAT_CAUSE_DESCR', 'STATE'])['FIRE_SIZE'].count().reset_index()
                y_label = "Nombre d'incendies"

            # Plot
            st.subheader("📈 Résultat")
            plt.figure(figsize=(16, 8))
            sns.barplot(
                data=agg_df,
                x='STAT_CAUSE_DESCR',
                y='FIRE_SIZE',
                hue='STATE',
                palette="Set2"
            )
            plt.xticks(rotation=45, ha='right')
            plt.title(f"🔥 Feux par Cause et État ({agg_type})", fontsize=18, weight='bold')
            plt.xlabel("Cause", fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(plt)




if page == pages[5]:

# load packages

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    st.title("Machine Learning: Prédiction de la Durée des Feux")
    st.markdown("Comparez deux modèles de régression : **Random Forest** et **XGBoost**")

    @st.cache_data
    def load_data():
        return pd.read_csv("fires_cleaned.csv")

    df1 = load_data()
    df_weather = pd.read_csv("US_wildfire_weather_data.csv")

# Fusion des données
    df_merged = df1.merge(df_weather, on="OBJECTID")

# Sélection du modèle
    model_choice = st.selectbox("Choisissez un modèle :", ["Random Forest", "XGBoost"])

# Fonction pour encodage cyclique
    def encode_cyclic(df, col, max_val):
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        return df

    df_merged = encode_cyclic(df_merged, "DISCOVERY_MONTH", 12)
    df_merged = encode_cyclic(df_merged, "DISCOVERY_DOW", 7)

# Encodage one-hot
    if "FIRE_SIZE_CLASS" in df_merged.columns:
        df_merged = pd.get_dummies(df_merged, columns=["FIRE_SIZE_CLASS"], drop_first=True)

# Sélection des variables
    features = [
        'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'STAT_CAUSE_CODE',
        'DISCOVERY_MONTH_sin', 'DISCOVERY_MONTH_cos',
        'DISCOVERY_DOW_sin', 'DISCOVERY_DOW_cos',
        'DISCOVERY_YEAR',
        'temp_mean_0', 'temp_mean_10', 'temp_mean_30', 'temp_mean_60', 'temp_mean_180',
        'prcp_sum_0', 'prcp_sum_10', 'prcp_sum_30', 'prcp_sum_60', 'prcp_sum_180',
        'wspd_mean_0', 'wspd_mean_10', 'wspd_mean_20', 'wspd_mean_60', 'wspd_mean_180'
    ]   
    features = [col for col in features if col in df_merged.columns]

# Données d'entraînement
    df_model = df_merged[features + ["FIRE_DURATION"]].dropna(subset=["FIRE_DURATION"])
    X = df_model.drop(columns=["FIRE_DURATION"])
    y = df_model["FIRE_DURATION"]

# Prétraitement
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

# Transformation log
    y_log = np.log1p(y)

# Nettoyage des valeurs invalides
    valid_mask = np.isfinite(y_log) & np.all(np.isfinite(X_scaled), axis=1)
    X_filtered = X_scaled[valid_mask]
    y_filtered = y_log[valid_mask]

# Découpage train/test
    X_train, X_test, y_train_log, y_test_log = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Entraînement
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

    model.fit(X_train, y_train_log)

# Prédictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)

# Évaluation
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

# Résultats
    st.subheader(f"📈 Évaluation du modèle : {model_choice}")
    st.metric("MAE (erreur absolue moyenne)", f"{mae:.2f}")
    st.metric("RMSE (erreur quadratique moyenne)", f"{rmse:.2f}")
    st.metric("R² (coefficient de détermination)", f"{r2:.3f}")

# Aperçu
    st.subheader("🔍 Aperçu des prédictions")
    df_results = pd.DataFrame({
        "Durée réelle": y_true[:10],
        "Durée prédite": y_pred[:10]
    }).round(2)
    st.dataframe(df_results)