import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#custom tools
from scripts.fetcher import fetch_from_NOAA
from scripts.data_processing import phone_a_friend
from scripts.ui_tools import plot_stations, file_gen_settings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#NOTE: Scroll to bottom to see run() function

load_dotenv()

STATE = os.getenv("STATE")
WEATHERDATA_DIR = os.getenv("WEATHERDATA_DIR")
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
important_features = ['PRCP', 'SNOW', 'SNWD', 'TMIN', 'TMAX', 'elevation']
weather_columns = ['PRCP', 'SNOW', 'SNWD', 'TMIN', 'TMAX']

#approximate values for stations missing data entries given their monthly means
#(we assume temperatures dont fluctuate too much over the month)
def find_means(df):
    df['TMAX'] = df.groupby('station')['TMAX'].transform(lambda x: x.fillna(x.mean()))
    df['TMIN'] = df.groupby('station')['TMIN'].transform(lambda x: x.fillna(x.mean()))
    return df

#reduce usable stations if lack of data (see EDA)
def reduce_dataframe(df,file):
    #remove stations that do not have enough important variables
    df_sliced = df.dropna(subset=important_features, thresh=4) 
    #remove stations that do not represent the month well (less than 15 days of data)
    station_counts = df_sliced['station'].value_counts()
    stations_to_keep = station_counts[station_counts >= 15].index
    df_sliced = df_sliced[df_sliced['station'].isin(stations_to_keep)]
    #reduction output
    print(f'reduced the {file} dataset to', df_sliced['station'].nunique(), 'stations (dropped stations with few entries)')
    return df_sliced

def impute_dataframe(df_sliced):
    #use sklearn's IterativeImputer to approximate the missing values on the remaining stations
    impute_df = df_sliced[important_features].copy()
    imputer = IterativeImputer(random_state=42, max_iter=10)
    imputed_array = imputer.fit_transform(impute_df)
    df_sliced.loc[:, important_features] = pd.DataFrame(imputed_array, columns=important_features, index=impute_df.index)
    print(f"All missing data in stations imputed\n")
    return df_sliced

def aggregate_dataframe(df_sliced):
    #turn all daily readings into the monthly parameters for each station
    df_agg = df_sliced.groupby('station').agg(
        mean_TMAX=('TMAX', 'mean'),
        std_TMAX=('TMAX', 'std'),
        mean_TMIN=('TMIN', 'mean'),
        total_PRCP=('PRCP', 'sum'),
        PRCP_days=('PRCP', lambda x: (x > 0).sum()), #rainy days
        snow_days=('SNOW', lambda x: (x > 0).sum()), #snowing days
        frost_days=('TMIN', lambda x: (x < 32).sum()), #days below freezing
        elevation=('elevation', 'first')
    )
    df_agg.columns = ['mean_TMAX','std_TMAX','mean_TMIN','total_PRCP',
                  'PRCP_days','snow_days','frost_days','elevation']
    df_agg = df_agg.reset_index()
    return df_agg

def standardize_dataframe(df_agg):
    #standardize the data so that we can use it in a model (using sklearn's StandardScaler)
    df_features = df_agg.drop(columns=['station'])
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)
    return df_scaled,scaler

def reduce_outliers_kmeans(df_scaled,df_final,kmeans,sil_cutoff=0.5):
    for threshold in range(100,0,-1):  #cutoff for deleting cluster outliers (after clustering)
        centroids = kmeans.cluster_centers_
        distances = np.linalg.norm(df_scaled - centroids[df_final['cluster']], axis=1)
        threshold = np.percentile(distances, threshold) #Define threshold to cut at

        remaining_stations = distances < threshold #Filter out outliers
        df_scaled_clean = df_scaled[remaining_stations]
        df_final_clean = df_final[remaining_stations].reset_index(drop=True)

        cleaned_clustering_labels = df_final_clean['cluster']
        sil_score_clean = silhouette_score(df_scaled_clean, cleaned_clustering_labels)

        if sil_score_clean > sil_cutoff:
            return df_final_clean, sil_score_clean
    #no ammount of cleaning can save this one:
    return df_final_clean, 0

def choose_best_k_combined(inertia, sil_scores, K_range):
    inertia = inertia[K_range[0]-1:K_range[-1]]
    sil_scores = sil_scores[K_range[0]-1:K_range[-1]]

    #This function standardizes inertia and silhouette scores,
    #combines them by dividing silhouette by inertia,
    #selects the k with the maximum combined score.
    inertia = np.array(inertia).reshape(-1, 1)
    sil_scores = np.array(sil_scores).reshape(-1, 1)

    #standardize both to 0–1 range
    scaler = MinMaxScaler()
    inertia_std = scaler.fit_transform(inertia).flatten()
    sil_std = scaler.fit_transform(sil_scores).flatten()

    #lower inertia is better, so we invert it (1 - inertia)
    inertia_std_inverted = np.clip(1 - inertia_std, 1e-6, 1)

    #calculate combined score
    combined_score = sil_std / inertia_std_inverted
    #print(combined_score)
    #pick the best k (where combined score is max)
    best_index = np.argmax(combined_score)
    best_k = K_range[best_index]
    return best_k

def k_means_clustering(df_scaled,df_sliced,df_agg,file):
    #generate inertia and silhouette scores
    inertia = []
    sil_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(df_scaled, kmeans.labels_))

    #get the optimal silhouette score
    k = choose_best_k_combined(inertia, sil_scores, range(4,7))
    
    #-------------------------------------------------------
    kmeans = KMeans(n_clusters=k, max_iter=2000, n_init=40, random_state=42)
    #start consolidating final info------------------------------------
    df_loc = df_sliced.groupby('station').agg(latitude=('latitude', 'first'),
                                    longitude=('longitude', 'first'),
                                    elevation=('elevation', 'first')
                                    ).reset_index(drop=True)
    df_final = df_agg['station'].to_frame().copy()
    df_final = pd.concat([df_final,
                        pd.DataFrame({'cluster':kmeans.fit_predict(df_scaled)}),
                        df_loc],axis=1)
    print(f'found {k} climates for {file}')

    #remove the outliers to improve accuracy
    a,b = reduce_outliers_kmeans(df_scaled,df_final,kmeans)
    return a,b,kmeans

def score_to_label(val): #for relative visualization
    if val <= -0.1: return "very low"
    elif val <= -0.5: return "low"
    elif val >= 1: return "very high"
    elif val >= 0.5: return "high"
    else: return "mid"

def label_climate(row): #name the clusters
    tmax = row['mean_TMAX']
    tmin = row['mean_TMIN']
    prcp = row['total_PRCP']
    prcp_days = row['PRCP_days']
    snow_days = row['snow_days']
    frost_days = row['frost_days']
    elevation = row['elevation']
    rel_elev = row['relative_elevation']
    
    avg_temp = (tmax + tmin) / 2
    sea_lvl = elevation < 200

    #create labels----------------------

    #temperance
    temptype = 'moderate'
    if frost_days > 20: temptype = 'frozen'
    elif avg_temp < 32: temptype = 'freezing'
    elif avg_temp < 40: temptype = 'very cold'
    elif avg_temp < 55: temptype = 'cold'
    elif avg_temp > 72: temptype = 'hot'
    elif avg_temp > 65: temptype = 'warm'

    #moisture
    precip_type = ''
    if snow_days > 10: precip_type = 'snowy'
    elif prcp_days > 10: precip_type = 'rainy'
    elif prcp < 1: precip_type = 'very dry'
    elif prcp < 2.36: precip_type = 'dry'
    elif prcp > 9: precip_type = 'very wet'
    elif prcp > 6: precip_type = 'wet'

    #region type based on elevation
    region = 'midlands'
    if sea_lvl: region = 'costal'
    elif rel_elev in ['very low','low']: region = 'lowland'
    elif rel_elev == 'high': region = 'highland'
    elif rel_elev == 'very high': region = 'mountainous'

    #-------------concatenate---------
    if precip_type == '':
        return temptype + ' ' + region
    else:
        return temptype + ' ' + precip_type + ' ' + region
    
def process_csv(file,write_over=False): 
    #check if already processed

    if file in os.listdir(PROCESSED_DIR) and (write_over==False):
        return None
    #file is not processed yet
    df = pd.read_csv(WEATHERDATA_DIR+'/'+file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    #reach out to neighboring stations to obtain missing data
    df = phone_a_friend(find_means(df), max_km=40, max_ele_meters=100)
    df_sliced = impute_dataframe(reduce_dataframe(df,file))
    df_agg = aggregate_dataframe(df_sliced)

    df_scaled,scaler = standardize_dataframe(df_agg)
    df_features = df_agg.drop(columns=['station'])

    df_final_clean,silhouette_score,model= k_means_clustering(df_scaled,df_sliced,df_agg,file)

    #transform from scaled to actual values
    cluster_info_relative = pd.DataFrame(model.cluster_centers_, columns=df_features.columns)
    cluster_info_actual = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_),columns=df_features.columns)
    #add relative elevations column (relative to rest of region)
    cluster_info_actual['relative_elevation'] = cluster_info_relative['elevation'].apply(score_to_label)

    #label the clusters
    cluster_info_actual['microclimate_label'] = cluster_info_actual.apply(label_climate, axis=1)
    cluster_to_climate = cluster_info_actual['microclimate_label'].to_dict()
    df_final_clean['climate'] = df_final_clean['cluster'].map(cluster_to_climate)

    #save the results
    df_final_clean.to_csv(PROCESSED_DIR+'/'+file, index=False)


#============================ Main Function =================================
def run(map=True,write=False):
    START_DATE, END_DATE, SAVEFILE_ISDATE = file_gen_settings()
    file = f"{STATE}_{SAVEFILE_ISDATE}.csv"

    if file not in os.listdir(WEATHERDATA_DIR): #no data file in directory
        print(f'fetching station data for {SAVEFILE_ISDATE} from NOAA')
        fetch_from_NOAA(START_DATE, END_DATE, SAVEFILE_ISDATE)

    print('working with ' + file)
    process_csv(file,write_over=write)

    df = pd.read_csv(PROCESSED_DIR + '/' + file)
    number_unique_climates = df['climate'].nunique()
    print(number_unique_climates)

    if map:
        plot_stations(df,k=number_unique_climates,show_climates=True)

#write will re-write the processed datafile if it exists
#it is recommended to run write=False if you just want to view the 
#previously run climates calc for the selected month
run(write=False)