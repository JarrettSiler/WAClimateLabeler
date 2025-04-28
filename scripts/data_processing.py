from geopy.distance import geodesic
import numpy as np
import pandas as pd
from tqdm import tqdm


#in the likely situation that some of the weatherstations used
#do not contain all relevant information (PRCP,SNOW,SNWD,TMAX,TMIN) this script
#implements Spatial-Temporal Neighbor Imputation, AKA a way for the weatherstation to 
#access values from other stations nearby

REQUIRED_VARS = ['PRCP', 'SNOW', 'SNWD','TMAX','TMIN']
FIXED_VARS = []  # These must already be present

#neighboring stations to impute from are within max_distance_km and elevation difference of max_elev_diff_m
def find_nearest_stations(target_station, stations_metadata, max_distance_km, max_elev_diff_m):
    neighbors = []
    for station_id, row in stations_metadata.iterrows():
        if station_id == target_station.name:
            continue

        dist = geodesic((target_station['latitude'], target_station['longitude']),
                        (row['latitude'], row['longitude'])).km

        elev_diff = abs(target_station['elevation'] - row['elevation'])

        if dist <= max_distance_km and (max_elev_diff_m is None or elev_diff <= max_elev_diff_m):
            neighbors.append((station_id, dist))

    neighbors.sort(key=lambda x: x[1])
    return neighbors

def get_temporal_neighbors(data, neighbor_ids, date, variable, window=0):
    values = []
    sources = []
    for offset in range(-window, window + 1):
        day = date + pd.Timedelta(days=offset)
        daily_vals = data[
            (data['datetime'] == day) &
            (data['station'].isin(neighbor_ids)) &
            (data[variable].notna())
        ][['station', variable]]

        values.extend(daily_vals[variable].values)
        sources.extend(daily_vals['station'].values)

    return values, sources

def weighted_imputation(values, distances):
    if not values or not distances:
        return np.nan
    weights = 1 / np.array(distances)
    weights /= weights.sum()  # Normalize weights
    return np.average(values, weights=weights)

def impute_partial_variables(station_id, data, metadata, variables, max_km, temporal_window=0, min_days_required=15, max_ele_meters=150):
    station_data = data[data['station'] == station_id].copy()
    target_station = metadata.loc[station_id]

    neighbors = find_nearest_stations(target_station, metadata, max_km, max_elev_diff_m=max_ele_meters)
    neighbor_ids = [n[0] for n in neighbors]
    distances = dict(neighbors)

    for variable in variables:
        dist_col = f'{variable}_DIST'
        station_data[dist_col] = 0.0  # default: 0 = not imputed

        for idx, row in station_data[station_data[variable].isna()].iterrows():
            vals, sources = get_temporal_neighbors(data, neighbor_ids, row['datetime'], variable, temporal_window)
            if not vals:
                continue

            dists = [distances[s] for s in sources if s in distances]
            if not dists:
                continue

            imputed = weighted_imputation(vals, dists)
            station_data.at[idx, variable] = imputed
            station_data.at[idx, dist_col] = min(dists)  # closest contributing neighbor

    # Return only if majority of desired vars are now filled, and if there are more than 15 days of data
    if (station_data[variables].isna().sum().sum() == 0) and (station_data['datetime'].nunique() > min_days_required):
        #print(station_data['datetime'].nunique()) debugging
        return station_data
    else:
        return None

def phone_a_friend(db, max_km=20, max_ele_meters=150, min_days_required = 15):
    #min days required added so stations have at least that many data points each
    included_stations = []
    all_station_data = []

    data = db.copy()
    metadata = data[['station', 'latitude', 'longitude', 'elevation']].drop_duplicates('station').set_index('station')

    station_ids = data['station'].unique()
    total = len(station_ids)
    print("Asking neighboring weather stations for missing data...\n")
    pbar = tqdm(total=total, ncols=100)

    for i, station_id in enumerate(station_ids, 1):
        pbar.update(1)
        station_df = data[data['station'] == station_id].copy()
        #if station_df['datetime'].nunique() < min_days_required:
        #    #print("a station did not have enough days of data. Removing it...")
        #    continue
        fixed_vars_ok = station_df[FIXED_VARS].notna().all().all()
        if not fixed_vars_ok:
            continue  # Must have TMAX and TMIN fully present

        # Determine which required variables are missing
        vars_to_impute = [v for v in REQUIRED_VARS if station_df[v].isna().any()]
        #if there are still variables to find or if there is not enough data days at this station
        #ask neighbors for info
        if vars_to_impute or (station_df['datetime'].nunique() < min_days_required):
            imputed = impute_partial_variables(
                station_id, data, metadata,
                vars_to_impute, max_km,
                temporal_window=0,
                min_days_required=min_days_required,
                max_ele_meters=max_ele_meters
            )

            if imputed is not None:
                for var in REQUIRED_VARS:
                    dist_col = f"{var}_DIST"
                    if dist_col not in imputed.columns:
                        imputed[dist_col] = 0.0
                all_station_data.append(imputed)
                included_stations.append(station_id)
            else:
                # Could not fully impute, but we still include the partially filled station
                for var in REQUIRED_VARS:
                    dist_col = f"{var}_DIST"
                    if dist_col not in station_df.columns:
                        station_df[dist_col] = 0.0
                all_station_data.append(station_df)
                included_stations.append(station_id)
        else:
            # No imputation needed, all vars were available
            for var in REQUIRED_VARS:
                dist_col = f"{var}_DIST"
                station_df[dist_col] = 0.0
            all_station_data.append(station_df)
            included_stations.append(station_id)

    final_dataset = pd.concat(all_station_data, ignore_index=True) if all_station_data else pd.DataFrame()

    print(f"\nTotal stations included: {len(included_stations)}")
    return final_dataset