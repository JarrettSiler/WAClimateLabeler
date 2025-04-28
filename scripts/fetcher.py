import os, time, requests, signal, sys, atexit, webbrowser
from dotenv import load_dotenv, set_key
import pandas as pd
from tqdm import tqdm
from scripts.ui_tools import file_gen_settings, map_stations

def fetch_from_NOAA(START_DATE, END_DATE, SAVEFILE_ISDATE, rewrite_good_stations = True, open_map = False):

    load_dotenv() # Load environment variables from .env file
    env_file = "../.env"
    API_TOKEN = os.getenv("NOAA_API_TOKEN")
    HEADERS = {"token": API_TOKEN}
    delay_between_requests = 1 / 5  # 5 requests per second max allowance
    max_daily_requests = 10000  # 10,000 requests per day max allowance
    requests_run = int(os.getenv("REQUESTS_TODAY"))  # Initialize request counter

    #rewrite_good_stations = True # Set to True to rewrite the good stations file
    #open_map = False # Set to True to open the map of obtained stations after processing

    #NOAA: Each token will be limited to five requests per second and 10,000 requests per day

    state_fips_df = pd.read_csv("data/ref/state_fips.csv")
    state_fips_dict = pd.Series(state_fips_df.FIPS.values, index=state_fips_df.state).to_dict()

    def get_fips(state):
        return state_fips_dict.get(state, "State not found")

    # Parameters from .env file
    STATE = os.getenv("STATE")
    WEATHERDATA_DIR = os.getenv("WEATHERDATA_DIR")

    #pre-specified dates
    if SAVEFILE_ISDATE:
        pass
    else:
        START_DATE, END_DATE, SAVEFILE_ISDATE = file_gen_settings()

    DATASET = "GHCND" # Global Historical Climatology Network Daily
    GOODSTATION_DIR = os.getenv("GOODSTATION_DIR")
    INTERACTIVE_MAPS_DIR = os.getenv("INTERACTIVE_MAPS_DIR")

    STATE_FIPS = get_fips(STATE)  # Washington State

    # =========================Saves REQUESTS_RUN to .env file==================
    def save_progress():
        print("\nSaving request count before exit...")
        set_key(env_file, "REQUESTS_TODAY", str(requests_run))
        print(f"Total requests made: {requests_run}")

    def setup_handlers():
        signal.signal(signal.SIGINT, lambda s, f: save_progress() or sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: save_progress() or sys.exit(0))
        atexit.register(save_progress)

    setup_handlers()

    # ======================== GET ALL STATIONS IN WASHINGTON ==================

    def get_ghcnd_stations(state_fips):
        url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
        limit = 1000
        offset = 1
        all_results = []
        print(f"Getting all GHCND stations in {STATE}... this could take a bit...")

        while True:
            params = {
                "datasetid": DATASET,
                "locationid": f"FIPS:{state_fips}",
                "limit": limit,
                "offset": offset
            }
            response = requests.get(url, headers=HEADERS, params=params)

            if response.status_code in [429, 503]:
                print(f"Status {response.status_code} received. Waiting and retrying...")
                time.sleep(3)
                continue

            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                break
            try:
                results = response.json().get("results", [])
                if not results:
                    break  # No more data
                all_results.extend(results)
                offset += limit
            except ValueError as e:
                print(f"Error decoding JSON: {e}")
                break

        print(f"Found {len(all_results)} GHCND stations in {STATE}")
        return all_results

    #=================================================================

    def get_daily_data(station_id, start_date, end_date):
        url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

        params = {
            "datasetid": DATASET,
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "units": "standard",
            "limit": 1000
        }
        response = requests.get(url, headers=HEADERS, params=params, timeout=6)
        return response.json().get("results", [])

    # === MAIN LOGIC ===

    stations = get_ghcnd_stations(STATE_FIPS)

    good_station_file = os.path.join(GOODSTATION_DIR, f"good_stations_{STATE}_{SAVEFILE_ISDATE}.txt")
    try: #try to re-use the good stations file
        with open(good_station_file, "r") as f:
            good_station_ids = set(line.strip() for line in f.readlines())
        stations = [s for s in stations if s["id"] in good_station_ids]
        print(f"Using {len(stations)} verified good stations.")
    except FileNotFoundError:
        print("No saved good stations file found. Processing all stations.")

    all_data = []

    try:
        with tqdm(stations, desc="Processing Stations", unit="station", 
                postfix={"station": ""}) as pbar:  # Initialize postfix dict
        
            for station in pbar:
                station_id = station["id"]
                station_name = station.get("name", "unnamed")[:15]  # Truncate long names
                pbar.set_postfix(station_id=station_id, station=station_name)

                if not station_id.startswith('GHCND:US'):
                    continue # Skip non-US stations
                try:
                    data = get_daily_data(station_id, START_DATE, END_DATE)
                    # Skip stations with a lack of daily recordings during this month
                    if len(data) < 10: #10 days minimum
                        print(f"Skipping {station_id} — not enough records")
                        continue  # not enough data
                    unique_datatypes = set(d["datatype"] for d in data)
                    useful_vars = {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD"} #variables needed
                    if not useful_vars & unique_datatypes:
                        print(f"Skipping {station_id} — no useful datatypes: {unique_datatypes}")
                        continue  # skip if none of the useful variables are present

                    for record in data:
                        try:
                            dt = pd.to_datetime(record["date"])
                            all_data.append({
                                "station": station_id,
                                "datetime": dt,
                                "datatype": record["datatype"],
                                "value": record["value"]
                            })
                        except Exception as parse_error:
                            print(f"\nSkipping malformed record from {station_id}: {parse_error}")

                    requests_run += 1
                    
                    if requests_run % 100 == 0:
                        print(f"\nRequests made: {requests_run}")  # Newline keeps progress bar clean
                    
                    time.sleep(delay_between_requests)
                    
                except Exception as e:
                    print(f"\nError with {station_id}: {e}")

    finally:
        print("A fatal error occurred in scraper.py")

    # === Format Data ===
    df = pd.DataFrame(all_data)

    # Filter to core columns before pivoting
    core_datatypes = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"] #values that matter
    df = df[df["datatype"].isin(core_datatypes)]
    stations_df = pd.DataFrame(stations)[["id", "latitude", "longitude", "elevation"]]
    stations_df = stations_df.rename(columns={"id": "station"})

    df_pivot = df.pivot_table(
        index=["station", "datetime"],
        columns="datatype",
        values="value",
        aggfunc="first"
    ).reset_index()

    df_pivot = df_pivot.merge(
        stations_df,
        on="station",
        how="left"
    )

    # Saving the stations that return the appropriate data to a file will help reduce the 
    # number of requests in the future
    if rewrite_good_stations:
        good_stations = df_pivot["station"].unique()
        good_station_file = os.path.join(GOODSTATION_DIR, f"good_stations_{STATE}_{SAVEFILE_ISDATE}.txt")
        with open(good_station_file, "w") as f:
            for s in good_stations:
                f.write(f"{s}\n")
        print(f"Saved {len(good_stations)} stations to be reused in the future to {good_station_file}")

    print("Database shape:", df_pivot.shape)
    print("Weatherstation data columns:", df_pivot.columns)
    print("Number of stations:", df_pivot['station'].nunique())
    print("Number of Pull Requests to build Database:", requests_run)

    # === Save Output ===
    filename = os.path.join(WEATHERDATA_DIR, f"{STATE}_{SAVEFILE_ISDATE}.csv")
    df_pivot.to_csv(filename, index=False)
    print(f"Daily weather data for {START_DATE} to {END_DATE} for {STATE} written to {filename}")
    print("CSV file size:", round(os.path.getsize(filename)/1024/1000,2), "MB")

    if open_map:
        map_stations(df_pivot, state=STATE, output_path=f"{INTERACTIVE_MAPS_DIR}/{STATE}_obtained_station_map.html")
        webbrowser.open('file://' + os.path.realpath(f"{INTERACTIVE_MAPS_DIR}/{STATE}_obtained_station_map.html"))