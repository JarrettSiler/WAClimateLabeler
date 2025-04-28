import os, calendar
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from dotenv import load_dotenv
import folium, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.patches import Circle, Patch
from sklearn.neighbors import KernelDensity


load_dotenv()
# Set the path to the directory where your CSVs are stored
WEATHERDATA_DIR = os.getenv("WEATHERDATA_DIR")
INTERACTIVE_MAPS_DIR = os.getenv("INTERACTIVE_MAPS_DIR")
STATE = os.getenv("STATE")

#============================== DATE SELECTOR & FORMATING =========================
def get_dates(year, month):
    
    start_date = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{end_day:02d}"
    #print(f"{month_name},{start_date},{end_date}")
    return start_date, end_date

def file_exists(year, month):
    filename = f"{STATE}_{year}_{month}.csv"
    return os.path.isfile(os.path.join(WEATHERDATA_DIR, filename))

def show_selector():
    result = {}
    def update_status(*args):
        year = year_var.get()
        month = month_var.get()

        if year and month:
            if file_exists(year, month):
                status_label.config(text="Generated", background="green", foreground="white")
            else:
                status_label.config(text="Not Generated", background="red", foreground="white")

    def submit():
        result['year'] = int(year_var.get())
        # Convert month name to number (e.g. 'January' -> 1)
        result['month'] = datetime.strptime(month_var.get(), "%B").month
        popup.destroy()

    popup = tk.Tk()
    popup.title("Select Year and Month")
    popup.geometry("350x200")
    popup.resizable(False, False)

    # Year
    tk.Label(popup, text="Select Year:").grid(row=0, column=0, pady=(10, 0), padx=10, sticky="w")
    year_var = tk.StringVar()
    current_year = datetime.now().year
    years = [str(y) for y in range(2024, 2017, -1)]
    year_menu = ttk.Combobox(popup, textvariable=year_var, values=years, state="readonly")
    year_menu.set(str(2024))
    year_menu.grid(row=0, column=1, pady=(10, 0), sticky="w")

    # Month
    tk.Label(popup, text="Select Month:").grid(row=1, column=0, pady=(10, 0), padx=10, sticky="w")
    month_var = tk.StringVar()
    months = list(calendar.month_name)[1:]
    month_menu = ttk.Combobox(popup, textvariable=month_var, values=months, state="readonly")
    month_menu.set('January')
    month_menu.grid(row=1, column=1, pady=(10, 0), sticky="w")

    # Status
    status_label = tk.Label(popup, text="", width=15)
    status_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))

    # Submit button
    ttk.Button(popup, text="Submit", command=submit).grid(row=3, column=0, columnspan=2, pady=15)

    # Update status when dropdowns change
    year_var.trace_add("write", update_status)
    month_var.trace_add("write", update_status)

    update_status()  # Initial status
    popup.mainloop()
    return result.get('year'), result.get('month')

def file_gen_settings(): #THE FUNCTION TO IMPORT
    year,month = show_selector()
    filenametags = f"{year}_{month}"
    start,end = get_dates(year,month)
    #print(start,end,filename)
    return start,end,filenametags

#file_gen_settings() ---------------------------------------------------------------------
#==============================================================================

#============================== FOLIUM MAPS ==================================
def map_stations(df_pivot, state="WA", output_path="station_map.html"):
    # Choose a center point (could be improved with mean lat/lon)
    wa_center = [47.5, -120.5] if state.upper() == "WA" else [39.5, -98.35]  # fallback: center of US

    m = folium.Map(location=wa_center, zoom_start=6)

    for _, row in df_pivot.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isnull(lat) or pd.isnull(lon):
            continue  # Skip rows without coordinates

        popup_content = (
            f"<b>Station:</b> {row['station']}<br>"
            f"<b>Lat:</b> {lat:.2f}, <b>Lon:</b> {lon:.2f}<br>"
            f"<b>Elevation:</b> {row.get('elevation', 'N/A')} m<br>"
            f"<b>TMAX:</b> {row.get('TMAX', 'N/A')}<br>"
            f"<b>TMIN:</b> {row.get('TMIN', 'N/A')}<br>"
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=250),
            icon = folium.Icon(color="green", icon="caret-up", prefix="fa")
        ).add_to(m)

    m.save(output_path)
    print(f"Map saved to {output_path}")

#manually run the function
#df = pd.read_csv(f"{WEATHERDATA_DIR}/Washington_2024-09-01_2024-09-28.csv")  # Load DataFrame
#map_stations(df, state="WA", output_path=f"{INTERACTIVE_MAPS_DIR}/station_map_test.html")
#webbrowser.open('file://' + os.path.realpath(f"{INTERACTIVE_MAPS_DIR}/station_map_test.html"))  # Open the map in a web browser
#==============================================================================
#another plot for data relevance

def plot_station_data_availability(df, image_path="images/washington-state-map.jpg"):
    """
    Plots the station locations in Washington based on available weather variables.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns: 'station', 'latitude', 'longitude', 
                           and any of ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'].
        image_path (str): Path to the Washington state map image.
    """
    # Create binary presence indicators
    for var in ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']:
        df[var] = df[var].notna().astype(int)

    # Generate binary signature and decode it into a human-readable label
    var_order = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']
    df['signature'] = df[var_order].astype(str).agg(''.join, axis=1)
    df['label'] = df['signature'].apply(
        lambda sig: ' + '.join([var for bit, var in zip(sig, var_order) if bit == '1'])
    )

    # Load map image
    img = mpimg.imread(image_path)
    extent = [-124.87, -116.71, 45.15, 49.45]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.imshow(img, extent=extent, aspect='auto', alpha=0.5)

    # Drop duplicates to avoid plotting multiple rows per station
    unique_stations = df.drop_duplicates('station')

    sns.scatterplot(
        data=unique_stations,
        x='longitude',
        y='latitude',
        hue='label',
        palette='tab10',
        edgecolor='black'
    )

    plt.title("Station Locations by Data Availability")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Data Available", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------

def plot_stations(df, image_path="images/washington-state-map.jpg", rings=False, k=1, show_climates=False):

    img = mpimg.imread(image_path)
    extent = [-124.87, -116.71, 45.15, 49.45]

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.imshow(img, extent=extent, aspect='auto', alpha=0.5)

    unique_stations = df.drop_duplicates('station')

    if (k > 1) and show_climates:
        #unique climates and colors
        n_colors = len(unique_stations['climate'].unique())
        palette = sns.color_palette('muted', n_colors=n_colors)

        climate_labels = unique_stations['climate'].unique()
        climate_to_color = {climate: palette[i] for i, climate in enumerate(climate_labels)}

        #plot colored by climate
        sns.scatterplot(data=unique_stations, x='longitude', y='latitude',
                        hue='climate', palette=climate_to_color, edgecolor='black', ax=ax, legend=False)

        #custom legend
        handles = [Patch(color=color, label=climate) for climate, color in climate_to_color.items()]
        cluster_legend = ax.legend(handles=handles,
                                   title="Climate Type", loc='upper right',
                                   bbox_to_anchor=(1.02, 1), borderaxespad=0,
                                   frameon=True)
        ax.add_artist(cluster_legend)

        # ADD DENSITY SHADING BASED ON CLIMATE
        for climate_type in climate_labels:
            climate_points = unique_stations[unique_stations['climate'] == climate_type][['longitude', 'latitude']].values

            if len(climate_points) < 2:
                continue  #skip small groups

            kde = KernelDensity(bandwidth=0.1)
            kde.fit(climate_points)

            #grid
            x_min, x_max = climate_points[:, 0].min() - 0.5, climate_points[:, 0].max() + 0.5
            y_min, y_max = climate_points[:, 1].min() - 0.5, climate_points[:, 1].max() + 0.5
            xgrid, ygrid = np.meshgrid(np.linspace(x_min, x_max, 100),
                                       np.linspace(y_min, y_max, 100))
            grid_samples = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
            dens = np.exp(kde.score_samples(grid_samples)).reshape(xgrid.shape)

            fill_color = climate_to_color[climate_type]
            ax.contour(xgrid, ygrid, dens, levels=[.07], colors=[fill_color], linewidths=2)
    else:
        sns.scatterplot(data=unique_stations, x='longitude', y='latitude',
                        hue='elevation', palette='viridis', edgecolor='black', ax=ax)
        elev_legend = ax.legend(title="Elevation (m)", loc='upper left', frameon=True)
        ax.add_artist(elev_legend)

    #draw rings
    if rings:
        ring_info = {
            'PRCP': 'red',
            'SNOW': 'green',
            'SNWD': 'blue'
        }

        for _, row in unique_stations.iterrows():
            lon, lat = row['longitude'], row['latitude']
            for var, color in ring_info.items():
                dist_col = f'{var}_DIST'
                if dist_col in row and row[dist_col] > 0:
                    radius_deg = row[dist_col] / 111.0  # km to degrees approx
                    circle = Circle((lon, lat), radius_deg,
                                    edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.8)
                    ax.add_patch(circle)

        #legend for rings
        ring_handles = [Patch(edgecolor=color, facecolor='none', label=f"reach to obtain {var} data", linewidth=1.5)
                        for var, color in ring_info.items()]
        ax.legend(handles=ring_handles, title="Reach to Neighbors", loc='lower right', frameon=True)

    plt.title("Station Locations" + (" by Climate" if k > 1 else " by Elevation"))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()