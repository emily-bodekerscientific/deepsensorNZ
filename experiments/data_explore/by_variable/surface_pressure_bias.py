import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from tqdm import tqdm
from cartopy import crs as ccrs

from nzdownscale.dataprocess.config_local import DATA_PATHS
from nzdownscale.dataprocess.config import VAR_ERA5, VAR_STATIONS, STATION_LATLON
from nzdownscale.dataprocess import era5, stations, wrf


# Load data
print('Loading data...')
var = 'surface_pressure'
year = np.arange(2009, 2012)

process_era5 = era5.ProcessERA5()
era5_var = VAR_ERA5[var]['var_name']
era5_ds = process_era5.load_ds(var, year)
era5_ds = era5_ds.compute() / 100 # Pa to hPa

process_stations = stations.ProcessStations()
station_var = VAR_STATIONS[var]['var_name']
station_df = process_stations.load_stations(var, year)

# Drop NaNs and reset index
df = station_df.dropna().copy()
df = df.reset_index()

df['elevation'] = df['station_name'].map(lambda s: STATION_LATLON[s]['elevation'])

times = xr.DataArray(df['time'].values,      dims=['points'])
lats  = xr.DataArray(df['latitude'].values,  dims=['points'])
lons  = xr.DataArray(df['longitude'].values, dims=['points'])

# vectorized interpolation
interp_da = era5_ds[era5_var].interp(
    time=times,
    latitude=lats,
    longitude=lons,
    method='linear'        # or ‘nearest’ if you don’t need bilinear
)

# pull out a NumPy array in one go
df['era5_pres_interp'] = interp_da.values

# Drop NaNs after lookup
df = df.dropna(subset=['era5_pres_interp'])

# Calculate bias
df['bias'] = df['stn_lev_pres'] - df['era5_pres_interp']

# Save df
print('Saving surface pressure bias data')
df.to_csv(f"surface_pressure_bias.csv", index=False)

df = pd.read_csv(f"surface_pressure_bias.csv")
# Fit random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['latitude', 'longitude']]
y = df['bias']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Fitting Random Forest Regressor...')
bias_model = RandomForestRegressor(n_estimators=100)

# print('Fitting Linear Regression for bias correction...')
# bias_model = LinearRegression()  # Using Linear Regression for bias correction

bias_model.fit(X_train, y_train)

# Evaluate
print(f"Train RMSE: {mean_squared_error(y_train, bias_model.predict(X_train), squared=False):.2f} hPa")
print(f"Test RMSE: {mean_squared_error(y_test, bias_model.predict(X_test), squared=False):.2f} hPa")

print('Saving random forest model...')
# Save model
import pickle
with open('surface_pressure_bias_model_noelev.pkl', 'wb') as f:
    pickle.dump(bias_model, f)

# Calculate ERA5 bias correction
lats = era5_ds['latitude'].values     # 1D, shape (n_lat,)
lons = era5_ds['longitude'].values 
LAT2D, LON2D = np.meshgrid(lats, lons, indexing='ij')

flat_lat   = LAT2D.ravel()
flat_lon   = LON2D.ravel()

X_pred = np.vstack([flat_lat, flat_lon]).T
y_pred_flat = bias_model.predict(X_pred)

pred_2d = y_pred_flat.reshape(len(lats), len(lons))

bias_field = xr.DataArray(
    pred_2d,
    dims=('latitude', 'longitude'),
    coords={
        'latitude': lats,
        'longitude': lons
    },
    name='predicted_bias'
)

bias_field.to_netcdf('surface_pressure_bias_field.nc')