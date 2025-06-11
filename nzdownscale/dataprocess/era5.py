import os
from typing import Literal, List, Union
import glob
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, VAR_ERA5
from nzdownscale.dataprocess.config_local import DATA_PATHS


class ProcessERA5(DataProcess):

    def __init__(self) -> None:
        super().__init__()


    def load_ds(self, 
                var: Literal[tuple(VARIABLE_OPTIONS)],
                years: List=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            var (str): variable
            years (list): specific years, retrieves all if set to None
        """
        if type(years) == int:
            years = [years]
        elif type(years) == str:
            years = [int(years)]
        elif type(years) == list:
            years = [int(year) for year in years]
            if len(set(years)) == 1:
                years = [years[0]]
        else:
            ValueError (f'Years should be int, str or list, not {type(years)}')
        filenames = self.get_filenames(var, years)
        return xr.open_mfdataset(filenames)

    
    def ds_to_da(self,
                 ds: xr.Dataset,
                 var: Literal[tuple(VARIABLE_OPTIONS)],
                 ) -> xr.DataArray:
        """
        Extracts dataarray from dataset (variable data only, loses some metadata)
        If variable is temperature, converts from Kelvin to Celsius
        Args: 
            ds (xr.Dataset): dataset
            var (str): variable
        """
        da = ds[VAR_ERA5[var]['var_name']]
        if var == 'temperature':
            da = self.kelvin_to_celsius(da)
        return da


    def convert_hourly_to_daily(self, 
                                ds, 
                                function:Literal['mean', 'sum']='mean'):
        if function == 'mean':
            ds = ds.resample(time='D').mean()
        elif function == 'sum':
            ds = ds.resample(time='D').sum()
        else:
            raise ValueError(f'function={function} not recognised')
        # ! below line causes error in convNP run
        #ds['time'] = ds['time'].dt.strftime('%Y-%m-%d')
        return ds
    

    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)


    def get_parent_path(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ):
        if VAR_ERA5[var]['folder'] == 'NZ_land_processed_synctodatasets':
            parent = "parent_processed_synctodatasets"
        elif VAR_ERA5[var]['folder'] == 'NZ_land_processed':
            parent = "parent_processed"
        else:
            parent = "parent"
        return f'{DATA_PATHS["era5"][parent]}/{VAR_ERA5[var]["subdir"]}'
    

    def get_filenames(self,
                      var: Literal[tuple(VARIABLE_OPTIONS)],
                      years: List=None,
                      ) -> List[str]:
        """ Get list of ERA5 filenames for variable and list of years (if specified) """ 

        parent_path = self.get_parent_path(var)
        
        if VAR_ERA5[var]['folder'] == 'NZ_land': #'temperature': #surface pressure # 10m_u_component_of_wind # 10m_v_component_of_wind # 
            if years is None:
                filenames = glob.glob(f'{parent_path}/*/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames_year = glob.glob(f'{parent_path}/{year}/*.nc')
                    if len(filenames_year) == 0:
                        filenames_year = glob.glob(f'{parent_path}/{year}/*/*.nc')
                    filenames = filenames + filenames_year

        elif 'NZ_land_processed' in VAR_ERA5[var]['folder']:# == 'NZ_land_processed': #var == 'precipitation':  # relative_humidity
            if years is None:
                filenames = glob.glob(f'{parent_path}/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames = filenames + [f'{parent_path}/{fname}' for fname in os.listdir(parent_path) if str(year) in fname]

        return filenames

    def load_ds_time(self, 
                     var: Literal[tuple(VARIABLE_OPTIONS)],
                     time,
                     ) -> xr.Dataset:
        """ 
        Loads dataset with time dimension
        Args: 
            var (str): variable
            time: datetime obj
                    """
        if isinstance(time, datetime):
            year = [time.year]
        elif isinstance(time, (list, pd.DatetimeIndex)):
            year = set(t.year for t in time)
        filenames = self.get_filenames(var, year)
        ds = xr.open_mfdataset(filenames)
        if 'expver' in list(ds.coords):
            ds = ds.sel(expver=1)
            ds = ds.drop('expver')
        return ds.sel(time=time).load()


    def kelvin_to_celsius(self, da: xr.DataArray):
        return da - 273.15

    def surface_to_sea_level_pressure(
        self,
        da: xr.DataArray,
        elevation: xr.DataArray,
        temperature: Union[xr.DataArray, float, None] = None,
    ) -> xr.DataArray:
        """
        Convert surface pressure (Pa) to sea-level pressure (Pa) via hypsometric equation.

        Parameters
        ----------
        da : xr.DataArray
            The surface pressure field, in Pascal (Pa).
            Expected dims: (“time”, “latitude”, “longitude”).
        elevation : xr.DataArray
            Elevation in meters (m).  Expected dims: (“latitude”, “longitude”).
            Must cover at least the same lat/lon grid as `da`.
        temperature : float or xr.DataArray, optional
            If float, a single isothermal value in Kelvin (default 288 K).
            If xr.DataArray, must have a “time” dimension matching `da.time`.

        Returns
        -------
        p_slp : xr.DataArray
            Sea-level pressure, in Pascal (Pa), dims (“time”, “latitude”, “longitude”).

        Notes
        -----
        Hypsometric equation: 
            p_slp = p_surface * exp( g * h / (R_d * T) )
        where
        - p_surface in Pa
        - g = 9.80665 m/s²
        - R_d = 287.05 J/(kg·K)
        - h = elevation in meters
        - T = temperature in Kelvin
        """
        # 1) Default T = 288 K if not provided
        if temperature is None:
            temperature = 288.0

        # 2) Constants
        R_d = 287.05      # J/(kg·K) (gas constant for dry air)
        g   = 9.80665     # m/s²         (gravitational acceleration)

        # 3) Reindex elevation onto the EXACT lat/lon coords of da
        elevation = elevation.reindex(
            {'latitude': da.latitude, 'longitude': da.longitude},
            method='nearest'
        )

        # 4) Compute the exponent: exp( g * h / (R_d * T) ).
        #    If temperature is a float, (g * elevation)/(R_d * temperature) has dims (lat, lon).
        #    If temperature is a DataArray with dim “time”, Xarray will automatically align 
        #    on “time” and broadcast elevation up to (time, lat, lon) behind the scenes.
        exponent = (g * elevation) / (R_d * temperature)
        exp_term = np.exp(exponent)

        # 5) If exp_term has no “time” dimension, give it the same time coords as da:
        if 'time' not in exp_term.dims:
            exp_term = exp_term.expand_dims(time=da.time)

        # 6) Element‐wise multiply:  da(time,lat,lon) * exp_term(time,lat,lon)
        p_slp = da * exp_term

        # 7) Return in Pa:
        return p_slp

import xesmf as xe
def interpolate_era5(era5, ds, var):
    "Interpolate ERA5 to match the resolution of ds"
    era5_var = VAR_ERA5[var]['var_name']
    if type(era5) is xr.Dataset:
        era5 = era5[era5_var]
    
    ds_out = xr.Dataset({
        'latitude': (['latitude'], ds.latitude.values),
        'longitude': (['longitude'], ds.longitude.values),
    })

    # method = 'bilinear'
    # Ny_in, Nx_in = len(era5.latitude), len(era5.longitude)
    # Ny_out, Nx_out = len(ds.latitude), len(ds.longitude)
    # filename = f'{method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc'
    # filepath = os.path.join(DATA_PATHS['regridder_weights']['parent'], filename)
    # if os.path.exists(filepath):
    #     regridder = xe.Regridder(hold.isel(Time=0), 
    #                                 ds_out, 
    #                                 'bilinear', 
    #                                 reuse_weights=True,
    #                                 filename=filepath
    #                                 )
    # else:
    regridder = xe.Regridder(era5.isel(time=0), 
                            ds_out, 
                            'bilinear', 
                            reuse_weights=False,
                            # filename=filepath
                            )
    # regridder.to_netcdf(filepath)
    
    interp_hold = regridder(era5)
    return interp_hold


if __name__ == '__main__':
    pass