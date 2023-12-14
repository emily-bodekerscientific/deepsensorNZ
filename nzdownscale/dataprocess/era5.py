import os
from typing import Literal, List
import glob

import xarray as xr

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, DATA_PATHS, VAR_ERA5


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
        filenames = self.get_filenames(var, years)
        return xr.open_mfdataset(filenames)

    
    def ds_to_da(self,
                 ds: xr.Dataset,
                 var: Literal[tuple(VARIABLE_OPTIONS)],
                 ) -> xr.DataArray:
        """
        Extracts dataarray from dataset (variable data only, loses some metadata)
        Args: 
            ds (xr.Dataset): dataset
            var (str): variable
        """
        return ds[VAR_ERA5[var]['var_name']]
    

    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)


    def get_parent_path(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ):
        return f'{DATA_PATHS["ERA5"]["parent"]}/{VAR_ERA5[var]["subdir"]}'
    

    def get_filenames(self,
                      var: Literal[tuple(VARIABLE_OPTIONS)],
                      years: List=None,
                      ) -> List[str]:
        """ Get list of ERA5 filenames for variable and list of years (if specified) """ 

        parent_path = self.get_parent_path(var)
        
        if var == 'temperature':
            if years is None:
                filenames = glob.glob(f'{parent_path}/*/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames = filenames + glob.glob(f'{parent_path}/{year}/*.nc')
        
        elif var == 'precipitation':  
            if years is None:
                filenames = glob.glob(f'{parent_path}/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames = filenames + [f'{parent_path}/{fname}' for fname in os.listdir(parent_path) if str(year) in fname]
        
        return filenames


if __name__ == '__main__':
    pass