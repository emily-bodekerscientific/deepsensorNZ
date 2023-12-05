import os
from typing import Literal, List
import glob

import xarray as xr

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, DATA_PATHS, DIR_ERA5


class ProcessERA5(DataProcess):

    def __init__(self) -> None:
        super().__init__()


    def load_ds(self, 
                var:Literal[tuple(VARIABLE_OPTIONS)],
                year:int=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            var (str): variable
            year (int): specific year, retrieves all if set to None
        """
        filenames = self.get_filenames(var, year)
        return xr.open_mfdataset(filenames)

    
    def ds_to_da(self,
                 ds:xr.Dataset,
                 var:Literal[tuple(VARIABLE_OPTIONS)],
                 ) -> xr.DataArray:
        """
        Extracts dataarray from dataset (variable data only, loses some metadata)
        Args: 
            ds (xr.Dataset): dataset
            var (str): variable
        """
        #return ds[self.names[var]['var_name']]
        return ds[DIR_ERA5[var]['var_name']]
    

    def get_filenames(self,
                      var:Literal[tuple(VARIABLE_OPTIONS)],
                      year:int=None,
                      ) -> List[str]:


        parent_path = f'{DATA_PATHS["ERA5"]}/{DIR_ERA5[var]["subdir"]}'
        
        if var == 'temperature':
            if year is None:
                filenames = glob.glob(f'{parent_path}/*/*.nc')
            else:
                filenames = glob.glob(f'{parent_path}/{year}/*.nc')
        
        elif var == 'precipitation':  
            if year is None:
                filenames = glob.glob(f'{parent_path}/*.nc')
            else:
                filenames = [f'{parent_path}/{fname}' for fname in os.listdir(parent_path) if str(year) in fname]
        
        return filenames


if __name__ == '__main__':
    pass