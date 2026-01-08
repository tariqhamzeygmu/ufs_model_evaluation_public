# ---------------------------------------------------------------------------------------------------------------------
#  Filename: oni.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: Define a class that organizes information about Oceanic Niño Index events.
# ---------------------------------------------------------------------------------------------------------------------

import os
import sys
import warnings
from typing import Optional, Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from ..regridder import Regrid
from ..datareader import datareader as dr
from . import stats, rws, timeutil


CUSTOM_CMAPS = {'custom_btr': [
    (36 / 255, 0 / 255, 216 / 255),
    (24 / 255, 28 / 255, 247 / 255),
    (40 / 255, 87 / 255, 255 / 255),
    (61 / 255, 135 / 255, 255 / 255),
    (86 / 255, 176 / 255, 255 / 255),
    (117 / 255, 211 / 255, 255 / 255),
    (153 / 255, 234 / 255, 255 / 255),
    (188 / 255, 249 / 255, 255 / 255),
    (234 / 255, 255 / 255, 255 / 255),
    (255 / 255, 255 / 255, 255 / 255),
    (255 / 255, 255 / 255, 234 / 255),
    (255 / 255, 241 / 255, 188 / 255),
    (255 / 255, 214 / 255, 153 / 255),
    (255 / 255, 172 / 255, 117 / 255),
    (255 / 255, 120 / 255, 86 / 255),
    (255 / 255, 61 / 255, 61 / 255),
    (247 / 255, 39 / 255, 53 / 255),
    (216 / 255, 21 / 255, 47 / 255),
    (165 / 255, 0 / 255, 33 / 255)],
    'beta_star': [
    (107 / 255, 107 / 255, 107 / 255),
    (179 / 255, 179 / 255, 179 / 255),
    (251 / 255, 246 / 255, 249 / 255),
    (237 / 255, 213 / 255, 227 / 255),
    (226 / 255, 190 / 255, 211 / 255),
    (219 / 255, 175 / 255, 201 / 255),
    (209 / 255, 154 / 255, 186 / 255),
    (198 / 255, 135 / 255, 171 / 255),
    (155 / 255, 68 / 255, 119 / 255),
    (148 / 255, 98 / 255, 41 / 255),
    (166 / 255, 146 / 255, 58 / 255),
    (176 / 255, 181 / 255, 74 / 255),
    (161 / 255, 207 / 255, 105 / 255),
    (155 / 255, 227 / 255, 134 / 255),
    (157 / 255, 242 / 255, 158 / 255),
    (126 / 255, 222 / 255, 196 / 255),
    (92 / 255, 183 / 255, 196 / 255),
    (61 / 255, 108 / 255, 168 / 255),
    (42 / 255, 29 / 255, 133 / 255)],
    'beta_star_diff': [
    (155 / 255, 68 / 255, 119 / 255),
    (169 / 255, 88 / 255, 135 / 255),
    (184 / 255, 110 / 255, 153 / 255),
    (198 / 255, 135 / 255, 171 / 255),
    (209 / 255, 154 / 255, 186 / 255),
    (219 / 255, 175 / 255, 201 / 255),
    (226 / 255, 190 / 255, 211 / 255),
    (237 / 255, 213 / 255, 227 / 255),
    (251 / 255, 246 / 255, 249 / 255),
    (255 / 255, 255 / 255, 255 / 255),
    (240 / 255, 255 / 255, 243 / 255),
    (200 / 255, 248 / 255, 225 / 255),
    (169 / 255, 239 / 255, 209 / 255),
    (130 / 255, 224 / 255, 195 / 255),
    (112 / 255, 212 / 255, 200 / 255),
    (92 / 255, 183 / 255, 196 / 255),
    (61 / 255, 108 / 255, 168 / 255),
    (51 / 255, 80 / 255, 158 / 255),
    (42 / 255, 29 / 255, 133 / 255)],
    'Ks': [
    (194 / 255, 194 / 255, 250 / 255),
    (219 / 255, 219 / 255, 255 / 255),
    (239 / 255, 239 / 255, 255 / 255),
    (219 / 255, 255 / 255, 219 / 255),
    (186 / 255, 245 / 255, 186 / 255),
    (252 / 255, 237 / 255, 128 / 255),
    (227 / 255, 209 / 255, 0 / 255),
    (250 / 255, 176 / 255, 125 / 255),
    (255 / 255, 128 / 255, 0 / 255),
    (255 / 255, 0 / 255, 0 / 255),
    (110 / 255, 14 / 255, 20 / 255)],
    'Ks_diff': [
    (84 / 255, 82 / 255, 0 / 255),
    (120 / 255, 112 / 255, 0 / 255),
    (161 / 255, 150 / 255, 0 / 255),
    (199 / 255, 186 / 255, 43 / 255),
    (227 / 255, 209 / 255, 0 / 255),
    (240 / 255, 220 / 255, 15 / 255),
    (244 / 255, 229 / 255, 75 / 255),
    (247 / 255, 236 / 255, 120 / 255),
    (250 / 255, 242 / 255, 164 / 255),
    (255 / 255, 255 / 255, 255 / 255),
    (255 / 255, 235 / 255, 250 / 255),
    (250 / 255, 227 / 255, 240 / 255),
    (247 / 255, 204 / 255, 230 / 255),
    (245 / 255, 173 / 255, 214 / 255),
    (240 / 255, 138 / 255, 194 / 255),
    (217 / 255, 92 / 255, 163 / 255),
    (189 / 255, 0 / 255, 130 / 255),
    (153 / 255, 0 / 255, 107 / 255),
    (117 / 255, 0 / 255, 82 / 255)],
    'rws': [
    (0 / 255, 102 / 255, 102 / 255),
    (0 / 255, 153 / 255, 153 / 255),
    (0 / 255, 204 / 255, 204 / 255),
    (51 / 255, 217 / 255, 217 / 255),
    (102 / 255, 230 / 255, 230 / 255),
    (140 / 255, 240 / 255, 240 / 255),
    (175 / 255, 246 / 255, 246 / 255),
    (200 / 255, 255 / 255, 255 / 255),
    (229 / 255, 255 / 255, 255 / 255),
    (255 / 255, 255 / 255, 255 / 255),
    (255 / 255, 240 / 255, 220 / 255),
    (255 / 255, 229 / 255, 203 / 255),
    (255 / 255, 202 / 255, 153 / 255),
    (255 / 255, 173 / 255, 101 / 255),
    (255 / 255, 142 / 255, 51 / 255),
    (255 / 255, 110 / 255, 0 / 255),
    (204 / 255, 85 / 255, 0 / 255),
    (153 / 255, 61 / 255, 0 / 255),
    (102 / 255, 39 / 255, 0 / 255)]
}

# Year and highest ONI recorded *in its strength category*
elnino_events = (
    (1951, 1.2),
    (1952, 0.8),
    (1953, 0.8),
    (1957, 1.8),
    (1958, 0.6),
    (1963, 1.4),
    (1965, 1.9),
    (1968, 1.1),
    (1969, 0.9),
    (1972, 1.8),
    (1976, 0.9),
    (1977, 0.8),
    (1979, 0.6),
    (1982, 2.2),
    (1986, 1.2),
    (1987, 1.7),
    (1991, 1.7),
    (1994, 1.1),
    (1997, 2.4),
    (2002, 1.3),
    (2004, 0.7),
    (2006, 0.94),
    (2009, 1.36),
    (2014, 0.93),
    (2015, 2.64),
    (2018, 0.90),
    (2023, 1.95)
)

# Year and highest ONI recorded *in its strength category*
lanina_events = (
    (1954, -0.9),
    (1955, -1.4),
    (1964, -0.8),
    (1970, -1.4),
    (1971, -0.9),
    (1973, -1.9),
    (1974, -0.8),
    (1975, -1.7),
    (1983, -0.9),
    (1984, -0.9),
    (1988, -1.8),
    (1995, -1.0),
    (1998, -1.6),
    (1999, -1.7),
    (2000, -0.7),
    (2005, -0.85),
    (2007, -1.64),
    (2008, -0.85),
    (2010, -1.64),
    (2011, -1.09),
    (2016, -0.69),
    (2017, -0.97),
    (2020, -1.27),
    (2021, -1.06),  # We sure about this one?
    (2022, -0.99)
)


class ONI:
    '''
    Oceanic Niño Index
    Weak:        0.5 to 0.9 SST anomaly
    Moderate:    1.0 to 1.4 SST anomaly
    Strong:      1.5 to 1.9 SST anomaly
    Very Strong: ≥ 2.0      SST anomaly
    '''

    def __init__(self, year: int, oni: float):

        if not isinstance(year, int):
            raise ValueError(f'Wrong data type year=integer')

        if year <= 1900 or year >= 2100:
            raise ValueError(f"year value ({year}) is outside our range of consideration.")

        if not isinstance(oni, (float, int)):
            raise ValueError(f'oni must be a number.')

        if oni <= -10 or oni >= 10:
            raise ValueError(f"oni value ({oni}) is physically unlikely.")

        self._oni = oni
        self._oni_magnitude = abs(oni)
        self._year = year

        # Default values for normal seasons.
        self._event_code = 0
        self._event = ''
        self._strength_code = 0  # 1=weak, 2=moderate, 3=strong, 4=very strong

        if oni < 0:
            self._event = 'LaNina'
            self._event_code = 1
        elif oni > 0:
            self._event = 'ElNino'
            self._event_code = -1

        # Calculate strength
        if self._event_code != 0:
            if 0.5 < self._oni_magnitude < 1.0:
                self._strength_code = 1

            elif 1.0 <= self._oni_magnitude < 1.5:
                self._strength_code = 2

            elif 1.5 <= self._oni_magnitude < 2.0:
                self._strength_code = 3

            elif self._oni_magnitude >= 2.0:
                self._strength_code = 4

        # Readable label
        self._strength = [None, 'Weak', 'Moderate', 'Strong', 'Very Strong'][self._strength_code]

    def __repr__(self):

        msg = f'Oceanic Niño Index object\n'
        msg += f'Event:    {self._event}\n'
        msg += f'Year:     {self._year}\n'
        msg += f'ONI:      {self._oni}\n'
        msg += f'Strength: {self._strength}\n'
        msg += f'Get characteristics of this ONI object with: <your_oni_object>.get(<attribute_name>)\n'

        return msg

    def get(self, att):
        '''Every attribute must be lower case.'''

        att = att.lower()

        try:
            return getattr(self, att)
        except AttributeError:
            att = f'_{att}'
            return getattr(self, att)


def prep_oni_datasets(statistics: Union[str, List[str]],
                      ufs_model: str,
                      ufs_var: str,
                      verif_var: str,
                      time_range: Tuple[str],
                      initmonth: int,
                      leads: tuple,
                      elnino_years: list,
                      lanina_years: list,
                      lev: Optional[float] = None,
                      ufs_scaling_factor: Optional[float] = None,
                      verif_scaling_factor: Optional[float] = None) -> Tuple[xr.Dataset]:

    '''Prepare datasets used for enso-teleconnections diagnostics.'''

    # First, process statistics argument
    available_statistics = ['anomaly',
                            'restoring effect',
                            'stationary wave number',
                            'rossby wave source']

    # Check data type
    if isinstance(statistics, str):
        statistics = [statistics]

    if not isinstance(statistics, list):
        msg = f"statistics must be a string value or a list of string values in {available_statistics}"
        raise ValueError(msg)

    if len(statistics) == 0:
        raise ValueError(f'Must enter statistics= one or more of {available_statistics}')

    # Coerce every statistic name to lower case
    statistics = [stat.lower() for stat in statistics]

    # if statistic not in available_statistics:
    not_available = set(statistics).difference(available_statistics)
    if len(not_available) > 0:
        msg = f"{not_available} is not available. statistics must be one or more of {available_statistics}'"
        raise ValueError(msg)

    # Process 'leads'.  DataReader expects to see (min_value, max_value) when slicing.
    min_lead = min(leads)
    max_lead = max(leads)
    leads = (min_lead, max_lead)

    # --- BEGIN WORK ---

    # Get UFS data reader
    ufs_data_reader = dr.getDataReader(datasource='UFS',
                                       filename=f'experiments/phase_1/{ufs_model}/atm_monthly.zarr',
                                       model='atm')

    # Get VERIF data reader (This is hardcoded to ERA5, for now...
    verif_data_reader = dr.getDataReader(datasource='ERA5')

    # Initialize Regridder
    regridder = Regrid.Regrid(data_reader1=ufs_data_reader,
                              data_reader2=verif_data_reader,
                              method='bilinear')

    # The user may be requesting a WIND field, in which case we need the orthogonal field as well.
    # Do this for VERIF
    other_wind_field = None
    for wind_set in verif_data_reader.WINDS:
        # Assume each set has 2 wind fields, but don't assume how the keys are named.
        var_keys = list(wind_set.keys())
        if verif_var in wind_set.values():

            if wind_set[var_keys[0]] == verif_var:
                other_wind_field = wind_set[var_keys[1]]

            elif wind_set[var_keys[1]] == verif_var:
                other_wind_field = wind_set[var_keys[0]]

    # Insert verif_var and other_wind_field into a list.
    # If other_wind_field is still None, just filter it out.
    verif_vars = [verif_var, other_wind_field]
    verif_vars = list(filter(None, verif_vars))

    # Do this for UFS
    other_wind_field = None
    for wind_set in ufs_data_reader.WINDS:
        # Assume each set has 2 wind fields, but don't assume how the keys are named.
        var_keys = list(wind_set.keys())
        if ufs_var in wind_set.values():

            if wind_set[var_keys[0]] == ufs_var:
                other_wind_field = wind_set[var_keys[1]]

            elif wind_set[var_keys[1]] == ufs_var:
                other_wind_field = wind_set[var_keys[0]]

    # Insert verif_var and other_wind_field into a list.
    # If other_wind_field is still None, just filter it out.
    ufs_vars = [ufs_var, other_wind_field]
    ufs_vars = list(filter(None, ufs_vars))

    # Temporally resample VERIF data to monthly resolution to match UFS
    regridder.resample(var=verif_vars,
                       lev=lev,
                       time=time_range,
                       use_mp=True)

    # Spatially regrid higher-resolution dataset down to lower-resolution dataset.

    # This is a very crude way of determining which dataset needs to be regridded and which result to extract.
    # Basically, if there is any error at all, try the other dataset. This could and should be improved!
    try:
        # Try regridding Verif onto UFS
        regridder.regrid(var=verif_vars,
                         lev=lev,
                         time=time_range)

        # Convert temporal coordinates to init_lead
        regridder.align()

        # Get VERIF dataset. lev has already been selected in resample.
        verif_ds = regridder.aligned.retrieve(var=verif_vars,
                                              time=time_range,
                                              lead=leads,
                                              initmonths=initmonth)

        ufs_ds = ufs_data_reader.retrieve(var=ufs_vars,
                                          lev=lev,
                                          time=time_range,
                                          lead=leads,  # Only consider first 4 leads.
                                          initmonths=initmonth,
                                          ens_avg=True)

    except:
        # UFS must be regridded onto Verif instead.
        regridder.regrid(var=ufs_vars,
                         lev=lev,
                         time=time_range,
                         ens_avg=True)

        # lev has already been selected in regrid. same with ens_avg.
        print(regridder.regridded.dataset())
        ufs_ds = regridder.regridded.retrieve(var=ufs_vars,
                                              time=time_range,
                                              lead=leads,  # Only consider first 4 leads.
                                              initmonths=initmonth)

        # Convert temporal coordinates to init_lead
        regridder.align()

        # lev has already been selected in resample.
        verif_ds = regridder.aligned.retrieve(var=verif_vars,
                                              time=time_range,
                                              lead=leads,
                                              initmonths=initmonth)

    if ufs_scaling_factor is not None:
        print('Scaling UFS data')
        ufs_ds = ufs_ds * ufs_scaling_factor

    if verif_scaling_factor is not None:
        print('Scaling VERIF data')
        verif_ds = verif_ds * verif_scaling_factor

    # Subset Data based on these years
    # UFS
    ufs_elnino_mask = (ufs_ds.init.dt.year.isin(elnino_years))
    ufs_lanina_mask = (ufs_ds.init.dt.year.isin(lanina_years))

    ufs_elnino_ds = ufs_ds.where(ufs_elnino_mask, drop=True)
    ufs_lanina_ds = ufs_ds.where(ufs_lanina_mask, drop=True)

    # Verif
    verif_elnino_mask = (verif_ds.init.dt.year.isin(elnino_years))
    verif_lanina_mask = (verif_ds.init.dt.year.isin(lanina_years))

    verif_elnino_ds = verif_ds.where(verif_elnino_mask, drop=True)
    verif_lanina_ds = verif_ds.where(verif_lanina_mask, drop=True)

    # Confirm that we have perfectly matching forecast times.
    n_verif_elnino = len(verif_elnino_ds.init.values) * len(verif_elnino_ds.lead.values)
    n_ufs_elnino = len(ufs_elnino_ds.init.values) * len(ufs_elnino_ds.lead.values)

    n_verif_lanina = len(verif_lanina_ds.init.values) * len(verif_lanina_ds.lead.values)
    n_ufs_lanina = len(ufs_lanina_ds.init.values) * len(ufs_lanina_ds.lead.values)

    # Check
    if n_verif_elnino != n_ufs_elnino or n_verif_lanina != n_ufs_lanina:

        msg = "Something went wrong... VERIF data and UFS data don't have identical time periods."
        raise ValueError(msg)

    if 'restoring effect' in statistics or 'stationary wave number' in statistics:

        print('Calculating restoring effect (Beta star) and stationary wave number (Ks)')

        # We must first check that U_WIND has been specified by the user.
        U_WIND_FOUND = False
        for wind_set in verif_data_reader.WINDS:
            if verif_var == wind_set['U_WIND']:
                U_WIND_FOUND = True

        if U_WIND_FOUND is False:
            msg = f'restoring effect and/or stationary wave number require U wind component, got {verif_var}'
            raise ValueError(msg)

        # Calculate UFS Beta* and Ks
        ufs_elnino_ds = stats.calc_betastar_kwavenumber(ufs_elnino_ds, uvar=ufs_var)
        ufs_lanina_ds = stats.calc_betastar_kwavenumber(ufs_lanina_ds, uvar=ufs_var)

        # VERIF VERIF Beta* and Ks
        verif_elnino_ds = stats.calc_betastar_kwavenumber(verif_elnino_ds, uvar=verif_var)
        verif_lanina_ds = stats.calc_betastar_kwavenumber(verif_lanina_ds, uvar=verif_var)

    if 'anomaly' in statistics:
        print("Calculating climatology statistics and anomalies.")

        # Compute climatology statistics
        ufs_stats = stats.calc_climatology_anomaly(ufs_ds[[ufs_var]], area_mean=False)
        verif_stats = stats.calc_climatology_anomaly(verif_ds[[verif_var]], area_mean=False)

        # Calculate UFS Anomaly
        ufs_elnino_ds = stats.calc_anomaly(ds=ufs_elnino_ds, var=ufs_var, stats=ufs_stats)
        ufs_lanina_ds = stats.calc_anomaly(ds=ufs_lanina_ds, var=ufs_var, stats=ufs_stats)

        # Calculate VERIF Anomaly
        verif_elnino_ds = stats.calc_anomaly(ds=verif_elnino_ds, var=verif_var, stats=verif_stats)
        verif_lanina_ds = stats.calc_anomaly(ds=verif_lanina_ds, var=verif_var, stats=verif_stats)

    if 'rossby wave source' in statistics:

        # RWS Components across entire data record.
        print('Calculating Rossby Wave Source (RWS) components.')
        ufs_ds = rws.calc_rws_components(ufs_ds, ufs_vars[0], ufs_vars[1])
        verif_ds = rws.calc_rws_components(verif_ds, verif_vars[0], verif_vars[1])

        # -----------
        # STATISTICS
        # -----------
        print('Calculating RWS component climatology statistics and anomalies.')
        # Climatologies
        ufs_absvrt_stats = stats.calc_climatology_anomaly(ufs_ds[['absvrt']], area_mean=False)
        ufs_uchi_stats = stats.calc_climatology_anomaly(ufs_ds[['uchi']], area_mean=False)
        ufs_vchi_stats = stats.calc_climatology_anomaly(ufs_ds[['vchi']], area_mean=False)

        verif_absvrt_stats = stats.calc_climatology_anomaly(verif_ds[['absvrt']], area_mean=False)
        verif_uchi_stats = stats.calc_climatology_anomaly(verif_ds[['uchi']], area_mean=False)
        verif_vchi_stats = stats.calc_climatology_anomaly(verif_ds[['vchi']], area_mean=False)

        # Anomalies
        ufs_absvrt_anomaly = stats.calc_anomaly(ds=ufs_ds, var='absvrt', stats=ufs_absvrt_stats)
        ufs_uchi_anomaly = stats.calc_anomaly(ds=ufs_ds, var='uchi', stats=ufs_uchi_stats)
        ufs_vchi_anomaly = stats.calc_anomaly(ds=ufs_ds, var='vchi', stats=ufs_vchi_stats)

        verif_absvrt_anomaly = stats.calc_anomaly(ds=verif_ds, var='absvrt', stats=verif_absvrt_stats)
        verif_uchi_anomaly = stats.calc_anomaly(ds=verif_ds, var='uchi', stats=verif_uchi_stats)
        verif_vchi_anomaly = stats.calc_anomaly(ds=verif_ds, var='vchi', stats=verif_vchi_stats)

        # ---------------
        # END STATISTICS
        # ---------------

        # Compute RWS
        ufs_elnino_ds = rws.calc_rws(ufs_elnino_ds,
                                     absvrt_stats=ufs_absvrt_stats,  # Absolute Vorticity
                                     absvrt_anomaly=ufs_absvrt_anomaly,
                                     uchi_stats=ufs_uchi_stats,  # UCHI
                                     uchi_anomaly=ufs_uchi_anomaly,
                                     vchi_stats=ufs_vchi_stats,  # VCHI
                                     vchi_anomaly=ufs_vchi_anomaly)

        ufs_lanina_ds = rws.calc_rws(ufs_lanina_ds,
                                     absvrt_stats=ufs_absvrt_stats,  # Absolute Vorticity
                                     absvrt_anomaly=ufs_absvrt_anomaly,
                                     uchi_stats=ufs_uchi_stats,  # UCHI
                                     uchi_anomaly=ufs_uchi_anomaly,
                                     vchi_stats=ufs_vchi_stats,  # VCHI
                                     vchi_anomaly=ufs_vchi_anomaly)

        verif_elnino_ds = rws.calc_rws(verif_elnino_ds,
                                       absvrt_stats=verif_absvrt_stats,  # Absolute Vorticity
                                       absvrt_anomaly=verif_absvrt_anomaly,
                                       uchi_stats=verif_uchi_stats,  # UCHI
                                       uchi_anomaly=verif_uchi_anomaly,
                                       vchi_stats=verif_vchi_stats,  # VCHI
                                       vchi_anomaly=verif_vchi_anomaly)

        verif_lanina_ds = rws.calc_rws(verif_lanina_ds,
                                       absvrt_stats=verif_absvrt_stats,  # Absolute Vorticity
                                       absvrt_anomaly=verif_absvrt_anomaly,
                                       uchi_stats=verif_uchi_stats,  # UCHI
                                       uchi_anomaly=verif_uchi_anomaly,
                                       vchi_stats=verif_vchi_stats,  # VCHI
                                       vchi_anomaly=verif_vchi_anomaly)

    print("ONI Datasets Ready.")

    return ufs_elnino_ds.load(), \
        ufs_lanina_ds.load(), \
        verif_elnino_ds.load(), \
        verif_lanina_ds.load()


def plot_composite(da: xr.DataArray,
                   title: str = '',
                   vmin: float = None,
                   vmax: float = None,
                   cmap: str = 'BuPu',
                   cmap_label: str = None,
                   topleft_label: str = None,
                   bottomright_label: str = None,
                   region: dict = None,
                   subtitle: str = ''):
    '''
    Generate shaded contour plot for composite statistics.
    '''
    cmap_center = False
    if vmin is not None and vmax is not None:
        if vmin == -1 * vmax:
            cmap_center = True

    center = 180
    projection = ccrs.PlateCarree(central_longitude=center)

    crs = ccrs.PlateCarree()

    # Instantiate plot
    plt.figure(figsize=(14, 7), dpi=200)
    ax = plt.axes(projection=projection)
    # ax.set_global()

    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.3,
                      linestyle='--')  # dashes=(5, 1))

    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.top_labels = False
    gl.right_labels = False

    # Remove degree symbol from gridline labels
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')

    cbar_kwargs = {
        'orientation': 'horizontal',
        'shrink': 0.7,
        'pad': 0.05
    }

    # Preserve the name of the cmap input. The following logic may coerce cmap to different variable type.
    cmap_string = cmap
    if cmap in CUSTOM_CMAPS:
        # Adjust n_levels
        n_levels = len(CUSTOM_CMAPS[cmap]) + 1
        # Load up the color map.  This variable is now a matplotlib object.
        # cmap = mcolors.LinearSegmentedColormap.from_list('', CUSTOM_CMAPS[cmap])
        cmap = ListedColormap(CUSTOM_CMAPS[cmap])
    else:
        n_levels = 20

    plot_args = {
        'ax': ax,
        'transform': crs,
        'cmap': cmap,  # cmap could be a string or a ListedColormap, at this point.
        'levels': n_levels,
        'extend': 'neither'  # Disable colorbar pointed extensions
    }

    plot_args['cbar_kwargs'] = cbar_kwargs

    if vmin is not None and vmax is not None:
        plot_args.update({'vmin': vmin, 'vmax': vmax})

    # We will add a label for the min, max, and average values across this field.
    min_value = da.min().values.item()
    avg_value = da.mean().values.item()
    max_value = da.max().values.item()
    std_value = da.std().values.item()

    # Cap values at the color bar range
    # (there is a matplotlib bug where values that deviate greatly from colorbar range show up as white)
    if vmin is not None:
        da = da.clip(min=vmin)

    if vmax is not None:
        da = da.clip(max=vmax)

    # Make plot
    p = da.plot.contourf(**plot_args)

    # Draw contour lines with hardcoded expectations for certain custom cmaps.
    if cmap_string == 'beta_star':
        da_for_lines = (da >= 0).astype(int)
        lines = da_for_lines.plot.contour(ax=ax, transform=crs, colors='black', linewidths=0.5, levels=1)

    if cmap_string in ['Ks', 'Ks_diff']:
        da_for_lines = da.notnull().astype(int)
        lines = da_for_lines.plot.contour(ax=ax, transform=crs, colors='black', linewidths=0.5, levels=1)

    # Center the colormap about 0
    if vmin is not None and vmax is not None:

        ticks = np.linspace(vmin, vmax, n_levels)
        cbar = p.colorbar

        tick_locations = []
        tick_labels = []

        tick_locations.append(vmin)
        tick_labels.append(f'{vmin:.1f}')

        for i in range(n_levels):

            # Skip ends.
            if i == 0 or i == (n_levels - 1):
                continue

            # Display 0 at center.
            if cmap_center is True and i == (n_levels / 2) - 1:
                tick_locations.append(0)
                tick_labels.append('0')
                continue

            # Don't display a value directly adjacent to 0.
            if cmap_center is True and i == (n_levels / 2):
                continue

            if i < (n_levels / 2):
                if i % 2 == 1:
                    continue
                tick_locations.append(ticks[i])
                tick_labels.append(f'{ticks[i]:.1f}')

            elif i > ((n_levels - 1) / 2):
                if i % 2 != 1:
                    continue
                tick_locations.append(ticks[i])
                tick_labels.append(f'{ticks[i]:.1f}')

        if cmap_string == 'beta_star':
            tick_labels[0] = ''

        tick_locations.append(vmax)
        tick_labels.append(f'{vmax:.1f}')

        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels(tick_labels)

        if cmap_label is not None:
            cbar.set_label(cmap_label, size=12)  # , weight='bold')

    ax.coastlines()

    # Draw square if a region is specified (e.g. nino 3.4)
    if region is not None:
        rect = mpatches.Rectangle((region['lonmin'], region['latmin']),
                                  width=(region['lonmax'] - region['lonmin']),
                                  height=(region['latmax'] - region['latmin']),
                                  color='black', fill=None, linewidth=0.5, alpha=0.75, zorder=1000,
                                  transform=ccrs.PlateCarree())

        ax.add_patch(rect)  # Add patch

    plt.title(f'{title}')

    # Add label to bottom right
    # values_label = f'min:    {min_value:.3f}\nmean: {avg_value:.3f}\nmax:    {max_value:.3f}'
    lower_left_values_label = f'max:\nmin:'
    lower_left_values_text = f'{max_value:.3f}\n{min_value:.3f}'

    if topleft_label is not None:
        ax.text(0.000001, 0.99999, topleft_label, ha='left', va='bottom', fontweight='bold', transform=ax.transAxes)

    if bottomright_label is not None:
        ax.text(0.99, 0.01, bottomright_label, ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)

    top_right_values_label = f'mean:\nstdev:'
    top_right_values_text = f'{avg_value:.3f}\n{std_value:.3f}'

    ax.text(0.01, 0.01, lower_left_values_label, ha='left', va='bottom', fontweight='bold', transform=ax.transAxes)
    ax.text(0.15, 0.01, lower_left_values_text, ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)

    ax.text(0.85999, 0.99999, top_right_values_label, ha='left', va='bottom', fontweight='bold', transform=ax.transAxes)
    ax.text(1.000001, .99999, top_right_values_text, ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)

    # Place title and subtitle
    if subtitle.strip() == '':
        plt.title(f'{title}')
    else:
        plt.title(f'{title}\n')
        ax.text(0.5, 1, subtitle, ha='center', va='bottom', fontweight='bold', transform=ax.transAxes)

    return plt
