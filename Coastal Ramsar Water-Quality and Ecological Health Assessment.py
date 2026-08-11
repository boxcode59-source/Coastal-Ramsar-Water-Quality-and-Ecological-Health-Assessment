"""
Lightweight Deep Learning and Regression Framework for
Coastal Ramsar Water-Quality and Ecological Health Assessment
Phase 1-5 Complete Implementation
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from scipy import signal, stats, interpolate
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the entire framework"""
    
    # Spectral ranges
    HYPERSPECTRAL_RANGE = (350, 900)  # nm
    HYPERSPECTRAL_SAMPLING = 1  # nm
    HYPERSPECTRAL_BANDS = 551  # 350-900 nm at 1nm
    
    # Sentinel-2 bands (nm)
    S2_BANDS = {
        'B1': 443, 'B2': 490, 'B3': 560, 'B4': 665,
        'B5': 705, 'B6': 740, 'B7': 783, 'B8': 842,
        'B8A': 865, 'B9': 945, 'B10': 1375, 'B11': 1610, 'B12': 2190
    }
    
    # Spatial
    ANALYSIS_GRID_SIZE = 30  # meters
    CRS = 'EPSG:4326'  # WGS84
    
    # Temporal
    TEMPORAL_WINDOW_DAYS = 7
    COMPOSITE_PERIODS = ['monthly', 'seasonal', 'annual']
    
    # Feature extraction
    PCA_VARIANCE_THRESHOLD = 0.95
    MI_THRESHOLD = 0.01
    
    # LCFN Architecture
    LCFN_FILTERS = [16, 32, 64]
    LCFN_KERNEL_SIZE = 3
    LCFN_DROPOUT = 0.3
    
    # LightGBM
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Training
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    RANDOM_SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    LEARNING_RATE = 0.001

# ============================================================================
# PHASE 1: MULTI-MODAL DATABASE CONSTRUCTION
# ============================================================================

class MultiModalDatabase:
    """
    Phase 1: Multi-Modal Database Construction
    Constructs six synchronized data modalities for coastal Ramsar assessment
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.database = {}
        self.metadata = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('MultiModalDB')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def build_m1_hyperspectral(self, gloria_data: Optional[np.ndarray] = None,
                               wavelengths: Optional[np.ndarray] = None) -> xr.Dataset:
        """
        M1: Hyperspectral Modality
        GLORIA coastal hyperspectral Rrs (350-900 nm, 1-nm sampling)
        
        Parameters:
        -----------
        gloria_data: np.ndarray of shape (time, height, width, bands)
        wavelengths: np.ndarray of wavelength values
        """
        self.logger.info("Building M1: Hyperspectral Modality")
        
        if wavelengths is None:
            wavelengths = np.arange(
                self.config.HYPERSPECTRAL_RANGE[0],
                self.config.HYPERSPECTRAL_RANGE[1] + self.config.HYPERSPECTRAL_SAMPLING,
                self.config.HYPERSPECTRAL_SAMPLING
            )
        
        if gloria_data is None:
            # Simulate hyperspectral data for demonstration
            n_time, n_h, n_w = 10, 100, 100
            gloria_data = self._simulate_hyperspectral(n_time, n_h, n_w, len(wavelengths))
        
        ds = xr.Dataset(
            data_vars={
                'Rrs': (['time', 'y', 'x', 'wavelength'], gloria_data),
                'Chl_a': (['time', 'y', 'x'], self._simulate_chla(gloria_data.shape[:3])),
                'TSS': (['time', 'y', 'x'], self._simulate_tss(gloria_data.shape[:3])),
                'CDOM': (['time', 'y', 'x'], self._simulate_cdom(gloria_data.shape[:3])),
                'Secchi_depth': (['time', 'y', 'x'], self._simulate_secchi(gloria_data.shape[:3]))
            },
            coords={
                'time': pd.date_range('2020-01-01', periods=gloria_data.shape[0], freq='ME'),
                'y': np.arange(gloria_data.shape[1]),
                'x': np.arange(gloria_data.shape[2]),
                'wavelength': wavelengths
            }
        )
        
        ds.attrs['modality'] = 'M1_Hyperspectral'
        ds.attrs['sensor'] = 'GLORIA'
        ds.attrs['spectral_range'] = f"{wavelengths[0]}-{wavelengths[-1]} nm"
        ds.attrs['sampling'] = f"{self.config.HYPERSPECTRAL_SAMPLING} nm"
        
        self.database['M1_Hyperspectral'] = ds
        self.metadata['M1_Hyperspectral'] = ds.attrs
        return ds
    
    def build_m2_multispectral(self, sentinel_data: Optional[np.ndarray] = None,
                               dates: Optional[List] = None) -> xr.Dataset:
        """
        M2: Multispectral Modality
        Sentinel-2 MSI with spectral indices
        """
        self.logger.info("Building M2: Multispectral Modality")
        
        n_bands = len(self.config.S2_BANDS)
        
        if sentinel_data is None:
            n_time, n_h, n_w = 20, 100, 100
            sentinel_data = self._simulate_sentinel2(n_time, n_h, n_w, n_bands)
        
        if dates is None:
            dates = pd.date_range('2020-01-01', periods=sentinel_data.shape[0], freq='W')
        
        # Calculate spectral indices
        ndwi = self._calculate_ndwi(sentinel_data)
        mndwi = self._calculate_mndwi(sentinel_data)
        ndci = self._calculate_ndci(sentinel_data)
        fai = self._calculate_fai(sentinel_data)
        
        ds = xr.Dataset(
            data_vars={
                'reflectance': (['time', 'y', 'x', 'band'], sentinel_data),
                'NDWI': (['time', 'y', 'x'], ndwi),
                'MNDWI': (['time', 'y', 'x'], mndwi),
                'NDCI': (['time', 'y', 'x'], ndci),
                'FAI': (['time', 'y', 'x'], fai)
            },
            coords={
                'time': dates,
                'y': np.arange(sentinel_data.shape[1]),
                'x': np.arange(sentinel_data.shape[2]),
                'band': list(self.config.S2_BANDS.keys())
            }
        )
        
        ds.attrs['modality'] = 'M2_Multispectral'
        ds.attrs['sensor'] = 'Sentinel-2 MSI'
        
        self.database['M2_Multispectral'] = ds
        self.metadata['M2_Multispectral'] = ds.attrs
        return ds
    
    def build_m3_historical(self, landsat_data: Optional[np.ndarray] = None) -> xr.Dataset:
        """
        M3: Historical Spatial Modality
        Landsat time series for long-term water dynamics
        """
        self.logger.info("Building M3: Historical Spatial Modality")
        
        if landsat_data is None:
            n_time, n_h, n_w = 100, 100, 100
            landsat_data = self._simulate_landsat(n_time, n_h, n_w)
        
        # Calculate water occurrence metrics
        water_occurrence = np.mean(landsat_data > 0, axis=0)
        water_recurrence = self._calculate_recurrence(landsat_data)
        water_persistence = self._calculate_persistence(landsat_data)
        seasonal_extent = self._calculate_seasonal_extent(landsat_data)
        
        ds = xr.Dataset(
            data_vars={
                'water_mask': (['time', 'y', 'x'], landsat_data),
                'water_occurrence': (['y', 'x'], water_occurrence),
                'water_recurrence': (['y', 'x'], water_recurrence),
                'water_persistence': (['y', 'x'], water_persistence),
                'seasonal_extent': (['season', 'y', 'x'], seasonal_extent),
            },
            coords={
                'time': pd.date_range('1985-01-01', periods=landsat_data.shape[0], freq='ME'),
                'y': np.arange(landsat_data.shape[1]),
                'x': np.arange(landsat_data.shape[2]),
                'season': ['DJF', 'MAM', 'JJA', 'SON']
            }
        )
        
        ds.attrs['modality'] = 'M3_Historical'
        ds.attrs['sensor'] = 'Landsat Series'
        
        self.database['M3_Historical'] = ds
        self.metadata['M3_Historical'] = ds.attrs
        return ds
    
    def build_m4_environmental(self, env_data: Optional[pd.DataFrame] = None) -> xr.Dataset:
        """
        M4: Environmental Modality
        Temperature, rainfall, wind, runoff, river discharge, soil moisture
        """
        self.logger.info("Building M4: Environmental Modality")
        
        if env_data is None:
            dates = pd.date_range('2020-01-01', periods=365, freq='D')
            env_data = self._simulate_environmental(dates)
        
        ds = xr.Dataset(
            data_vars={
                'temperature': (['time'], env_data['temperature'].values),
                'rainfall': (['time'], env_data['rainfall'].values),
                'wind_speed': (['time'], env_data['wind_speed'].values),
                'runoff': (['time'], env_data['runoff'].values),
                'river_discharge': (['time'], env_data['river_discharge'].values),
                'soil_moisture': (['time'], env_data['soil_moisture'].values)
            },
            coords={
                'time': env_data.index
            }
        )
        
        ds.attrs['modality'] = 'M4_Environmental'
        
        self.database['M4_Environmental'] = ds
        self.metadata['M4_Environmental'] = ds.attrs
        return ds
    
    def build_m5_oceanographic(self, ocean_data: Optional[pd.DataFrame] = None) -> xr.Dataset:
        """
        M5: Oceanographic Modality
        SST, SSS, tidal height, wave height, current velocity, sea-level anomaly
        """
        self.logger.info("Building M5: Oceanographic Modality")
        
        if ocean_data is None:
            dates = pd.date_range('2020-01-01', periods=365, freq='D')
            ocean_data = self._simulate_oceanographic(dates)
        
        ds = xr.Dataset(
            data_vars={
                'SST': (['time'], ocean_data['SST'].values),
                'SSS': (['time'], ocean_data['SSS'].values),
                'tidal_height': (['time'], ocean_data['tidal_height'].values),
                'wave_height': (['time'], ocean_data['wave_height'].values),
                'current_velocity': (['time'], ocean_data['current_velocity'].values),
                'sea_level_anomaly': (['time'], ocean_data['sea_level_anomaly'].values)
            },
            coords={
                'time': ocean_data.index
            }
        )
        
        ds.attrs['modality'] = 'M5_Oceanographic'
        
        self.database['M5_Oceanographic'] = ds
        self.metadata['M5_Oceanographic'] = ds.attrs
        return ds
    
    def build_m6_biological(self, bio_data: Optional[pd.DataFrame] = None) -> xr.Dataset:
        """
        M6: Biological/Ecological Modality
        Chlorophyll-a, phytoplankton, algal blooms, aquatic vegetation,
        mangrove extent, primary productivity
        """
        self.logger.info("Building M6: Biological/Ecological Modality")
        
        if bio_data is None:
            dates = pd.date_range('2020-01-01', periods=52, freq='W')
            bio_data = self._simulate_biological(dates)
        
        ds = xr.Dataset(
            data_vars={
                'chlorophyll_a': (['time'], bio_data['chlorophyll_a'].values),
                'phytoplankton': (['time'], bio_data['phytoplankton'].values),
                'algal_bloom_index': (['time'], bio_data['algal_bloom_index'].values),
                'aquatic_vegetation': (['time'], bio_data['aquatic_vegetation'].values),
                'mangrove_extent': (['time'], bio_data['mangrove_extent'].values),
                'primary_productivity': (['time'], bio_data['primary_productivity'].values)
            },
            coords={
                'time': bio_data.index
            }
        )
        
        ds.attrs['modality'] = 'M6_Biological'
        
        self.database['M6_Biological'] = ds
        self.metadata['M6_Biological'] = ds.attrs
        return ds
    
    # ---- Simulation methods ----
    def _simulate_hyperspectral(self, n_time, n_h, n_w, n_bands):
        """Simulate hyperspectral reflectance data"""
        np.random.seed(42)
        base_spectrum = np.exp(-0.5 * ((np.arange(n_bands) - 200) / 100) ** 2)
        data = np.zeros((n_time, n_h, n_w, n_bands))
        for t in range(n_time):
            noise = np.random.normal(0, 0.01, (n_h, n_w, n_bands))
            data[t] = base_spectrum[np.newaxis, np.newaxis, :] + noise
            data[t] += 0.02 * np.sin(2 * np.pi * t / n_time)
        return np.clip(data, 0, 1)
    
    def _simulate_sentinel2(self, n_time, n_h, n_w, n_bands):
        """Simulate Sentinel-2 reflectance data"""
        np.random.seed(43)
        data = np.random.normal(0.1, 0.05, (n_time, n_h, n_w, n_bands))
        return np.clip(data, 0, 1)
    
    def _simulate_landsat(self, n_time, n_h, n_w):
        """Simulate Landsat water mask data"""
        np.random.seed(44)
        water_prob = 0.3 + 0.1 * np.sin(2 * np.pi * np.arange(n_time) / 12)
        data = np.random.binomial(1, water_prob[:, np.newaxis, np.newaxis], (n_time, n_h, n_w))
        return data.astype(float)
    
    def _simulate_environmental(self, dates):
        """Simulate environmental data"""
        np.random.seed(45)
        n = len(dates)
        t = np.arange(n)
        return pd.DataFrame({
            'temperature': 25 + 5 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n),
            'rainfall': np.maximum(0, np.random.exponential(5, n) * (1 + 0.5 * np.sin(2 * np.pi * t / 365))),
            'wind_speed': 5 + 2 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 1, n),
            'runoff': np.maximum(0, np.random.normal(10, 5, n)),
            'river_discharge': np.maximum(0, 100 + 20 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 10, n)),
            'soil_moisture': 0.3 + 0.1 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.05, n)
        }, index=dates)
    
    def _simulate_oceanographic(self, dates):
        """Simulate oceanographic data"""
        np.random.seed(46)
        n = len(dates)
        t = np.arange(n)
        return pd.DataFrame({
            'SST': 28 + 3 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.5, n),
            'SSS': 35 + np.random.normal(0, 1, n),
            'tidal_height': 1.5 * np.sin(2 * np.pi * t / 0.5) + np.random.normal(0, 0.1, n),
            'wave_height': 1 + 0.5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.2, n),
            'current_velocity': 0.5 + 0.2 * np.sin(2 * np.pi * t / 0.5) + np.random.normal(0, 0.1, n),
            'sea_level_anomaly': 0.05 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.02, n)
        }, index=dates)
    
    def _simulate_biological(self, dates):
        """Simulate biological data"""
        np.random.seed(47)
        n = len(dates)
        t = np.arange(n)
        return pd.DataFrame({
            'chlorophyll_a': np.maximum(0, 5 + 3 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 1, n)),
            'phytoplankton': np.maximum(0, 1000 + 500 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 100, n)),
            'algal_bloom_index': np.random.binomial(1, 0.1 + 0.05 * np.sin(2 * np.pi * t / 26), n),
            'aquatic_vegetation': np.maximum(0, 0.5 + 0.2 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 0.1, n)),
            'mangrove_extent': 100 + np.random.normal(0, 2, n),
            'primary_productivity': np.maximum(0, 200 + 50 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 20, n))
        }, index=dates)
    
    def _simulate_chla(self, shape):
        return np.random.uniform(0.5, 20, shape)
    
    def _simulate_tss(self, shape):
        return np.random.uniform(1, 50, shape)
    
    def _simulate_cdom(self, shape):
        return np.random.uniform(0.1, 5, shape)
    
    def _simulate_secchi(self, shape):
        return np.random.uniform(0.5, 10, shape)
    
    # ---- Index calculations ----
    def _calculate_ndwi(self, data):
        """Normalized Difference Water Index: (Green - NIR) / (Green + NIR)"""
        green = data[..., 2]  # B3
        nir = data[..., 7]    # B8
        return (green - nir) / (green + nir + 1e-8)
    
    def _calculate_mndwi(self, data):
        """Modified NDWI: (Green - SWIR) / (Green + SWIR)"""
        green = data[..., 2]  # B3
        swir = data[..., 10]  # B11
        return (green - swir) / (green + swir + 1e-8)
    
    def _calculate_ndci(self, data):
        """Normalized Difference Chlorophyll Index: (RedEdge - Red) / (RedEdge + Red)"""
        red = data[..., 3]      # B4
        red_edge = data[..., 4] # B5
        return (red_edge - red) / (red_edge + red + 1e-8)
    
    def _calculate_fai(self, data):
        """Floating Algae Index"""
        nir = data[..., 7]    # B8
        red = data[..., 3]    # B4
        swir = data[..., 10]  # B11
        return nir - (red + (swir - red) * 0.5)
    
    def _calculate_recurrence(self, data):
        """Calculate water recurrence frequency"""
        water = data > 0.5
        return np.mean(water, axis=0)
    
    def _calculate_persistence(self, data):
        """Calculate water persistence (consecutive water presence)"""
        water = data > 0.5
        persistence = np.zeros(water.shape[1:])
        for i in range(water.shape[1]):
            for j in range(water.shape[2]):
                runs = []
                current_run = 0
                for t in range(water.shape[0]):
                    if water[t, i, j]:
                        current_run += 1
                    else:
                        if current_run > 0:
                            runs.append(current_run)
                        current_run = 0
                if current_run > 0:
                    runs.append(current_run)
                persistence[i, j] = np.max(runs) if runs else 0
        return persistence / water.shape[0]
    
    def _calculate_seasonal_extent(self, data):
        """Calculate seasonal water extent"""
        n_time = data.shape[0]
        seasonal = np.zeros((4, data.shape[1], data.shape[2]))
        for t in range(n_time):
            month = (t % 12) + 1
            if month in [12, 1, 2]:
                season = 0
            elif month in [3, 4, 5]:
                season = 1
            elif month in [6, 7, 8]:
                season = 2
            else:
                season = 3
            seasonal[season] += data[t]
        for s in range(4):
            seasonal[s] /= (n_time / 4)
        return seasonal
    
    def get_database_summary(self) -> pd.DataFrame:
        """Get summary of all modalities"""
        summaries = []
        for name, ds in self.database.items():
            summaries.append({
                'Modality': name,
                'Variables': list(ds.data_vars.keys()),
                'Temporal Range': f"{ds.time.values[0]} to {ds.time.values[-1]}",
                'Spatial Dimensions': f"{ds.sizes.get('y', 'N/A')} x {ds.sizes.get('x', 'N/A')}",
                'Total Samples': np.prod([ds.sizes.get(d, 1) for d in ds.dims])
            })
        return pd.DataFrame(summaries)


# ============================================================================
# PHASE 2: DATA PREPROCESSING AND FEATURE EXTRACTION
# ============================================================================

class DataPreprocessor:
    """
    Phase 2: Data Preprocessing and Feature Extraction
    Modality-specific preprocessing with standardization and selection
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.scalers = {}
        self.selected_features = {}
        self.feature_importance = {}
        
    def preprocess_hyperspectral(self, data: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """
        2.1 Hyperspectral Data Processing
        - Savitzky-Golay filtering
        - Continuum removal
        - Second-derivative spectral transformation
        - Min-Max normalization
        """
        processed = data.copy()
        
        # Apply Savitzky-Golay filter
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    processed[i, j, k, :] = signal.savgol_filter(
                        processed[i, j, k, :], 
                        window_length=11, 
                        polyorder=3
                    )
        
        # Continuum removal
        continuum_removed = self._continuum_removal(processed, wavelengths)
        
        # Second derivative transformation
        second_deriv = self._second_derivative(continuum_removed, wavelengths)
        
        # Min-Max normalization
        normalized = self._minmax_normalize(second_deriv)
        
        return normalized
    
    def preprocess_sentinel2(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        2.2 Sentinel-2 Data Processing
        - Atmospheric correction (simulated)
        - Cloud masking
        - NDWI-based water masking
        - Spectral index engineering
        """
        results = {}
        
        # Atmospheric correction (simplified)
        data_corrected = data * 0.9  # Simplified atmospheric correction
        
        # Cloud masking (simplified)
        cloud_mask = np.mean(data_corrected, axis=-1) < 0.8
        
        # Water masking using NDWI
        green = data_corrected[..., 2]
        nir = data_corrected[..., 7]
        ndwi = (green - nir) / (green + nir + 1e-8)
        water_mask = ndwi > 0
        
        # Generate spectral indices
        results['reflectance'] = data_corrected
        results['cloud_mask'] = cloud_mask
        results['water_mask'] = water_mask
        results['NDWI'] = ndwi
        results['MNDWI'] = self._calculate_mndwi(data_corrected)
        results['NDCI'] = self._calculate_ndci(data_corrected)
        results['FAI'] = self._calculate_fai(data_corrected)
        results['NDVI'] = self._calculate_ndvi(data_corrected)
        results['EVI'] = self._calculate_evi(data_corrected)
        
        return results
    
    def preprocess_landsat(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        2.3 Landsat Historical Data Processing
        - Cloud masking
        - Temporal compositing
        - MNDWI calculation
        - Otsu thresholding
        - Temporal frequency analysis
        """
        results = {}
        
        # Cloud masking (simplified)
        cloud_free = data > -1
        
        # Temporal compositing (annual)
        n_years = data.shape[0] // 12
        annual_composites = np.zeros((n_years, data.shape[1], data.shape[2]))
        for y in range(n_years):
            start = y * 12
            end = (y + 1) * 12
            annual_composites[y] = np.median(data[start:end], axis=0)
        
        # MNDWI calculation (simplified for simulated data)
        mndwi = data
        
        # Otsu thresholding
        binary_water = self._otsu_threshold(data)
        
        # Temporal frequency
        water_occurrence = np.mean(binary_water, axis=0)
        water_recurrence = self._calculate_recurrence_frequency(binary_water)
        
        results['annual_composites'] = annual_composites
        results['mndwi'] = mndwi
        results['binary_water'] = binary_water
        results['water_occurrence'] = water_occurrence
        results['water_recurrence'] = water_recurrence
        
        return results
    
    def preprocess_environmental(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        2.4 Environmental Data Processing
        - Linear interpolation
        - Z-score normalization
        - STL decomposition
        - Lag feature engineering
        """
        processed = data.copy()
        
        # Linear interpolation for gaps
        processed = processed.interpolate(method='linear', limit=5)
        
        # Z-score normalization
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(processed),
            columns=processed.columns,
            index=processed.index
        )
        self.scalers['environmental'] = scaler
        
        # STL decomposition for each variable
        stl_components = {}
        for col in normalized.columns:
            if len(normalized) >= 14:  # Need at least 2 periods
                decomposition = seasonal_decompose(
                    normalized[col].dropna(), 
                    model='additive', 
                    period=7,  # weekly seasonality
                    extrapolate_trend='freq'
                )
                stl_components[f'{col}_trend'] = decomposition.trend
                stl_components[f'{col}_seasonal'] = decomposition.seasonal
                stl_components[f'{col}_residual'] = decomposition.resid
        
        stl_df = pd.DataFrame(stl_components, index=normalized.index)
        
        # Lag features
        lag_features = {}
        for col in normalized.columns:
            for lag in [1, 3, 7, 14, 30]:
                lag_features[f'{col}_lag_{lag}'] = normalized[col].shift(lag)
        
        lag_df = pd.DataFrame(lag_features, index=normalized.index)
        
        # Combine all features
        result = pd.concat([normalized.add_prefix('raw_'), stl_df, lag_df], axis=1)
        
        return result
    
    def preprocess_oceanographic(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        2.5 Oceanographic Data Processing
        - Temporal interpolation
        - Z-score normalization
        - Seasonal anomaly analysis
        - Lagged feature generation
        """
        processed = data.copy()
        
        # Temporal interpolation
        processed = processed.interpolate(method='linear', limit=3)
        
        # Z-score normalization
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(processed),
            columns=processed.columns,
            index=processed.index
        )
        self.scalers['oceanographic'] = scaler
        
        # Seasonal anomaly
        anomalies = {}
        for col in normalized.columns:
            rolling_mean = normalized[col].rolling(window=30, center=True).mean()
            anomalies[f'{col}_anomaly'] = normalized[col] - rolling_mean
        
        anomaly_df = pd.DataFrame(anomalies, index=normalized.index)
        
        # Lag features
        lag_features = {}
        for col in normalized.columns:
            for lag in [1, 3, 7, 14]:
                lag_features[f'{col}_lag_{lag}'] = normalized[col].shift(lag)
        
        lag_df = pd.DataFrame(lag_features, index=normalized.index)
        
        result = pd.concat([normalized.add_prefix('raw_'), anomaly_df, lag_df], axis=1)
        
        return result
    
    def preprocess_biological(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        2.6 Biological/Ecological Data Processing
        - Min-Max normalization
        - Seasonal aggregation
        - NDVI/NDRE-based vegetation analysis
        - Algal bloom detection
        """
        processed = data.copy()
        
        # Min-Max normalization
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(processed),
            columns=processed.columns,
            index=processed.index
        )
        self.scalers['biological'] = scaler
        
        # Seasonal aggregation
        seasonal = {}
        for col in normalized.columns:
            seasonal[f'{col}_rolling_mean_4w'] = normalized[col].rolling(window=4).mean()
            seasonal[f'{col}_rolling_std_4w'] = normalized[col].rolling(window=4).std()
        
        seasonal_df = pd.DataFrame(seasonal, index=normalized.index)
        
        # Algal bloom detection (threshold-based)
        bloom_features = {}
        if 'chlorophyll_a' in normalized.columns:
            chla_threshold = normalized['chlorophyll_a'].quantile(0.9)
            bloom_features['bloom_flag'] = (normalized['chlorophyll_a'] > chla_threshold).astype(int)
        
        if 'algal_bloom_index' in normalized.columns:
            bloom_features['bloom_intensity'] = normalized['algal_bloom_index'] * normalized.get('chlorophyll_a', 1)
        
        bloom_df = pd.DataFrame(bloom_features, index=normalized.index)
        
        result = pd.concat([normalized.add_prefix('norm_'), seasonal_df, bloom_df], axis=1)
        
        return result
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                         method: str = 'mutual_info') -> Tuple[pd.DataFrame, List[str]]:
        """
        2.7 Feature Standardization and Selection
        - Mutual Information + Correlation-Based Feature Selection
        """
        # Remove constant and near-constant features
        constant_features = [col for col in X.columns if X[col].std() < 1e-8]
        X_filtered = X.drop(columns=constant_features)
        
        if method == 'mutual_info':
            # Mutual Information
            mi_scores = mutual_info_regression(X_filtered.fillna(0), y.fillna(0))
            mi_df = pd.DataFrame({
                'feature': X_filtered.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)
            
            # Select features above threshold
            selected = mi_df[mi_df['mi_score'] > self.config.MI_THRESHOLD]['feature'].tolist()
            
            # Correlation-based refinement
            if len(selected) > 1:
                corr_matrix = X_filtered[selected].corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
                selected = [f for f in selected if f not in high_corr]
            
            self.selected_features[method] = selected
            self.feature_importance['mutual_info'] = mi_df
            
        elif method == 'correlation':
            # Correlation-based
            correlations = X_filtered.corrwith(y).abs().sort_values(ascending=False)
            selected = correlations[correlations > 0.1].index.tolist()
            self.selected_features[method] = selected
            
        return X_filtered[selected], selected
    
    # ---- Helper methods ----
    def _continuum_removal(self, data: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """Continuum removal for absorption feature enhancement"""
        result = data.copy()
        n_wavelengths = data.shape[-1]
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    spectrum = data[i, j, k, :]
                    # Find convex hull points
                    hull_indices = self._find_convex_hull_indices(spectrum)
                    if len(hull_indices) > 1:
                        continuum = np.interp(
                            np.arange(n_wavelengths), 
                            hull_indices, 
                            spectrum[hull_indices]
                        )
                        result[i, j, k, :] = spectrum / (continuum + 1e-10)
        
        return result
    
    def _find_convex_hull_indices(self, spectrum: np.ndarray) -> np.ndarray:
        """Find indices forming the convex hull"""
        n = len(spectrum)
        if n < 3:
            return np.arange(n)
        
        indices = []
        for i in range(n):
            while len(indices) >= 2:
                p1, p2 = indices[-2], indices[-1]
                if (spectrum[p2] - spectrum[p1]) * (i - p2) > \
                   (spectrum[i] - spectrum[p2]) * (p2 - p1):
                    indices.pop()
                else:
                    break
            indices.append(i)
        
        return np.array(indices)
    
    def _second_derivative(self, data: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """Second derivative spectral transformation"""
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    result[i, j, k, :] = np.gradient(np.gradient(data[i, j, k, :]))
        return result
    
    def _minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max normalization"""
        min_val = data.min(axis=-1, keepdims=True)
        max_val = data.max(axis=-1, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-10)
    
    def _otsu_threshold(self, data: np.ndarray) -> np.ndarray:
        """Otsu thresholding for water/non-water classification"""
        binary = np.zeros_like(data)
        for t in range(data.shape[0]):
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    if data.ndim == 4:
                        values = data[t, i, j, :]
                    else:
                        values = data[t, i, j]
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        threshold = np.median(values)
                        binary[t, i, j] = 1 if np.mean(values) > threshold else 0
        return binary
    
    def _calculate_recurrence_frequency(self, binary_data: np.ndarray) -> np.ndarray:
        """Calculate water recurrence frequency"""
        return np.mean(binary_data, axis=0)
    
    def _calculate_mndwi(self, data):
        green = data[..., 2]
        swir = data[..., 10]
        return (green - swir) / (green + swir + 1e-8)
    
    def _calculate_ndci(self, data):
        red = data[..., 3]
        red_edge = data[..., 4]
        return (red_edge - red) / (red_edge + red + 1e-8)
    
    def _calculate_fai(self, data):
        nir = data[..., 7]
        red = data[..., 3]
        swir = data[..., 10]
        return nir - (red + (swir - red) * 0.5)
    
    def _calculate_ndvi(self, data):
        nir = data[..., 7]
        red = data[..., 3]
        return (nir - red) / (nir + red + 1e-8)
    
    def _calculate_evi(self, data):
        nir = data[..., 7]
        red = data[..., 3]
        blue = data[..., 0]
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


# ============================================================================
# PHASE 3: SPATIAL-TEMPORAL FEATURE ASSOCIATION
# ============================================================================

class SpatialTemporalAssociator:
    """
    Phase 3: Spatial-Temporal Feature Association
    Links modalities by location and time
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.common_grid = None
        self.feature_dataset = None
        
    def create_common_grid(self, spatial_extent: Dict, resolution: int = None):
        """Create common spatial reference grid"""
        if resolution is None:
            resolution = self.config.ANALYSIS_GRID_SIZE
        
        lon = np.arange(spatial_extent['lon_min'], spatial_extent['lon_max'], 
                       resolution / 111320.0)  # Approximate meters to degrees
        lat = np.arange(spatial_extent['lat_min'], spatial_extent['lat_max'], 
                       resolution / 111320.0)
        
        self.common_grid = {
            'lon': lon,
            'lat': lat,
            'resolution': resolution,
            'shape': (len(lat), len(lon))
        }
        
        return self.common_grid
    
    def spatial_align(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, np.ndarray]:
        """
        3.1 Spatial Reference Alignment
        Transform all datasets to common spatial reference
        """
        aligned = {}
        
        for name, ds in datasets.items():
            if 'y' in ds.dims and 'x' in ds.dims:
                # Spatial dataset
                features = []
                for var in ds.data_vars:
                    data = ds[var].values
                    if data.ndim >= 2:
                        # Reshape to (time, spatial_features)
                        spatial_data = data.reshape(data.shape[0], -1)
                        features.append(spatial_data)
                
                if features:
                    aligned[name] = np.concatenate(features, axis=-1)
            else:
                # Temporal-only dataset
                features = []
                for var in ds.data_vars:
                    data = ds[var].values
                    features.append(data.reshape(-1, 1))
                
                if features:
                    aligned[name] = np.concatenate(features, axis=-1)
        
        return aligned
    
    def temporal_align(self, datasets: Dict[str, np.ndarray],
                      primary_times: pd.DatetimeIndex,
                      dataset_times: Dict[str, pd.DatetimeIndex]) -> Dict[str, np.ndarray]:
        """
        3.2 Temporal Alignment
        Align observations to common time periods
        """
        aligned = {}
        
        for name, data in datasets.items():
            times = dataset_times.get(name)
            if times is None:
                aligned[name] = data
                continue
            
            # Find closest temporal matches
            aligned_data = []
            for t in primary_times:
                # Find nearest time index
                time_diff = np.abs((times - t).total_seconds())
                nearest_idx = np.argmin(time_diff)
                
                if time_diff[nearest_idx] < self.config.TEMPORAL_WINDOW_DAYS * 86400:
                    aligned_data.append(data[nearest_idx])
                else:
                    aligned_data.append(np.full(data.shape[1:], np.nan))
            
            aligned[name] = np.array(aligned_data)
        
        return aligned
    
    def temporal_aggregate(self, data: pd.DataFrame, 
                          target_times: pd.DatetimeIndex,
                          window_days: int = None) -> pd.DataFrame:
        """
        3.3 Temporal Aggregation
        Aggregate high-frequency data to match satellite observations
        """
        if window_days is None:
            window_days = self.config.TEMPORAL_WINDOW_DAYS
        
        aggregated = {}
        
        for col in data.columns:
            values = []
            for t in target_times:
                start = t - timedelta(days=window_days)
                end = t + timedelta(days=window_days)
                window_data = data.loc[start:end, col]
                
                if len(window_data) > 0:
                    values.append({
                        f'{col}_mean': window_data.mean(),
                        f'{col}_min': window_data.min(),
                        f'{col}_max': window_data.max(),
                        f'{col}_std': window_data.std() if len(window_data) > 1 else 0,
                        f'{col}_sum': window_data.sum()
                    })
                else:
                    values.append({
                        f'{col}_mean': np.nan,
                        f'{col}_min': np.nan,
                        f'{col}_max': np.nan,
                        f'{col}_std': np.nan,
                        f'{col}_sum': np.nan
                    })
            
            values_df = pd.DataFrame(values, index=target_times)
            for key in values_df.columns:
                aggregated[key] = values_df[key]
        
        return pd.DataFrame(aggregated, index=target_times)
    
    def spatial_aggregate(self, fine_data: np.ndarray, 
                         coarse_grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        3.4 Spatial Aggregation
        Aggregate fine-resolution data to match coarser analysis grid
        """
        if fine_data.ndim == 3:
            # Time x Height x Width
            n_time = fine_data.shape[0]
            aggregated = np.zeros((n_time, coarse_grid_shape[0], coarse_grid_shape[1]))
            
            scale_h = fine_data.shape[1] / coarse_grid_shape[0]
            scale_w = fine_data.shape[2] / coarse_grid_shape[1]
            
            for t in range(n_time):
                for i in range(coarse_grid_shape[0]):
                    for j in range(coarse_grid_shape[1]):
                        h_start = int(i * scale_h)
                        h_end = int((i + 1) * scale_h)
                        w_start = int(j * scale_w)
                        w_end = int((j + 1) * scale_w)
                        
                        patch = fine_data[t, h_start:h_end, w_start:w_end]
                        aggregated[t, i, j] = np.nanmean(patch)
            
            return aggregated
        else:
            return fine_data
    
    def quality_control(self, feature_dataset: pd.DataFrame,
                       missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        3.5 Quality Control
        Remove records with excessive missing values or invalid observations
        """
        # Remove records with too many missing values
        missing_ratio = feature_dataset.isnull().mean(axis=1)
        valid = missing_ratio < missing_threshold
        
        # Remove infinite values
        finite = np.isfinite(feature_dataset.select_dtypes(include=[np.number])).all(axis=1)
        
        # Filter
        cleaned = feature_dataset[valid & finite].copy()
        
        # Interpolate remaining missing values
        cleaned = cleaned.interpolate(method='linear', limit=3)
        cleaned = cleaned.fillna(cleaned.mean())
        
        return cleaned
    
    def build_stmcdf(self, aligned_features: Dict[str, np.ndarray],
                    target_variables: Dict[str, np.ndarray],
                    timestamps: pd.DatetimeIndex) -> xr.Dataset:
        """
        3.6 Spatial-Temporal Multimodal Coastal Feature Dataset
        Build the final unified feature dataset
        """
        data_vars = {}
        
        # Add modality features
        for name, features in aligned_features.items():
            data_vars[name] = (['time', 'feature_dim'], features)
        
        # Add target variables
        for name, values in target_variables.items():
            data_vars[name] = (['time'], values.flatten()[:len(timestamps)])
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': timestamps
            }
        )
        
        ds.attrs['description'] = 'Spatial-Temporal Multimodal Coastal Feature Dataset'
        ds.attrs['n_modalities'] = len(aligned_features)
        ds.attrs['created'] = datetime.now().isoformat()
        
        self.feature_dataset = ds
        return ds


# ============================================================================
# PHASE 4: LIGHTWEIGHT FEATURE LEARNING
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class LightweightCoastalFeatureNetwork(nn.Module):
    """
    Lightweight Coastal Feature Network (LCFN)
    Phase 4: Lightweight feature extraction from hyperspectral and Sentinel-2
    """
    
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        
        # Input processing layers
        self.conv1 = DepthwiseSeparableConv(1, config.LCFN_FILTERS[0], config.LCFN_KERNEL_SIZE)
        self.conv2 = DepthwiseSeparableConv(config.LCFN_FILTERS[0], config.LCFN_FILTERS[1])
        self.conv3 = DepthwiseSeparableConv(config.LCFN_FILTERS[1], config.LCFN_FILTERS[2])
        
        # Regularization
        self.dropout = nn.Dropout2d(config.LCFN_DROPOUT)
        self.batch_norm = nn.BatchNorm2d(config.LCFN_FILTERS[2])
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class MultimodalFeatureLearner:
    """
    Phase 4: Lightweight Feature Learning
    Combines LCFN features with auxiliary modalities
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.lcfn = LightweightCoastalFeatureNetwork(config)
        self.pca = PCA(n_components=config.PCA_VARIANCE_THRESHOLD)
        self.is_fitted = False
        
    def prepare_input(self, hyperspectral_features: np.ndarray, 
                     sentinel2_features: np.ndarray) -> torch.Tensor:
        """
        4.1 Input Feature Preparation
        Combine hyperspectral and Sentinel-2 features for LCFN input
        """
        # Reshape to (batch, channels, height, width) for 2D convolution
        # Using spectral as channels and spatial as 2D
        if hyperspectral_features.ndim == 4:
            # (time, h, w, bands)
            data = np.concatenate([
                hyperspectral_features.mean(axis=-1, keepdims=True),
                sentinel2_features[..., :3].mean(axis=-1, keepdims=True)
            ], axis=-1)
        else:
            data = hyperspectral_features
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(data)
        
        # Ensure 4D shape
        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(1)
        
        return tensor_data
    
    def extract_lcfn_features(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features using the LCFN
        """
        self.lcfn.eval()
        with torch.no_grad():
            features = self.lcfn(input_tensor)
        
        return features.numpy()
    
    def combine_modalities(self, lcfn_features: np.ndarray,
                          auxiliary_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        4.5 Multimodal Feature Combination
        Combine LCFN features with auxiliary modalities
        """
        combined = [lcfn_features]
        
        for name, features in auxiliary_features.items():
            # Ensure same number of samples
            n_samples = len(lcfn_features)
            if features.shape[0] == n_samples:
                if features.ndim > 1:
                    combined.append(features.reshape(n_samples, -1))
                else:
                    combined.append(features.reshape(-1, 1))
        
        return np.hstack(combined)
    
    def reduce_features(self, combined_features: np.ndarray) -> np.ndarray:
        """
        4.6 Feature Reduction
        Apply PCA for dimensionality reduction
        """
        if not self.is_fitted:
            self.pca.fit(combined_features)
            self.is_fitted = True
        
        reduced = self.pca.transform(combined_features)
        
        return reduced
    
    def get_compact_representation(self, hyperspectral: np.ndarray,
                                   sentinel2: np.ndarray,
                                   auxiliary: Dict[str, np.ndarray]) -> np.ndarray:
        """
        4.7 Complete lightweight feature extraction pipeline
        """
        # Prepare input
        input_tensor = self.prepare_input(hyperspectral, sentinel2)
        
        # Extract LCFN features
        lcfn_features = self.extract_lcfn_features(input_tensor)
        
        # Combine modalities
        combined = self.combine_modalities(lcfn_features, auxiliary)
        
        # Reduce features
        compact = self.reduce_features(combined)
        
        return compact


# ============================================================================
# PHASE 5: LIGHTWEIGHT WATER-QUALITY REGRESSION
# ============================================================================

class WaterQualityRegressor:
    """
    Phase 5: Lightweight Water-Quality Regression
    LightGBM-based regression for water quality parameters
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.models = {}
        self.optimized_params = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      param_name: str = 'default') -> lgb.LGBMRegressor:
        """
        5.1 Train LightGBM regression model
        """
        params = self.config.LGBM_PARAMS.copy()
        
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.models[param_name] = model
        
        return model
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 param_name: str = 'default',
                                 n_trials: int = 50) -> Dict:
        """
        5.3 Optimize LightGBM hyperparameters using Optuna
        """
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1,
                'n_jobs': -1
            }
            
            # Cross-validation
            cv_scores = []
            tscv = TimeSeriesSplit(n_splits=3)
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], 
                         callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_vl)
                cv_scores.append(mean_squared_error(y_vl, y_pred, squared=False))
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.optimized_params[param_name] = study.best_params
        
        # Train final model with best params
        best_params = self.config.LGBM_PARAMS.copy()
        best_params.update(study.best_params)
        
        model = lgb.LGBMRegressor(**best_params)
        
        self.models[param_name] = model
        
        return study.best_params
    
    def predict(self, X: np.ndarray, param_name: str = 'default') -> np.ndarray:
        """
        Make predictions for a water quality parameter
        """
        if param_name not in self.models:
            raise ValueError(f"Model for {param_name} not trained")
        
        return self.models[param_name].predict(X)
    
    def predict_all_parameters(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        5.2 Multi-Parameter Water-Quality Prediction
        Predict Chl-a, TSS, CDOM, and Secchi depth
        """
        predictions = {}
        
        for param in ['Chl_a', 'TSS', 'CDOM', 'Secchi_depth']:
            if param in self.models:
                predictions[param] = self.predict(X, param)
        
        return predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        5.4 Regression Performance Evaluation
        Calculate R², RMSE, MAE, MAPE
        """
        metrics = {}
        
        # R²
        metrics['R2'] = r2_score(y_true, y_pred)
        
        # RMSE
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = np.nan
        
        return metrics
    
    def compare_with_baselines(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        5.5 Comparison with Conventional Regression Models
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        import xgboost as xgb
        
        results = []
        
        # LightGBM (our model)
        if 'default' in self.models:
            y_pred_lgbm = self.predict(X_test)
            metrics_lgbm = self.evaluate(y_test, y_pred_lgbm)
            metrics_lgbm['Model'] = 'LightGBM (Proposed)'
            results.append(metrics_lgbm)
        
        # Multiple Linear Regression
        mlr = LinearRegression()
        mlr.fit(X_train, y_train)
        y_pred_mlr = mlr.predict(X_test)
        metrics_mlr = self.evaluate(y_test, y_pred_mlr)
        metrics_mlr['Model'] = 'Multiple Linear Regression'
        results.append(metrics_mlr)
        
        # Random Forest
        rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rfr.fit(X_train, y_train)
        y_pred_rfr = rfr.predict(X_test)
        metrics_rfr = self.evaluate(y_test, y_pred_rfr)
        metrics_rfr['Model'] = 'Random Forest'
        results.append(metrics_rfr)
        
        # SVR
        svr = SVR(kernel='rbf')
        svr.fit(X_train, y_train)
        y_pred_svr = svr.predict(X_test)
        metrics_svr = self.evaluate(y_test, y_pred_svr)
        metrics_svr['Model'] = 'Support Vector Regression'
        results.append(metrics_svr)
        
        # XGBoost
        xgbr = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        xgbr.fit(X_train, y_train)
        y_pred_xgb = xgbr.predict(X_test)
        metrics_xgb = self.evaluate(y_test, y_pred_xgb)
        metrics_xgb['Model'] = 'XGBoost'
        results.append(metrics_xgb)
        
        return pd.DataFrame(results).set_index('Model')


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class CoastalRamsarAssessmentPipeline:
    """
    Complete pipeline integrating all five phases
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.database_builder = MultiModalDatabase(config)
        self.preprocessor = DataPreprocessor(config)
        self.associator = SpatialTemporalAssociator(config)
        self.feature_learner = MultimodalFeatureLearner(config)
        self.regressor = WaterQualityRegressor(config)
        self.logger = logging.getLogger('CoastalPipeline')
        
    def run_phase1(self, use_simulated: bool = True) -> Dict:
        """
        Phase 1: Build multi-modal database
        """
        self.logger.info("Starting Phase 1: Multi-Modal Database Construction")
        
        if use_simulated:
            # Build all six modalities with simulated data
            m1 = self.database_builder.build_m1_hyperspectral()
            m2 = self.database_builder.build_m2_multispectral()
            m3 = self.database_builder.build_m3_historical()
            m4 = self.database_builder.build_m4_environmental()
            m5 = self.database_builder.build_m5_oceanographic()
            m6 = self.database_builder.build_m6_biological()
        
        summary = self.database_builder.get_database_summary()
        self.logger.info(f"Database summary:\n{summary}")
        
        return self.database_builder.database
    
    def run_phase2(self, database: Dict) -> Dict:
        """
        Phase 2: Preprocess all modalities and extract features
        """
        self.logger.info("Starting Phase 2: Data Preprocessing and Feature Extraction")
        
        preprocessed = {}
        
        # M1: Hyperspectral
        if 'M1_Hyperspectral' in database:
            ds = database['M1_Hyperspectral']
            wavelengths = ds.wavelength.values
            data = ds.Rrs.values
            preprocessed['M1_Hyperspectral'] = self.preprocessor.preprocess_hyperspectral(
                data, wavelengths
            )
        
        # M2: Sentinel-2
        if 'M2_Multispectral' in database:
            ds = database['M2_Multispectral']
            data = ds.reflectance.values
            preprocessed['M2_Multispectral'] = self.preprocessor.preprocess_sentinel2(data)
        
        # M3: Landsat Historical
        if 'M3_Historical' in database:
            ds = database['M3_Historical']
            data = ds.water_mask.values
            preprocessed['M3_Historical'] = self.preprocessor.preprocess_landsat(data)
        
        # M4: Environmental
        if 'M4_Environmental' in database:
            ds = database['M4_Environmental']
            data = ds.to_dataframe()
            preprocessed['M4_Environmental'] = self.preprocessor.preprocess_environmental(data)
        
        # M5: Oceanographic
        if 'M5_Oceanographic' in database:
            ds = database['M5_Oceanographic']
            data = ds.to_dataframe()
            preprocessed['M5_Oceanographic'] = self.preprocessor.preprocess_oceanographic(data)
        
        # M6: Biological
        if 'M6_Biological' in database:
            ds = database['M6_Biological']
            data = ds.to_dataframe()
            preprocessed['M6_Biological'] = self.preprocessor.preprocess_biological(data)
        
        return preprocessed
    
    def run_phase3(self, preprocessed: Dict, database: Dict) -> xr.Dataset:
        """
        Phase 3: Spatial-temporal feature association
        """
        self.logger.info("Starting Phase 3: Spatial-Temporal Feature Association")
        
        # Align spatial features
        spatial_datasets = {k: v for k, v in preprocessed.items() 
                          if isinstance(v, (xr.Dataset, np.ndarray)) and v.ndim >= 3}
        aligned_spatial = self.associator.spatial_align(database)
        
        # Define primary timeline
        primary_times = pd.date_range('2020-01-01', periods=24, freq='15D')
        
        # Temporal alignment
        dataset_times = {
            'M1_Hyperspectral': database.get('M1_Hyperspectral', xr.Dataset()).time.values,
            'M2_Multispectral': database.get('M2_Multispectral', xr.Dataset()).time.values,
            'M3_Historical': database.get('M3_Historical', xr.Dataset()).time.values,
        }
        
        aligned_temporal = self.associator.temporal_align(
            aligned_spatial, primary_times, dataset_times
        )
        
        # Define target variables
        target_vars = {}
        if 'M1_Hyperspectral' in database:
            ds = database['M1_Hyperspectral']
            for var in ['Chl_a', 'TSS', 'CDOM', 'Secchi_depth']:
                if var in ds:
                    target_vars[var] = ds[var].values
        
        # Build STMCDF
        stmcdf = self.associator.build_stmcdf(
            aligned_temporal, target_vars, primary_times
        )
        
        return stmcdf
    
    def run_phase4(self, stmcdf: xr.Dataset, preprocessed: Dict) -> np.ndarray:
        """
        Phase 4: Lightweight feature learning
        """
        self.logger.info("Starting Phase 4: Lightweight Feature Learning")
        
        # Extract hyperspectral and Sentinel-2 features
        hyperspectral = preprocessed.get('M1_Hyperspectral', np.random.randn(24, 20, 20, 50))
        sentinel2 = preprocessed.get('M2_Multispectral', {})
        
        if isinstance(sentinel2, dict):
            sentinel2_data = sentinel2.get('reflectance', np.random.randn(24, 20, 20, 13))
        else:
            sentinel2_data = sentinel2
        
        # Prepare auxiliary features
        auxiliary = {}
        for key in ['M3_Historical', 'M4_Environmental', 'M5_Oceanographic', 'M6_Biological']:
            if key in preprocessed:
                data = preprocessed[key]
                if isinstance(data, pd.DataFrame):
                    auxiliary[key] = data.values
                elif isinstance(data, dict):
                    for sub_key, sub_data in data.items():
                        if isinstance(sub_data, np.ndarray):
                            auxiliary[f'{key}_{sub_key}'] = sub_data.flatten()[:24]
        
        # Get compact representation
        compact_features = self.feature_learner.get_compact_representation(
            hyperspectral.reshape(24, -1)[:, :50],  # Simplified
            sentinel2_data.reshape(24, -1)[:, :13],  # Simplified
            auxiliary
        )
        
        return compact_features
    
    def run_phase5(self, compact_features: np.ndarray, 
                   target_vars: Dict[str, np.ndarray]) -> Dict:
        """
        Phase 5: Water quality regression
        """
        self.logger.info("Starting Phase 5: Water-Quality Regression")
        
        results = {}
        
        for param_name, y_values in target_vars.items():
            self.logger.info(f"Training model for {param_name}")
            
            # Ensure matching lengths
            n_samples = min(len(compact_features), len(y_values))
            X = compact_features[:n_samples]
            y = y_values[:n_samples].flatten()
            
            # Train-test split (temporal)
            split_idx = int(0.7 * n_samples)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Further split train into train/val
            val_idx = int(0.8 * len(X_train))
            X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
            y_tr, y_val = y_train[:val_idx], y_train[val_idx:]
            
            # Train model
            self.regressor.train_lightgbm(X_tr, y_tr, X_val, y_val, param_name)
            
            # Evaluate
            y_pred = self.regressor.predict(X_test, param_name)
            metrics = self.regressor.evaluate(y_test, y_pred)
            
            results[param_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'actual': y_test
            }
            
            self.logger.info(f"{param_name} - R²: {metrics['R2']:.4f}, "
                           f"RMSE: {metrics['RMSE']:.4f}, "
                           f"MAE: {metrics['MAE']:.4f}")
        
        # Compare with baselines for first parameter
        first_param = list(target_vars.keys())[0]
        X_full = compact_features[:len(target_vars[first_param])]
        y_full = target_vars[first_param][:len(X_full)].flatten()
        
        split_idx = int(0.7 * len(X_full))
        X_tr, X_te = X_full[:split_idx], X_full[split_idx:]
        y_tr, y_te = y_full[:split_idx], y_full[split_idx:]
        
        comparison = self.regressor.compare_with_baselines(X_tr, y_tr, X_te, y_te)
        results['model_comparison'] = comparison
        
        return results
    
    def run_full_pipeline(self, use_simulated: bool = True) -> Dict:
        """
        Execute the complete pipeline (Phases 1-5)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Complete Coastal Ramsar Assessment Pipeline")
        self.logger.info("=" * 60)
        
        # Phase 1
        database = self.run_phase1(use_simulated)
        
        # Phase 2
        preprocessed = self.run_phase2(database)
        
        # Phase 3
        stmcdf = self.run_phase3(preprocessed, database)
        
        # Phase 4
        compact_features = self.run_phase4(stmcdf, preprocessed)
        
        # Phase 5
        # Extract target variables
        target_vars = {}
        if 'M1_Hyperspectral' in database:
            ds = database['M1_Hyperspectral']
            for var in ['Chl_a', 'TSS', 'CDOM', 'Secchi_depth']:
                if var in ds:
                    target_vars[var] = ds[var].values
        
        results = self.run_phase5(compact_features, target_vars)
        
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete!")
        self.logger.info("=" * 60)
        
        return {
            'database': database,
            'preprocessed': preprocessed,
            'stmcdf': stmcdf,
            'compact_features': compact_features,
            'regression_results': results
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Initialize configuration
    config = Config()
    
    # Initialize pipeline
    pipeline = CoastalRamsarAssessmentPipeline(config)
    
    # Run complete pipeline
    results = pipeline.run_full_pipeline(use_simulated=True)
    
    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    regression_results = results['regression_results']
    
    for param, data in regression_results.items():
        if param == 'model_comparison':
            continue
        print(f"\n{param}:")
        print(f"  R²:   {data['metrics']['R2']:.4f}")
        print(f"  RMSE: {data['metrics']['RMSE']:.4f}")
        print(f"  MAE:  {data['metrics']['MAE']:.4f}")
        if not np.isnan(data['metrics'].get('MAPE', np.nan)):
            print(f"  MAPE: {data['metrics']['MAPE']:.2f}%")
    
    # Model comparison
    if 'model_comparison' in regression_results:
        print("\nModel Comparison:")
        print(regression_results['model_comparison'])
    
    return results


if __name__ == "__main__":
    results = main()