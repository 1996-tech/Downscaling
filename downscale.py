import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')

def r_squared(y_true, y_pred):
    """Custom R-squared metric for Keras"""
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def AutoEncoderModel(nfea, nout, nodes, acts, mdropout, reg, batchnorm, isres, outtype, fact="linear"):
    """
    Create autoencoder model with residual connections
    
    Parameters:
    - nfea: number of input features
    - nout: number of output features
    - nodes: list of node counts for each layer
    - acts: list of activation functions
    - mdropout: dropout rate
    - reg: regularization (not implemented in this version)
    - batchnorm: whether to use batch normalization
    - isres: whether to use residual connections
    - outtype: output type (0 for regression)
    - fact: final activation function
    """
    
    input_layer = layers.Input(shape=(nfea,))
    x = input_layer
    
    # Encoder layers
    for i, (node, act) in enumerate(zip(nodes, acts)):
        if isres and i > 0 and x.shape[-1] == node:
            # Residual connection
            residual = x
            x = layers.Dense(node, activation=act)(x)
            if batchnorm:
                x = layers.BatchNormalization()(x)
            if mdropout > 0:
                x = layers.Dropout(mdropout)(x)
            x = layers.Add()([x, residual])
        else:
            x = layers.Dense(node, activation=act)(x)
            if batchnorm:
                x = layers.BatchNormalization()(x)
            if mdropout > 0:
                x = layers.Dropout(mdropout)(x)
    
    # Output layer
    output_layer = layers.Dense(nout, activation=fact)(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def resample_raster(source_data, source_transform, target_transform, target_shape, method='nearest'):
    """Resample raster data to match target grid"""
    destination = np.zeros(target_shape, dtype=source_data.dtype)
    
    reproject(
        source_data,
        destination,
        src_transform=source_transform,
        dst_transform=target_transform,
        src_crs='EPSG:4326',
        dst_crs='EPSG:4326',
        resampling=getattr(Resampling, method)
    )
    
    return destination

def ResautoDownscale(r2_stack, fpredict0, c_grid, ss=0.2, nepoch=30, cores=1, thresh=0.01, ntime=5):
    """
    Iterative implementation of downscaling with autoencoder based residual network
    
    Parameters:
    - r2_stack: Stack of covariates for downscaling (numpy array with shape [height, width, n_bands])
    - fpredict0: Starting predictions (numpy array with shape [height, width])
    - c_grid: Coarsely resolved grid (numpy array with shape [coarse_height, coarse_width])
    - ss: sampling proportion for independent test (default: 0.2)
    - nepoch: number of epochs for residual network training (default: 30)
    - cores: number of CPU cores (not used in this implementation)
    - thresh: stopping criterion threshold (default: 0.01)
    - ntime: maximum number of iterations (default: 5)
    
    Returns:
    - Dictionary containing performance metrics and downscaled images
    """
    
    lastpredict = fpredict0.copy()
    diogRMSE = pd.DataFrame({'tindex': [0]})
    
    # Create coarse grid cell numbers
    coarse_height, coarse_width = c_grid.shape
    fine_height, fine_width = fpredict0.shape
    
    # Create cell number grid for coarse resolution
    c_gridCELL = np.arange(1, coarse_height * coarse_width + 1).reshape(coarse_height, coarse_width)
    
    # Resample coarse grids to fine grid
    c_grid_ds = np.zeros((fine_height, fine_width))
    c_gridCELL_ds = np.zeros((fine_height, fine_width))
    
    # Simple nearest neighbor resampling
    scale_y = fine_height / coarse_height
    scale_x = fine_width / coarse_width
    
    for i in range(fine_height):
        for j in range(fine_width):
            coarse_i = int(i / scale_y)
            coarse_j = int(j / scale_x)
            coarse_i = min(coarse_i, coarse_height - 1)
            coarse_j = min(coarse_j, coarse_width - 1)
            c_grid_ds[i, j] = c_grid[coarse_i, coarse_j]
            c_gridCELL_ds[i, j] = c_gridCELL[coarse_i, coarse_j]
    
    # Prepare coarse grid reference data
    c_dat = []
    for i in range(coarse_height):
        for j in range(coarse_width):
            if not np.isnan(c_grid[i, j]):
                cell_num = i * coarse_width + j + 1
                c_dat.append([cell_num, j, i, c_grid[i, j]])
    
    c_dat_ref = pd.DataFrame(c_dat, columns=['cell', 'X', 'Y', 'C_grid'])
    
    # Initialize callbacks
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.000001, patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=15, factor=0.5, min_lr=1e-7)
    
    map1 = lastpredict.copy()
    maxr2 = -np.inf
    minrmse = np.inf
    maxmap = map1.copy()
    
    for i in range(1, ntime + 2):
        print(f"Iteration {i}")
        
        # Calculate downscaled fit by aggregating fine grid values within coarse grid cells
        downFit = {}
        for cell_num in c_dat_ref['cell']:
            mask = c_gridCELL_ds == cell_num
            if np.any(mask):
                downFit[cell_num] = np.mean(map1[mask])
        
        # Merge with coarse grid data
        c_dat_ref['F_grid'] = c_dat_ref['cell'].map(downFit)
        xx = c_dat_ref.dropna().copy()
        
        # Calculate performance metrics
        if len(xx) > 0:
            corr = np.corrcoef(xx['C_grid'], xx['F_grid'])[0, 1]
            rsquare = r2_score(xx['C_grid'], xx['F_grid'])
            rmse_val = rmse(xx['C_grid'], xx['F_grid'])
            
            # Calculate confidence intervals
            residuals = xx['C_grid'] - xx['F_grid']
            var_calc = np.sum(residuals**2) / (len(xx) * (len(xx) - 1))
            se_var = np.sqrt(var_calc)
            t_crit = t.ppf(1 - 0.05/2, df=len(xx) - 1)
            cI = se_var * t_crit
            
            mean_squared_residual = np.mean(residuals**2)
            upper_ci = np.sqrt(mean_squared_residual + cI)
            lower_ci = np.sqrt(mean_squared_residual - cI)
            mid_val = np.sqrt(mean_squared_residual)
            
            # Store results
            row_data = {
                'tindex': i,
                'lowerCi': lower_ci,
                'upperCi': upper_ci,
                'mid': mid_val,
                'resnetccr2': rsquare,
                'resnetccrmse': rmse_val,
                'resnetcccor': corr
            }
            
            diogRMSE = pd.concat([diogRMSE, pd.DataFrame([row_data])], ignore_index=True)
            
            # Update best results
            if i == 1:
                maxr2 = rsquare
                minrmse = rmse_val
                maxmap = map1.copy()
            elif i > 1 and maxr2 < rsquare:
                maxr2 = rsquare
                maxmap = map1.copy()
                minrmse = rmse_val
        
        if i == ntime + 1:
            break
            
        # Early stopping based on R2 degradation
        if i > 3:
            if len(diogRMSE) >= 3:
                recent_r2 = diogRMSE['resnetccr2'].iloc[-3:]
                if len(recent_r2) >= 3:
                    if (recent_r2.iloc[0] - recent_r2.iloc[2]) > 0.02 and \
                       (recent_r2.iloc[0] - recent_r2.iloc[1]) > 0.02:
                        break
        
        # Calculate adjustment factor
        if len(xx) > 0:
            xx['AF'] = xx['C_grid'] / xx['F_grid']
            
            # Create adjustment factor grid
            AF_grid = np.ones_like(c_grid_ds)
            for _, row in xx.iterrows():
                x_coord, y_coord = int(row['X']), int(row['Y'])
                if 0 <= x_coord < coarse_width and 0 <= y_coord < coarse_height:
                    # Map coarse grid coordinates to fine grid
                    fine_x_start = int(x_coord * scale_x)
                    fine_y_start = int(y_coord * scale_y)
                    fine_x_end = min(int((x_coord + 1) * scale_x), fine_width)
                    fine_y_end = min(int((y_coord + 1) * scale_y), fine_height)
                    
                    AF_grid[fine_y_start:fine_y_end, fine_x_start:fine_x_end] = row['AF']
            
            # Apply adjustment
            upd_test = map1 * AF_grid
            
            # Prepare data for neural network training
            # Stack all data
            full_data = []
            for h in range(fine_height):
                for w in range(fine_width):
                    if not np.isnan(upd_test[h, w]) and not np.isnan(c_gridCELL_ds[h, w]):
                        row = [upd_test[h, w], c_gridCELL_ds[h, w]]
                        # Add covariate data
                        for band in range(r2_stack.shape[2]):
                            row.append(r2_stack[h, w, band])
                        full_data.append(row)
            
            if len(full_data) > 0:
                full_data = np.array(full_data)
                
                # Remove rows with NaN values
                valid_rows = ~np.isnan(full_data).any(axis=1)
                full_data = full_data[valid_rows]
                
                if len(full_data) > 0:
                    # Normalize features (columns 2 onwards)
                    scaler = StandardScaler()
                    features = full_data[:, 2:]
                    features_scaled = scaler.fit_transform(features)
                    full_data[:, 2:] = features_scaled
                    
                    # Split data
                    train_idx, test_idx = train_test_split(
                        np.arange(len(full_data)), 
                        test_size=ss, 
                        random_state=42
                    )
                    
                    x_train = full_data[train_idx, 2:]
                    y_train = full_data[train_idx, 0]
                    x_test = full_data[test_idx, 2:]
                    y_test = full_data[test_idx, 0]
                    
                    # Model parameters
                    nfea = x_train.shape[1]
                    nout = 1
                    nodes = [32, 16, 8, 4]
                    mdropout = 0.2
                    isres = True
                    outtype = 0
                    fact = "linear"
                    acts = ["relu"] * len(nodes)
                    batchnorm = True
                    reg = None
                    
                    # Create and compile model
                    autoresmodel = AutoEncoderModel(
                        nfea, nout, nodes, acts, mdropout, reg, batchnorm, isres, outtype, fact
                    )
                    
                    autoresmodel.compile(
                        loss='mean_squared_error',
                        optimizer='rmsprop',
                        metrics=['mean_squared_error', r_squared]
                    )
                    
                    # Train model
                    history = autoresmodel.fit(
                        x_train, y_train,
                        epochs=nepoch,
                        batch_size=min(2560, len(x_train)),
                        callbacks=[early_stopping, reduce_lr],
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    # Make predictions
                    pred_test = autoresmodel.predict(x_test, batch_size=min(2560, len(x_test)))
                    pred_full = autoresmodel.predict(full_data[:, 2:], batch_size=min(2560, len(full_data)))
                    
                    # Calculate test metrics
                    test_r2 = r2_score(y_test, pred_test)
                    test_rmse = rmse(y_test, pred_test)
                    
                    # Update diagnostics
                    final_epoch = len(history.history['r_squared']) - 1
                    diogRMSE.loc[diogRMSE['tindex'] == i, 'resnetr2'] = history.history['r_squared'][final_epoch]
                    diogRMSE.loc[diogRMSE['tindex'] == i, 'resnetrmse'] = history.history['mean_squared_error'][final_epoch]
                    diogRMSE.loc[diogRMSE['tindex'] == i, 'resnetvalr2'] = history.history['val_r_squared'][final_epoch]
                    diogRMSE.loc[diogRMSE['tindex'] == i, 'resnetvalrmse'] = history.history['val_mean_squared_error'][final_epoch]
                    diogRMSE.loc[diogRMSE['tindex'] == i, 'resnettestr2'] = test_r2
                    diogRMSE.loc[diogRMSE['tindex'] == i, 'resnettestrmse'] = test_rmse
                    
                    # Create new prediction map
                    map2 = np.zeros_like(fpredict0)
                    data_idx = 0
                    for h in range(fine_height):
                        for w in range(fine_width):
                            if not np.isnan(upd_test[h, w]) and not np.isnan(c_gridCELL_ds[h, w]):
                                if data_idx < len(pred_full):
                                    map2[h, w] = pred_full[data_idx, 0]
                                    data_idx += 1
                                else:
                                    map2[h, w] = upd_test[h, w]
                            else:
                                map2[h, w] = np.nan
                    
                    print(f"Modeling RMSE = {test_rmse:.3f}; r2 = {test_r2:.3f}")
                    
                    # Check convergence
                    if i >= 3 and len(diogRMSE) >= 3:
                        recent_mid = diogRMSE['mid'].iloc[-3:]
                        if len(recent_mid) >= 3:
                            FF = np.mean([
                                abs(recent_mid.iloc[0] - recent_mid.iloc[1]),
                                abs(recent_mid.iloc[1] - recent_mid.iloc[2]),
                                abs(recent_mid.iloc[0] - recent_mid.iloc[2])
                            ])
                            if FF <= thresh:
                                break
                    
                    map1 = map2.copy()
                    
                    # Cleanup
                    del autoresmodel, full_data, x_train, y_train, x_test, y_test
                    gc.collect()
    
    return {
        'diogRMSE': diogRMSE,
        'raster': maxmap,
        'r2': maxr2,
        'rmse': minrmse
    }

# Example usage:
# Assuming you have your data loaded as numpy arrays:
# r2_stack: shape (height, width, n_covariates)
# fpredict0: shape (height, width) - initial predictions
# c_grid: shape (coarse_height, coarse_width) - coarse resolution grid
#
# result = ResautoDownscale(r2_stack, fpredict0, c_grid, ss=0.2, nepoch=30, thresh=0.01, ntime=5)
# downscaled_raster = result['raster']
# performance_metrics = result['diogRMSE']
# final_r2 = result['r2']
# final_rmse = result['rmse']
