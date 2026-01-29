# Taken from: https://github.com/SchattenGenie/mlhep2019_2_phase/blob/master/analysis/calogan_metrics.py
import numpy as np
from numba import njit

@njit
def get_assymmetry(imgs, ps, points, orthog=False):
    n_imgs = imgs.shape[0]
    assym_res = np.zeros(n_imgs)
    zoff = 25.0
    
    # Сетка от -14.5 до 14.5 (30 шагов)
    grid_coords = np.linspace(-14.5, 14.5, 30)
    
    for k in range(n_imgs):
        p0 = points[k, 0] + zoff * ps[k, 0] / ps[k, 2]
        p1 = points[k, 1] + zoff * ps[k, 1] / ps[k, 2]
        
        # Определяем коэффициенты линии
        ps0 = ps[k, 0]
        ps1 = ps[k, 1]
        
        sign = 1.0
        if not orthog:
            sign = 1.0 if ps1 > 0 else -1.0
            
        sum_zz = 0.0
        sum_not_zz = 0.0
        total_img_sum = 0.0
        
        for i in range(30):
            y_val = grid_coords[i]
            for j in range(30):
                x_val = grid_coords[j]
                
                # Вычисляем line_func на лету
                if orthog:
                    val = (x_val - p0) / (ps0 / ps1) + p1
                else:
                    val = -(x_val - p0) / (ps1 / ps0) + p1
                
                # Условие маски zz
                is_zz = (y_val - val) * sign >= 0
                
                pixel_val = imgs[k, i, j]
                total_img_sum += pixel_val
                if is_zz:
                    sum_zz += pixel_val
                else:
                    sum_not_zz += pixel_val
                    
        assym_res[k] = (sum_zz - sum_not_zz) / (total_img_sum + 1e-10)
        
    return assym_res

@njit
def zz_to_line(zz):
    n, h, w = zz.shape
    res = np.zeros((n, h, w))
    for k in range(n):
        for i in range(h):
            for j in range(w):
                diff_h = 0.0
                diff_w = 0.0
                if j < w - 1:
                    diff_w = abs(zz[k, i, j+1] - zz[k, i, j])
                if i < h - 1:
                    diff_h = abs(zz[k, i+1, j] - zz[k, i, j])
                
                val = diff_h + diff_w
                if val > 1.0:
                    res[k, i, j] = 1.0
                else:
                    res[k, i, j] = val
    return res

@njit
def get_shower_width(data, ps, points, orthog=False):
    n_imgs = data.shape[0]
    sigmas = np.zeros(n_imgs)
    zoff = 25.0
    grid_coords = np.linspace(-14.5, 14.5, 30)
    
    for k in range(n_imgs):
        p0 = points[k, 0] + zoff * ps[k, 0] / ps[k, 2]
        p1 = points[k, 1] + zoff * ps[k, 1] / ps[k, 2]
        ps0, ps1 = ps[k, 0], ps[k, 1]
        
        rescale = np.sqrt(1 + (ps1 / ps0)**2)
        sign = 1.0
        if not orthog:
            sign = -1.0 if ps1 < 0 else 1.0

        # Сначала генерируем маску zz для текущего изображения
        zz_current = np.ones((30, 30))
        for i in range(30):
            y_val = grid_coords[i]
            for j in range(30):
                if orthog:
                    val = -(grid_coords[j] - p0) / (ps0 / ps1) + p1
                else:
                    val = (grid_coords[j] - p0) / (ps1 / ps0) + p1
                if (y_val - val) * sign < 0:
                    zz_current[i, j] = 0
        
        # Вычисляем line и моменты
        sum_0, sum_1, sum_2 = 0.0, 0.0, 0.0
        for i in range(30):
            for j in range(30):
                # Находим line[i, j] аналогично zz_to_line
                dw = abs(zz_current[i, j+1] - zz_current[i, j]) if j < 29 else 0.0
                dh = abs(zz_current[i+1, j] - zz_current[i, j]) if i < 29 else 0.0
                line_val = min(1.0, dw + dh)
                
                ww = line_val * data[k, i, j]
                scaled_x = rescale * grid_coords[j]
                
                sum_0 += ww
                sum_1 += ww * scaled_x
                sum_2 += ww * (scaled_x**2)
        
        s1 = sum_1 / (sum_0 + 1e-5)
        s2 = sum_2 / (sum_0 + 1e-5)
        var = s2 - s1 * s1
        sigmas[k] = np.sqrt(max(0, var)) # max(0, var) для стабильности
        
    return sigmas

@njit
def get_ms_ratio2(data, alpha=0.1):
    n_imgs = data.shape[0]
    res = np.zeros(n_imgs)
    # Сумма по осям (1, 2)
    for k in range(n_imgs):
        ms = 0.0
        for i in range(30):
            for j in range(30):
                ms += data[k, i, j]
        
        threshold = ms * alpha
        count = 0
        for i in range(30):
            for j in range(30):
                if data[k, i, j] >= threshold:
                    count += 1
        res[k] = count / 900.0
    return res

@njit
def get_sparsity_level(data):
    alphas = np.logspace(-5, -1, 20)
    sparsity = np.zeros((len(alphas), data.shape[0]))
    for idx in range(len(alphas)):
        sparsity[idx] = get_ms_ratio2_numba(data, alphas[idx])
    return sparsity


def get_physical_stats(EnergyDeposit, ParticleMomentum, ParticlePoint):
    assym = get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)
    assym_ortho = get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)
    sh_width = get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)
    sh_width_ortho = get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)
    sparsity_level = get_sparsity_level(EnergyDeposit)
    stats = np.c_[
        assym, 
        assym_ortho, 
        sh_width, 
        sh_width_ortho, 
        sparsity_level.T
    ]
    return stats
