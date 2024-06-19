import numpy as np
import pytplot
from datetime import datetime
import scipy.signal as signal
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
from pytplot import tplot, data_quants, store_data, options, split_vec, cdf_to_tplot, xlim, get_data
import sys
import os
import numpy as np
import math
import spiceypy as spice
import datetime
from datetime import datetime, timedelta
import pyspedas
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def convert_string_to_datetime(input_string):
    elements = input_string.split()
    date_string = f"{elements[0]}-{elements[1]}T{elements[2].zfill(2)}:{elements[3].zfill(2)}:{elements[4].zfill(2)}.{elements[5]}"
    return date_string

def convert_date(input_date):
    date_object = datetime.strptime(input_date, '%Y-%jT%H:%M:%S.%f')
    return f'{date_object.strftime("%Y-%m-%d")}T{date_object.strftime("%H:%M:%S.%f")}'

def convert_to_float_list(input_list):
    return [list(map(float, sublist)) for sublist in input_list]


def lt_iauJup( TARGET: str, utc):
    """
    Args:
        TARGET: target in the Jupiter system (like EUROPA and JUNO)
        utc: observation date at the target (UT)

    Returns:
        td: local time in datetime [hh:mm:ss]
    """
    et = spice.str2et(utc)

    # Eigen vector toward the sun's position seen from Jupiter in IAU_JUPITER coordinate.
    posSUN, lighttime = spice.spkpos(
        targ='SUN', et=et, ref='IAU_JUPITER', abcorr='LT+S', obs='JUPITER'
    )
    posSUN = posSUN/math.sqrt(posSUN[0]**2 + posSUN[1]**2 + posSUN[2]**2)

    # Eigen vector toward target's position seen from Jupiter in IAU_JUPITER coordinate.
    posTARG, _ = spice.spkpos(
        targ=TARGET, et=et, ref='IAU_JUPITER', abcorr='NONE', obs='JUPITER'
    )
    posTARG = posTARG / \
        math.sqrt(posTARG[0]**2 + posTARG[1]**2 + posTARG[2]**2)

    # Dusk terminator
    dusk_term = np.array([-posSUN[1], posSUN[0]])
    dusk_dot = dusk_term[0]*posTARG[0] + dusk_term[1]*posTARG[1]
    if dusk_dot >= 0:
        # Target is in the dusk side.
        d_phi = np.pi + \
            math.acos(posSUN[0]*posTARG[0]+posSUN[1]*posTARG[1])
        print('Dusk [deg]:', math.degrees(d_phi))
    else:
        # Target is in the dawn side.
        d_phi = np.pi - \
            math.acos(posSUN[0]*posTARG[0] + posSUN[1]*posTARG[1])
        print('Dawn [deg]:', math.degrees(d_phi))

    sec = (3600*24/360)*math.degrees(d_phi)    # [sec]
    td = timedelta(seconds=sec)
    print(td)

    return td

def save_data_png(sts_file_path, window, shift):
    times, orbit_positions, magetic_fields = [], [], []

    with open(sts_file_path, 'r') as sts_file:
        for line in sts_file:
            parts = line.split()
            if len(parts) >= 10:
                times.append(' '.join(parts[0:6]))
                orbit_positions.append(parts[11:14])
                magetic_fields.append(parts[7:10])

    times = [convert_date(convert_string_to_datetime(time)) for time in times[2:]]
    orbit_positions = np.array(convert_to_float_list(orbit_positions[2:])) / 71492
    magetic_fields = np.array(convert_to_float_list(magetic_fields[2:]))

    pytplot.store_data('juno_mag', data={'x': np.array(times, dtype=np.datetime64), 'y': magetic_fields})
    pytplot.store_data('juno_orbit', data={'x': np.array(times, dtype=np.datetime64), 'y': orbit_positions})
    pytplot.options('juno_orbit', opt_dict={'ytitle': 'PC coordinate', 'ysubtitle': '[$R_J$]', 'legend_names': ['X', 'Y', 'Z']})

    data_x, data_y, data_z = magetic_fields.T

    norm = np.sqrt(data_x**2 + data_y**2 + data_z**2)
    pytplot.store_data('juno_mag_norm', data={'x': np.array(times, dtype=np.datetime64), 'y': norm})
    pytplot.options('juno_mag_norm', opt_dict={'legend_names': ['B'], 'ylog': 0, 'ytitle':'Total Mag', 'ysubtitle':'[$nT$]'})

    mag = data_quants['juno_mag']
    # サンプリング周波数の計算
    fs = len(mag) / (24 * 60 * 60)  # データの長さと1日の秒数から計算

    # 200秒間の移動平均を計算するためのサンプル数
    window_size_seconds = 200  # 秒
    window_size_samples = int(window_size_seconds * fs)

    # 移動平均を計算
    mag_avg = mag.rolling(time=window_size_samples, center=True).mean()

    store_data('juno_mag_smooth', data={'x': mag['time'], 'y': mag_avg})

    pytplot.split_vec('juno_mag_smooth')

    data_x = pytplot.get_data('juno_mag_smooth_x')
    data_y = pytplot.get_data('juno_mag_smooth_y')
    data_z = pytplot.get_data('juno_mag_smooth_z')

    rotmat=np.zeros((3, 3, (len(data_x[0]))))
    rotmat_t=np.zeros((3, 3, (len(data_x[0]))))

    for i in range(len(data_x[0])-1):
        bvec = [data_x[1][i], data_y[1][i], data_z[1][i]]
        zz = [0, 0, 1]

        yhat = np.cross(zz, bvec)
        xhat = np.cross(yhat, bvec)
        zhat = bvec

        yhat = yhat / np.linalg.norm(yhat)
        xhat = xhat / np.linalg.norm(xhat)
        zhat = zhat / np.linalg.norm(zhat)

        rotmat[:,:,i] = np.array([xhat, yhat, zhat])
        rotmat_t[:,:,i] = np.array([xhat, yhat, zhat]).T

    rot_mag = np.zeros((3, len(data_x[0])))

    pytplot.split_vec('juno_mag')
    d_x = pytplot.get_data('juno_mag_x')
    d_y = pytplot.get_data('juno_mag_y')
    d_z = pytplot.get_data('juno_mag_z')

    for i in range(len(d_x[0])-1):
        rot_mag[:,i] = np.dot(rotmat[:,:,i], [d_x[1][i], d_y[1][i], d_z[1][i]])

    store_data('juno_mag_rot', data={'x': mag['time'], 'y': rot_mag.T})

    norm_smooth = np.sqrt(mag_avg[:,0]**2 + mag_avg[:,1]**2 + mag_avg[:,2]**2)
    pytplot.split_vec('juno_mag_rot')

    perp = np.sqrt(data_quants['juno_mag_rot_y']**2 + data_quants['juno_mag_rot_x']**2)

    store_data('juno_mag_rot_perp', data={'x': mag['time'], 'y': perp})

    para = norm - norm_smooth

    store_data('juno_mag_rot_para', data={'x': mag['time'], 'y': para})

    ratio = perp / para

    store_data('juno_mag_rot_ratio', data={'x': mag['time'], 'y': ratio})

    options('juno_mag_rot_ratio', opt_dict={'ylog': 0, 'ytitle': 'Ratio', 'legend_names': ['Perp/Para'], 'yrange': [-5000, 5000]})


    yaxis_max = 0.04

    # データの取り出し
    base_time = pd.to_datetime(data_quants['juno_mag_rot_para']['time'][0].values)
    b_para = data_quants['juno_mag_rot_para']
    b_perp = data_quants['juno_mag_rot_perp']
    # サンプリング周波数の設定 (例: 1Hz)
    length = b_para.shape
    fs = length[0] / 24 / 60 / 60

    # スペクトログラムの計算
    f, t, Sxx = signal.spectrogram(b_para, fs, nperseg=window, noverlap=shift)

    # 時間の変換
    resulting_times = [(base_time + timedelta(seconds=time)).strftime('%Y-%m-%dT%H:%M:%S.%f') for time in t]
    middle_times = np.array(resulting_times, dtype=np.datetime64)

    # データの保存
    pytplot.store_data(f'para_{window}_{shift}', data={'x': middle_times, 'y': Sxx.T, 'v': f})

    f, t, Sxx = signal.spectrogram(b_perp, fs, nperseg=window, noverlap=shift)
    pytplot.store_data(f'perp_{window}_{shift}', data={'x': middle_times, 'y': Sxx.T, 'v': f})

    f, t, Sxx = signal.spectrogram(norm, fs, nperseg=window, noverlap=shift)
    pytplot.store_data(f'norm_{window}_{shift}', data={'x': middle_times, 'y': Sxx.T, 'v': f})

    pytplot.store_data(f'ratio_{window}_{shift}', data={'x': middle_times, 'y': data_quants[f'perp_{window}_{shift}']/data_quants[f'para_{window}_{shift}'], 'v': f})

    yrange = [0.008, fs/2]
    options(f'para_{window}_{shift}', opt_dict={'zlog': True, 'ylog': 1, 'ysubtitle': '[Hz]', 'spec': True, 'ztitle': '[$nT^2/Hz$]', 'ytitle': 'MFA_para', 'Colormap': 'jet', 'yrange': yrange})
    options(f'perp_{window}_{shift}', opt_dict={'zlog': True, 'ylog': 1, 'ysubtitle': '[Hz]', 'spec': True, 'ztitle': '[$nT^2/Hz$]', 'ytitle': 'MFA_perp', 'Colormap': 'jet', 'yrange': yrange})
    options(f'norm_{window}_{shift}', opt_dict={'zlog': True, 'ylog': 1, 'ysubtitle': '[Hz]', 'spec': True, 'ztitle': '[$nT^2/Hz$]', 'ytitle': 'MFA_norm', 'Colormap': 'jet', 'yrange': yrange})
    options(f'ratio_{window}_{shift}', opt_dict={'zlog': False, 'ylog': 1, 'ysubtitle': '[Hz]', 'spec': True, 'ztitle': 'Ratio', 'ytitle': 'perp/para', 'Colormap': 'Greys', 'yrange': yrange, 'zrange': [0, 10]})
    #tplot([f'para_{window}_{shift}', f'perp_{window}_{shift}', f'norm_{window}_{shift}', f'ratio_{window}_{shift}'], xsize=12, ysize=12)

    sys.path.append('/home/kooki/Documents/Juno_ULC_EMIC')
    import Juno_pos as jp


    RJ = 71492000
    pos_arr = data_quants['juno_orbit'] * RJ


    # 時間をソート
    sorted_pos_arr = pos_arr.sortby('time')

    # 5分ごとのデータをサンプリング
    resampled_pos_arr = sorted_pos_arr.resample(time='1T').nearest()
    CLat = np.zeros(len(resampled_pos_arr['time']))
    Sys3 = np.zeros(len(resampled_pos_arr['time']))

    for i in range(len(resampled_pos_arr['time'])):
        CLat[i] = jp.Clat(np.array(resampled_pos_arr[i]))
        Sys3[i] = jp.S3(np.array(resampled_pos_arr[i]))


    sys.path.append('/home/kooki/Documents/Juno_ULC_EMIC/temp')
    import juice_spice_lib as jsl
    import juno_spice_lib as juno

    source_directory = '/home/kooki/Documents/Juno_ULC_EMIC/kernel/kernel/juno/'
    juno.spice_ini(source_directory)

    temp_utc = resampled_pos_arr['time'][0]
    con_s = (temp_utc.values).astype('datetime64[s]')


    if isinstance(con_s, np.datetime64):
        utc = np.datetime_as_string(con_s, unit='s')
    time_delta = lt_iauJup(TARGET='JUNO', utc=utc)

    hours = time_delta.total_seconds() / 3600.0
    #print(f"Local time: {hours} hours")

    lt = np.zeros(len(resampled_pos_arr['time']))
    for i in range(len(resampled_pos_arr['time'])):
        con_s = (resampled_pos_arr['time'][i].values).astype('datetime64[s]')
        if isinstance(con_s, np.datetime64):
            utc = np.datetime_as_string(con_s, unit='s')
        lt[i] = lt_iauJup(TARGET='JUNO', utc=utc).total_seconds() / 3600.0


    R = np.sqrt(resampled_pos_arr[:,0]**2 + resampled_pos_arr[:,1]**2 + resampled_pos_arr[:,2]**2) / RJ

    orb_data = np.zeros((len(resampled_pos_arr['time']), 4))
    orb_data[:,0] = R
    orb_data[:,1] = CLat
    orb_data[:,2] = Sys3
    orb_data[:,3] = lt

    store_data('juno_orbit_rclatsys3lt', data={'x': resampled_pos_arr['time'], 'y': orb_data})

    time_da = resampled_pos_arr['time'][10]
    date_only = str(time_da.values)[:10]
    date_only
    year = date_only[:4]
    month = date_only[5:7]
    year

    mh = 1.67262192369 * 1e-27
    mh2 = 1.6737 * 1e-27 * 2

    mo = 2.6567 * 1e-26
    mo2 = 2.6567 * 1e-26 * 2

    ms = 5.324 * 1e-26

    moh = mo + mh
    moh2 = mo + mh2

    mso = ms + mo
    mso2 = ms + mo2

    q = 1.60217662 * 1e-19

    pyspedas.tinterpol(names='juno_mag_norm', interp_to=f'ratio_{window}_{shift}', newname='juno_mag_norm_intpl')

    b = pytplot.data_quants['juno_mag_norm_intpl']

    label = ['H+', 'H2+', 'O+', 'O2+', 'S+', 'OH+', 'H2O+', 'SO+', 'SO2+']

    cyc_data = np.zeros((len(b['time']), 9))
    cyc_data[:,0] = 1 / (2 * np.pi) * q * b * 1e-9 / mh
    cyc_data[:,1] = 1 / (2 * np.pi) * q * b * 1e-9 / mh2
    cyc_data[:,2] = 1 / (2 * np.pi) * q * b * 1e-9 / mo
    cyc_data[:,3] = 1 / (2 * np.pi) * q * b * 1e-9 / mo2
    cyc_data[:,4] = 1 / (2 * np.pi) * q * b * 1e-9 / ms
    cyc_data[:,5] = 1 / (2 * np.pi) * q * b * 1e-9 / moh
    cyc_data[:,6] = 1 / (2 * np.pi) * q * b * 1e-9 / moh2
    cyc_data[:,7] = 1 / (2 * np.pi) * q * b * 1e-9 / mso
    cyc_data[:,8] = 1 / (2 * np.pi) * q * b * 1e-9 / mso2

    store_data('juno_cyc_freq', data={'x': b['time'], 'y': cyc_data})
    options('juno_cyc_freq', opt_dict={'ylog': 1, 'ytitle': 'Cyclotron Frequency', 'ysubtitle': '[$Hz$]', 'thick':2, 'linestyle': '-'})#,
                                        #'Color': ['red', 'red', 'green', 'green', 'black', 'blue', 'blue', 'purple', 'purple']})

    labels = split_vec('juno_orbit_rclatsys3lt')
    options('juno_orbit_rclatsys3lt_0', opt_dict={'ytitle': 'R[Rj]'})
    options('juno_orbit_rclatsys3lt_1', opt_dict={'ytitle': 'CLat[deg]'})
    options('juno_orbit_rclatsys3lt_2', opt_dict={'ytitle': 'Sys3[deg]'})
    options('juno_orbit_rclatsys3lt_3', opt_dict={'ytitle': 'LT[hours]'})

    store_data('para', data=['para_1024_1', 'juno_cyc_freq', 'para_1024_1'])
    store_data('perp', data=['perp_1024_1', 'juno_cyc_freq', 'perp_1024_1'])
    store_data('norm', data=['norm_1024_1', 'juno_cyc_freq', 'norm_1024_1'])
    store_data('ratio', data=['ratio_1024_1', 'juno_cyc_freq', 'ratio_1024_1'])

    time_ranges = [f"{date_only} {str(hour).zfill(2)}:00:00" for hour in range(0, 24, 3)]
    time_ranges.append(f"{date_only} 23:59:59")
    xlim(time_ranges[0], time_ranges[-1])
    save_dir = '/mnt/d/JUNO_EMIC/component/{}/{}/1day/'.format(year, month)
    os.makedirs(save_dir, exist_ok=True)
    tplot(['para', 'perp', 'norm', 'ratio'], xsize=12, ysize=18 ,var_label=labels, save_png=save_dir + date_only)
    plt.close()

    for t in range(len(time_ranges)-1):
        ptr = [time_ranges[t], time_ranges[t+1]]
        xlim(ptr[0], ptr[1])
        save_dir = '/mnt/d/JUNO_EMIC/component/{}/{}/3hours/'.format(year, month)
        os.makedirs(save_dir, exist_ok=True)
        tplot(['para', 'perp', 'norm', 'ratio'], xsize=12, ysize=18 ,var_label=labels, save_png=save_dir + ptr[0])
        plt.close()



    def remove_invalid_attrs(data_array):
        valid_types = (str, int, float, np.ndarray, list, tuple)
        attrs = {k: v for k, v in data_array.attrs.items() if isinstance(v, valid_types)}
        data_array.attrs = attrs


    juno_mag_norm = data_quants['juno_mag_norm' ]
    juno_mag_rot_perp = data_quants['juno_mag_rot_perp']
    juno_mag_rot_para = data_quants['juno_mag_rot_para']
    juno_cyc_freq = data_quants['juno_cyc_freq']
    para_1024_1 = data_quants['para_1024_1']
    perp_1024_1 = data_quants['perp_1024_1']
    norm_1024_1 = data_quants['norm_1024_1']
    ratio_1024_1 = data_quants['ratio_1024_1']
    juno_orbit_rclatsys3lt = data_quants['juno_orbit_rclatsys3lt']

    remove_invalid_attrs(juno_mag_norm)
    remove_invalid_attrs(juno_mag_rot_perp)
    remove_invalid_attrs(juno_mag_rot_para)
    remove_invalid_attrs(juno_cyc_freq)
    remove_invalid_attrs(para_1024_1)
    remove_invalid_attrs(perp_1024_1)
    remove_invalid_attrs(norm_1024_1)
    remove_invalid_attrs(ratio_1024_1)
    remove_invalid_attrs(juno_orbit_rclatsys3lt)

    # 複数の DataArray を Dataset にまとめる
    dataset = xr.Dataset({
        "juno_mag_norm": juno_mag_norm,
        "juno_mag_rot_perp": juno_mag_rot_perp,
        "juno_mag_rot_para": juno_mag_rot_para,
        "juno_cyc_freq": juno_cyc_freq
    })

    # ディレクトリを指定して保存
    mag_line = '/mnt/d/Juno_xarray/mag_line/{}/{}/'.format(year, month)
    os.makedirs(mag_line, exist_ok=True)
    file_path = mag_line + "mag_norm_perp_para_{}.nc".format(date_only)

    # ファイルが存在する場合は削除する
    if os.path.exists(file_path):
        os.remove(file_path)

    # データセットを保存する
    dataset.to_netcdf(file_path)

    dataset = xr.Dataset({
        "para_1024_1": para_1024_1,
        "perp_1024_1": perp_1024_1,
        "norm_1024_1": norm_1024_1,
        "ratio_1024_1": ratio_1024_1
    })

    mag_spec = '/mnt/d/Juno_xarray/mag_spec/{}/{}/'.format(year, month)
    os.makedirs(mag_spec, exist_ok=True)
    file_path = mag_spec + "spec1024_para_perp_norm_ratio_{}.nc".format(date_only)

    # ファイルが存在する場合は削除する
    if os.path.exists(file_path):
        os.remove(file_path)

    # データセットを保存する
    dataset.to_netcdf(file_path)

    orb = '/mnt/d/Juno_xarray/orb/{}/{}/'.format(year, month)
    os.makedirs(orb, exist_ok=True)
    file_path = orb + "orbit_rclatsys3lt_{}.nc".format(date_only)

    # ファイルが存在する場合は削除する
    if os.path.exists(file_path):
        os.remove(file_path)

    # データセットを保存する
    juno_orbit_rclatsys3lt.to_netcdf(file_path)
    pytplot.del_data()