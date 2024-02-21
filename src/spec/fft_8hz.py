import numpy as np
import pytplot
from datetime import datetime
import scipy.signal as signal
import numpy as np
from datetime import datetime, timedelta

def convert_string_to_datetime(input_string):
    elements = input_string.split()
    date_string = f"{elements[0]}-{elements[1]}T{elements[2].zfill(2)}:{elements[3].zfill(2)}:{elements[4].zfill(2)}.{elements[5]}"
    return date_string

def convert_date(input_date):
    date_object = datetime.strptime(input_date, '%Y-%jT%H:%M:%S.%f')
    return f'{date_object.strftime("%Y-%m-%d")}T{date_object.strftime("%H:%M:%S.%f")}'

def convert_to_float_list(input_list):
    return [list(map(float, sublist)) for sublist in input_list]

def fgm_spec(sts_file_path, window):

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

    fs = 8
    f, t, Sxx = signal.spectrogram(norm, fs, nperseg=window, noverlap=window // 2)

    base_time_str = times[0]

    # Convert base time to datetime object
    base_time = datetime.strptime(base_time_str, '%Y-%m-%dT%H:%M:%S.%f')

    # Add t to base time and convert to strings
    resulting_times = [(base_time + timedelta(seconds=time)).strftime('%Y-%m-%dT%H:%M:%S.%f') for time in t]
    middle_times = np.array(resulting_times, dtype=np.datetime64)

    pytplot.store_data(f'juno_spec_{window}', data={'x': middle_times, 'y': Sxx.T, 'v': f})
    pytplot.options(f'juno_spec_{window}', opt_dict={'ylog': 1, 'zlog': 1, 'ytitle': 'Frequency', 'ysubtitle':'[Hz]', 
                                        'xtitle': 'Time [sec]', 'spec': True, 'yrange': [0.001, 4],
                                        'ztitle': 'Intensity [$(nT)^2/Hz$]'})

    return 'juno_mag_norm', f'juno_spec_{window}', 'juno_orbit'

############################################################################################################
# example usage

# file_path is the path for the FGM STS file
# window is the window size for the spectrogram
# fft_8hz.py file is for the 8hz sampling frequency data

""" 
file_path = 'path_to_file/fgm_sts_8hz.txt' # set your file paht
window = 256

fgm_spec(file_path, window)
pytplot.tplot(['juno_mag_norm', f'juno_spec_{window}', 'juno_orbit'])

 """

