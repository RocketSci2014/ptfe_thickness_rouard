# -*- coding: utf-8 -*-
"""
@author: Bo Wang
@file: rouard_script.py
@time: 6/23/2024 6:14 PM
"""
import time
from math import radians

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlbo_optimization import get_transfer_func_params, get_transfer_func, simulate_reflection_signal, tlbo_optimize


def estimate_layer_thickness_rouard():
    df = pd.read_csv('data/ptfe complex n.csv')
    freq = df['freq'].values
    n = df['n'].values
    k = df['k'].values
    n_comp = n + 1j * k

    data_path = 'data/离轴7.5°聚四氟乙烯薄片.xls'
    df = pd.read_excel(data_path)
    t = df['位置'].values
    ref = df['初始值'].values
    sample_names = df.columns[2:]

    for sample_name in sample_names:
        signal = df[sample_name].values
        theta = 7.5
        theta = radians(theta)
        trans_func_params = get_transfer_func_params(theta, freq, 1, n_comp)

        start = time.time()
        bounds = np.array([[0.02, 0.04], [-500, 500]])
        d, offset = tlbo_optimize(bounds, ref, signal, trans_func_params)

        # This bias is found by fitting numerous measurement data.
        print(f'Elapsed time: {time.time() - start} s.')

        thickness = sample_name.split('_')[1]
        print(f'Estimated thickness for {thickness}: {d} mm.')

        trans_func = get_transfer_func(d, trans_func_params)
        simulated = simulate_reflection_signal(ref, trans_func)
        simulated = np.roll(simulated, int(offset))

        plt.rcParams['font.size'] = 20
        plt.figure(thickness, figsize=(11, 9))
        plt.subplot(2, 1, 1)
        plt.plot(t, simulated / np.max(signal), linewidth=2, label='reconstructed')
        plt.plot(t, signal / np.max(signal), linewidth=2, label='signal')
        plt.xlabel('Time (ps)')
        plt.ylabel('THz-TDS (a.u.)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(t, (signal - simulated) / np.max(signal), linewidth=2)
        plt.xlabel('Time (ps)')
        plt.ylabel('Deviation (a.u.)')
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    estimate_layer_thickness_rouard()
