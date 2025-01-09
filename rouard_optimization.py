# -*- coding: utf-8 -*-
"""
@author: Bo Wang
@file: rouard_optimization.py
@time: 2023/6/16 12:04

This module is referred from "Krimi 2016". A model based reflection signal simulation is inspired by Rouard method to
estimate the thickness of individual layers of a multi-layer structure. The s polarization is used.
"""
from functools import partial

import numpy as np
from scipy.optimize import differential_evolution


def get_transfer_func(d, params):
    """Calculate the transfer function of the three-layer structure.

    Args:
        d: float
            The thickness of the second layer, unit mm.
        params: tuple
            See get_transfer_func_params

    Returns:
        array_like
            The frequency varying transfer function.
    """
    r_one_two = params[0]
    r_two_one = params[1]
    t_one_two = params[2]
    t_two_one = params[3]
    r_two_three = params[4]
    sl_phase_shift = params[5] * d
    phase_factor = np.exp(1j * sl_phase_shift)
    trans_func = r_one_two \
                 + t_one_two * t_two_one * r_two_three * phase_factor \
                 / (1 - r_two_three * r_two_one * phase_factor)
    # trans_func = r_one_two

    return trans_func


def get_transfer_func_params(theta_one, freq, n_one, n_two, n_three=None):
    """Calculate the parameters that should be pre-calculated for the transfer function according to Fresnel's law.
    One, two, three respectively represents the layers of a three-layer structure.

    Args:
        theta_one: float
            The incidence angle to the upper surface, unit rad.
        freq: array_like
            The frequencies of the transfer function.
        n_one: float, complex, array_like
            The refractive index of the upper layer.
        n_two: float, complex, array_like
            The refractive index of the middle layer.
        n_three: float, complex, array_like, optional
            The refractive i ndex of the bottom layer. If not given, the bottom layer is metalic so that the reflection
            coefficient regarding the layer is -1.

    Returns:
        tuple
            The reflective coefficients and transmission coefficients regarding the interfaces and the factor regarding
           the phase shift induced by the middle layer.
    """
    n_list = [n_one, n_two, n_three]
    n_sizes = [len(n) for n in n_list if isinstance(n, np.ndarray) or isinstance(n, list)]
    if np.any(np.asarray(n_sizes) != len(freq)):
        raise ValueError('The refractive index should be a float, a complex number, or an array of the same length '
                         'as freq.')

    theta_two = _get_refractive_angle(theta_one, n_one, n_two)
    r_one_two = _get_reflection_coeff(theta_one, theta_two, n_one, n_two)
    r_two_one = _get_reflection_coeff(theta_two, theta_one, n_two, n_one)
    t_one_two = _get_transmission_coeff(theta_one, theta_two, n_one, n_two)
    t_two_one = _get_transmission_coeff(theta_two, theta_one, n_two, n_one)

    if n_three is None:
        r_two_three = -1
    else:
        theta_three = _get_refractive_angle(theta_two, n_two, n_three)
        r_two_three = _get_reflection_coeff(theta_two, theta_three, n_two, n_three)

    # Phase shift factor regarding the second layer.
    phase_shift_factor = _get_phase_shift_factor(theta_two, n_two, freq)

    return r_one_two, r_two_one, t_one_two, t_two_one, r_two_three, phase_shift_factor


def optimize_correlation(bounds, ref, signal, params, half_wave_loss=True, precision=1.e-3):
    """Find the maximum correlation coefficient between the reconstructed signal and the original signal. This method
    works well for signal-layer measurement.
    See optimize

    Args:
        precision: float
            The precision of the thickness, default value is 1 um.
    """
    d_list = np.arange(bounds[0], bounds[1], precision)
    corr_coeffs = []

    for d in d_list:
        trans_func = get_transfer_func(d, params)
        simulated = simulate_reflection_signal(ref, trans_func, half_wave_loss)
        offset = np.argmax(signal) - np.argmax(simulated)
        corr_coeff = np.corrcoef(np.roll(simulated, offset), signal)[0][1]
        corr_coeffs.append(corr_coeff)

    max_corr_coeff_index = np.argmax(corr_coeffs)
    optimal_d = d_list[max_corr_coeff_index]

    return optimal_d


def optimize(bounds, ref, signal, params, half_wave_loss=True):
    """Find the optimal value of the thickness of the middle layer using differential evolution equation. Referred from
    'R. Storn and K. Price 1997'. See simulate_reflection_signal

    Args:
        bounds: np.ndarray, np.array([lb, ub],[lb, ub])
            The upper and lower bounds of the searching range, as well as the range of the offset between the simulated
            data and the real data.
``      ref: np.ndarray
            The reference signal.
        signal: np.ndarray
            The transmissive THz signal of the sample.
        params: array_like
            See Also get_transfer_func_params.

    Returns:
        d: float
            The estimated thickness of the middle layer.
        offset: float
            The offset between the simulated data and the real data.
    """
    # Perform the differential evolution search
    result = differential_evolution(_objective, bounds, args=(ref, signal, params, half_wave_loss))

    # Summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])

    # Evaluate solution
    d = result['x'][0]
    offset = result['x'][1]

    return d, offset


def tlbo_optimize(bounds, ref, signal, params, half_wave_loss=True):
    """
    The function is the standard TLBO algorithm, designed to optimize the solution to a target function with teaching
    and learning processes.

    Args:
        bounds: np.ndarray, np.array([lb, ub],[lb, ub])
            The upper and lower bounds of the searching range, as well as the range of the offset between the simulated
            data and the real data.
``      ref: np.ndarray
            The reference signal.
        signal: np.ndarray
            The transmissive THz signal of the sample.
        params: array_like
            See Also get_transfer_func_params.

    Returns:
        d: float
            The estimated thickness of the middle layer.
        offset: float
            The offset between the simulated data and the real data.
    """
    np.random.seed(0)  # 为了结果可重复，设置随机种子

    # 参数设置
    pn = 30  # 种群大小
    gn = 50  # 迭代次数
    dim = 2  # 种群维度（d,offset）

    # 种群范围设置
    population_min = bounds[:, 0]
    population_max = bounds[:, 1]

    # 创建目标函数
    fun = partial(_objective, ref=ref, signal=signal, params=params, half_wave_loss=half_wave_loss)

    # 初始化种群
    population = population_min + (population_max - population_min) * np.random.rand(pn, dim)
    fitness = np.array([fun(individual) for individual in population])
    best_fit = np.min(fitness)

    for j in range(gn):
        population_mean = np.mean(population, axis=0)
        best_fit_vector = population[np.argmin(fitness), :]

        # 教学阶段
        new_population = np.copy(population)
        for i in range(pn):
            delta = best_fit_vector - np.round(1 + np.random.rand() * population_mean)
            new_population[i, :] = population[i, :] + np.random.rand() * delta
            new_population[i, :] = np.clip(new_population[i, :], population_min, population_max)  # 越界处理
            fitness_new = fun(new_population[i, :])
            if fitness_new < fitness[i]:
                population[i, :] = new_population[i, :]
                fitness[i] = fitness_new

        # 学习阶段
        for i in range(pn):
            k = np.random.randint(0, pn)
            while k == i:
                k = np.random.randint(0, pn)

            if fun(population[i, :]) < fun(population[k, :]):
                new_population[i, :] = population[i, :] + np.random.rand() * (population[i, :] - population[k, :])
            else:
                new_population[i, :] = population[i, :] + np.random.rand() * (population[k, :] - population[i, :])

            new_population[i, :] = np.clip(new_population[i, :], population_min, population_max)  # 越界处理
            fitness_new = fun(new_population[i, :])
            if fitness_new < fitness[i]:
                population[i, :] = new_population[i, :]
                fitness[i] = fitness_new

        # 更新全局最优
        if np.min(fitness) < best_fit:
            best_fit = np.min(fitness)
            best_fit_vector = population[np.argmin(fitness)]
            d = best_fit_vector[0]
            offset = best_fit_vector[1]

    return d, offset


def simulate_reflection_signal(ref, trans_func, half_wave_loss=True):
    """Simulate the reflection signal by multiplying the reference signal and the transfer function in the frequency
    domain.

    Args:
        ref: array_like
            The reference signal regarding the electric field of the incident pulse.
        trans_func: array_like
            The transfer function of the reflective system.
        half_wave_loss: bool
            If true, a phase shift of pi has to be applied to the reference signal to compensate for the half-wave loss
            induced by the metalic surface.

     Returns:
         array_like
            The simulated reflection signal.
    """
    ref_freq = np.fft.fft(ref)

    if half_wave_loss is True:
        ref_freq *= np.exp(1j * np.pi)

    reflect_signal_freq = ref_freq * trans_func
    reflect_signal = np.real(np.fft.ifft(reflect_signal_freq))

    return reflect_signal


def _get_phase_shift_factor(theta, n, freq):
    """ Calculate the phase shift factor. The product of the factor and the layer's thickness is hte phase shift.
        See get_transfer_func_params
    """
    # Unit mm / ps.
    light_speed = 0.3
    phase_shift_factor = 2 * n / np.cos(theta) \
                         * freq * 2 * np.pi / light_speed

    return phase_shift_factor


def _get_refractive_angle(theta_in, n_in, n_out):
    """See _get_transmission_coeff"""
    refrac_angle = np.arcsin(n_in * np.sin(theta_in) / n_out)

    return refrac_angle


def _get_transmission_coeff(theta_in, theta_out, n_in, n_out):
    """Calculate the transmission coefficient according to Fresnel law.

    Args:
        theta_in: float
            The incidence angle, unit rad.
        theta_out: float
            The refractive angle, unit rad.
        n_in: float
            The refractive index of the incident substance.
        n_out: float
            The refractive index of the refractive substance.

    Returns:
        float
            The transmission coefficient.
    """
    trans_coeff = 2 * n_in * np.cos(theta_in) \
                  / (n_in * np.cos(theta_in) + n_out * np.cos(theta_out))

    return trans_coeff


def _get_reflection_coeff(theta_in, theta_out, n_in, n_out):
    """Calculate the reflection coefficient according to Fresnel law.

    Args:
        theta_in: float
            The incidence angle, unit rad.
        theta_out: float
            The refractive angle, unit rad.
        n_in: float
            The refractive index of the incident substance.
        n_out: float
            The refractive index of the refractive substance.

    Returns:
        float
            The reflection coefficient.
    """
    ref_coeff = (n_in * np.cos(theta_in) - n_out * np.cos(theta_out)) \
                / (n_in * np.cos(theta_in) + n_out * np.cos(theta_out))

    return ref_coeff


def _objective(x, ref, signal, params, half_wave_loss):
    """ The objective function of the differential evolution optimization.

    Args:
        x: list
            The first item is the thickness, the last item is the offset between the simulated signal and the original
            signal.
    """
    # The thickness.
    trans_func = get_transfer_func(x[0], params)
    simulated = simulate_reflection_signal(ref, trans_func, half_wave_loss)
    loss = np.sum(np.power(np.roll(simulated, int(x[1])) - signal, 2))

    return loss
