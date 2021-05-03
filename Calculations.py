import numpy as np


def VR(log_prices, k):
    returns = np.diff(
        log_prices)  # Logarithmic returns by calculated with price difference [x_1, x_2...x_t], where x_n = ln(p_n/p_(n-1))
    T = len(returns)  # Lenght of the return series
    mu = np.mean(returns)  # Mean of the log return
    sqr_demean_returns = np.square(returns - mu)  # Calculate the factor within the bracets
    var1 = np.sum(sqr_demean_returns) / (T - 1)  # One period return variance

    # Variance for the whole period
    kreturns = (log_prices - np.roll(log_prices, k))[
               k:]  # Return series for the sum calculated by discarding the first k elements
    m = k * (T - k + 1) * (1 - k / T)  # m factor for the k period return sum
    vark = np.sum(np.square(kreturns - k * mu)) / m

    # Variance ratio
    vr = vark / var1
    Aarr = np.square(np.arange(k - 1, 0, step=-1, dtype=np.int) * 2 / k)
    Barr = np.empty(k - 1, dtype=np.float64)
    for j in range(1, k):
        Barr[j - 1] = np.sum((sqr_demean_returns * np.roll(sqr_demean_returns, j))[j + 1:])
    delt_arr = (T * Barr) / np.square(np.sum(sqr_demean_returns))
    assert len(delt_arr) == len(Aarr) == k - 1
    phi1 = 2 * (2 * k - 1) * (k - 1) / (3 * k * T)
    phi2 = np.sum(Aarr * delt_arr)

    # Hetero and homoscedasticity
    VR_Ho = (vr - 1) / np.sqrt(phi1)
    VR_He = (vr - 1) / np.sqrt(phi2)

    return vr, VR_Ho, VR_He
