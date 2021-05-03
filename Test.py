import numpy as np
import pandas as pd
import Calculations as Ca
from pprint import pprint


# Variance ratio test for market efficiency by Lo MacKinlay (1988)

# Aleksius Kurkela
# kurkela.aleksius@gmail.com


def estimate(data, lags):
    """
    :param data: Data frame of prices [p1, p2, p3 ... pt] with the label "Price" as header
    :param lag:  Int lag for interpreting the price array
    :return: Array of tuples in the following way {Variance ratio for set lag, Heteroscedasticity, Homoscedasticity}
    """
    TargetPrices = data['Price'].to_numpy(dtype=np.float64)
    TestResult = []
    k = 0
    while k < len(lags):
        vr, res1, res2 = Ca.VR(np.log(TargetPrices), int(lags[k]))
        TestResult.append({
            f'Homoscedasticity': res1,
            f'Heteroscedasticity': res2,
            f'Variance Ratio': vr,
            f'k=': lags[k]
        })
        k += 1
    return TestResult


def main():
    # Create random prices
    np.random.seed(13)
    steps = np.random.normal(0, 1, size=100000)
    steps[0] = 0
    P = 10000 + np.cumsum(steps)
    data = pd.DataFrame(P, columns=['Price'])
    lags = []
    k = int(input("Input the amount of lags wanted: "))
    n = 0
    while n < k:
        lag = input("Set lag: ")
        if lag == "/s":
            break
        lags.append(lag)
        n += 1
    result = estimate(data, lags)
    pprint(result)


main()
