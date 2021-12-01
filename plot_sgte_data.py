import numpy as np
import matplotlib.pyplot as plt


def main():
    temp_range = np.arange(298.15, 368.3)
    gpres = get_gpres(A=7.042095 * 10 ** (-6), a0=2.3987 * 10 ** (-5), a1=2.569 * 10 ** (-8), T=temp_range, p=10 ** 5)
    gmag = get_gmag(T=temp_range, p=10 ** 5, Tcrit=1043, B0=2.22)

    G = evaluate_function(temp_range, {'a': -5198.294, 'b': 53.913855, 'c': 10.726, 'd': 27.3801 * 10 ** (-3), 'e': 8.179537 * 10 ** (-6), 'f': 0})

    temp_range_1 = np.arange(368.3, 1300)
    G1 = evaluate_function(temp_range_1, {'a': -6475.706, 'b': 94.182332, 'c': 17.8693298, 'd': 10.936877 * 10 ** (-3), 'e': 1.406467 * 10 ** (-6), 'f': 36871})

    temp_range_2 = np.arange(1300, 1301)
    G2 = evaluate_function(temp_range_2, {'a': -12485.546, 'b': 188.304687, 'c': 32, 'd': 0, 'e': 0, 'f': 0})

    temp_range_3 = np.arange(298.15, 335)
    G3 = evaluate_function(temp_range_3, {'a': -4196.575, 'b': 85.63027, 'c': 17.413, 'd': 9.93935E-3, 'e': 0.070062E-6, 'f': 1250})

    temp_range_4 = np.arange(335, 388.36)
    G4 = evaluate_function(temp_range_4, {'a': 1790361.982, 'b': -44195.451432, 'c': -7511.6194258, 'd': 13985.517511E-3, 'e': -4838.738601E-6, 'f': -79880891})

    plt.scatter(temp_range, G, s=1)
    plt.scatter(temp_range_1, G1, s=1)
    plt.scatter(temp_range_2, G2, s=1)
    plt.scatter(temp_range_3, G3, s=1)
    plt.scatter(temp_range_4, G4, s=1)

    plt.grid(True)
    plt.show()


def evaluate_function(T, coefficients: dict, gpres=0, gmag=0):
    a = coefficients['a']
    b = coefficients['b']
    c = coefficients['c']
    d = coefficients['d']
    e = coefficients['e']
    f = coefficients['f']

    return a + b * T - c * T * np.log(T) - d * T ** 2 - e * T ** 3 + f * 1/T + gpres + gmag








if __name__ == '__main__':
    main()