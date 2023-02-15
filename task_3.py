# Произвести вычисления как в пункте 2, но с вычислением intercept.
# Учесть, что изменение коэффициентов должно производиться на каждом
# шаге одновременно (то есть изменение одного коэффициента не должно
# влиять на изменение другого во время одной итерации).

import numpy as np

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])


def mse(b0, b1, x, y, n):
    return np.sum((y - (b0 + b1 * x))**2) / n


n = 10
alpha = 0.00001
b1 = 0.1
b0 = 0.1

mse_min = np.sum((b1+b0*zp-ks)**2)/len(zp)


for i in range(6000001):
    b1 -= alpha * (2 * np.sum((b1 + b0 * zp) - ks) / len(zp))
    b0 -= alpha * (2 * np.sum(((b1 + b0 * zp) - ks) * zp) / len(ks))
    if np.sum(((b1 + b0 * zp) - ks) ** 2) / len(zp) > mse_min:
        print(f"На {i} итерации достигнут минимум mse={mse_min}\n\
    intercept равен {b1:.2f} и Коэффициент линейной регрессии {b0:.2f}")
        break
    else:
        mse_min = np.sum(((b1 + b0 * zp) - ks) ** 2) / len(zp)
        if i % 100000 == 0:
            print(f"Iteranion = {i}, b0 = {b0}, b1 = {b1}, mse = {mse(b0, b1, zp, ks, n)}")