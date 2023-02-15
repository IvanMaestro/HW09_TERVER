# Даны значения величины заработной платы заемщиков банка (zp) и
# значения их поведенческого кредитного скоринга (ks):
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
# Используя математические операции, посчитать коэффициенты
# линейной регрессии, приняв за X заработную плату (то есть, zp - признак),
# а за y - значения скорингового балла (то есть, ks - целевая переменная).
# Произвести расчет как с использованием intercept, так и без.

import numpy as np
import matplotlib.pyplot as plt

print('Расчет с использованием intercept')

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

b0 = (np.mean(zp * ks) - np.mean(zp) * np.mean(ks)) / (np.mean(zp**2) - np.mean(zp)**2)
print(f"Коэффициент линейной регрессии: {round(b0, 2)}")
b1 = np.mean(ks) - b0 * np.mean(zp)
print(f"intercept: {round(b1, 2)}")

print()
print('Расчет без использования intercept')

x = zp.reshape(len(zp), 1)
y = ks.reshape(len(ks), 1)
b = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T @ y)
print(b)

plt.scatter(zp, ks)
plt.plot(zp, b1 + b0 * zp)
plt.plot(x, b * x)
plt.show()
