#Importing Numpy, Matplotlib, Math and Stats from Scipy
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import stats

def get_info(arr, n):
    new_arr = np.sort(arr)
    sum_arr = np.sum(new_arr)
    mean = sum_arr/n
    var = (1/(n-1))*np.sum((new_arr-mean)**2)
    std_dev = math.sqrt(var)
    arr_Star = (new_arr - mean)/std_dev

    return new_arr, sum_arr, mean, var, std_dev, arr_Star

#Data Normalization-------------------------------------------------------------
print('#Data Normalization-------------------------------------------------------------')
orig_xi = np.array([])
N = int(input('Enter the size of the measurement list: '))

for i in range(N):
    entered_val = float(input('Enter element: '))
    orig_xi = np.append(orig_xi, entered_val)

xi, sum_xi, mean, var, std_dev, xi_Star = get_info(orig_xi, N)

print('xi: ' + str(xi))
print('N: ' + str(N))
print('Sum xi: ' + str(sum_xi))
print('Mean: ' + str(mean))
print('Variance: ' + str(var))
print('Standard Deviation: ' + str(std_dev))
print('xi Star: ' + str(np.around(xi_Star, decimals = 5)))

num_bins = int(input('Enter the amount of histogram bins that you want: '))

#Chi-Squared Test-------------------------------------------------------------
print('#Chi-Squared Test-------------------------------------------------------------')
x_min = -5.0
x_max = 5.0

n = 1000
dx = (x_max-x_min)/n

x = np.zeros(n)
func = np.zeros(n)

for i in range(n):
    x[i] = x_min + (i + 0.5)*dx
    func[i] = (1.0/math.sqrt((2.0*math.pi)))*math.exp(-(x[i])**2/(2.0))

lim1 = float(input('Enter the limit 1: '))
lim2 = float(input('Enter the limit 2: '))
lim3 = float(input('Enter the limit 3: '))

n1 = int(input('Enter the value for n1: '))
n2 = int(input('Enter the value for n2: '))
n3 = int(input('Enter the value for n3: '))
n4 = int(input('Enter the value for n4: '))

j1, y1 = np.zeros(0), np.array([-1e12, lim1])
j2, y2 = np.zeros(0), np.array([lim1, lim2])
j3, y3 = np.zeros(0), np.array([lim2, lim3])
j4, y4 = np.zeros(0), np.array([lim3, 1e12])


for i in range(n):
    if y1[0] < x[i] <= y1[1]:
        j1 = np.append(j1, func[i])
    elif y2[0] < x[i] <= y2[1]:
        j2 = np.append(j2, func[i])
    elif y3[0] < x[i] <= y3[1]:
        j3 = np.append(j3, func[i])
    elif y4[0] < x[i] <= y4[1]:
        j4 = np.append(j4, func[i])

p1 = np.trapz(j1, dx=dx)
p2 = np.trapz(j2, dx=dx)
p3 = np.trapz(j3, dx=dx)
p4 = np.trapz(j4, dx=dx)

chi1 = (n1/N-p1)**2/p1
chi2 = (n2/N-p2)**2/p2
chi3 = (n3/N-p3)**2/p3
chi4 = (n4/N-p4)**2/p4

chi_square = chi1 + chi2 + chi3 + chi4
chi_prob = (1.0 - stats.chi2.cdf(chi_square, 1))

print('n1: ' +  str(n1) + ' | ' + 'y1: ' +  '-inf' + ' | ' + 'y2: ' +  str(y1[1]) + ' | ' + 'p1: ' +  str(p1))
print('n2: ' +  str(n2) + ' | ' + 'y1: ' +  str(y2[0]) + ' | ' + 'y2: ' +  str(y2[1]) + ' | ' + 'p2: ' +  str(p2))
print('n3: ' +  str(n3) + ' | ' + 'y1: ' +  str(y3[0]) + ' | ' + 'y2: ' +  str(y3[1]) + ' | ' + 'p3: ' +  str(p3))
print('n4: ' +  str(n4) + ' | ' + 'y1: ' +  str(y4[0]) + ' | ' + 'y2: ' +  '+inf' + ' | ' + 'p4: ' +  str(p4))
print('Chi Squared: ' + str(chi_square))
print('Chi Squared Probability: ' + str(chi_prob))

#Outlier Detection-------------------------------------------------------------
print('#Outlier Detection-------------------------------------------------------------')

T = float(input('Enter you Tau value based on Chauvenet criterion for N = ' + str(N) + ': '))
od_xi = xi

for i in range(np.size(od_xi)):
    if T*std_dev <= abs(od_xi[i] - mean):
        print('Outlier found in: ' + od_xi[i])
        od_xi = np.delete(od_xi, od_xi[i])

od_xiNew, sum_odxi, mean_od, var_od, std_oddev, xi_odStar = get_info(od_xi, np.size(od_xi))

print('Outlier xi: ' + str(od_xiNew))
print('Outlier N: ' + str(np.size(od_xi)))
print('Outlier Sum xi: ' + str(sum_odxi))
print('Outlier Mean: ' + str(mean_od))
print('Outlier Variance: ' + str(var_od))
print('Outlier Standard Deviation: ' + str(std_oddev))
print('Outlier xi Star: ' + str(np.around(xi_odStar, decimals = 5)))

#Plotting Histogram-------------------------------------------------------------
plt.hist(xi_Star, density=True, bins = num_bins)
plt.plot(x, func)
plt.title('Normalized Histogram')
plt.xlabel('Normalized Data')
plt.ylabel('Probability')
plt.show()

#Plotting Pattern-------------------------------------------------------------
x_domain = np.arange(N)
plt.scatter(x_domain, orig_xi)
plt.title('Pattern Detection')
plt.xlabel('Data')
plt.ylabel('Measured Value')
plt.show()

