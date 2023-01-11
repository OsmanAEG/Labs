#Importing Numpy, Matplotlib, and Math
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import optimize

##########################################################
#Curve Fitting
def curve_fit_func(x, a, b):
    return a*np.sin(b*x)

def curve_fit_param(x, y):
    params, params_c = optimize.curve_fit(curve_fit_func, x, y, p0 = [2, 2]) 
    return params

##########################################################
#Calculation Executions

def part1_calc(p, p_v, v, t):
    h = np.zeros(6)
    q = np.zeros(6)
    for i in range(6):
        h[i] = p[i] - p_v[i]
        q[i] = v[i]/t[i]
    
    return h, q

def part2_calc(p, p_v, v, t, N_m, Ratio, T):
    h = np.zeros(6)
    q = np.zeros(6)

    N_p   = Ratio*N_m
    P_in  = np.zeros(6)
    eta   = np.zeros(6)
    P_out = np.zeros(6)

    for i in range(6):
        h[i] = p[i] - p_v[i]
        q[i] = v[i]/t[i]
        P_in[i]  = (2.0*math.pi*N_p*T[i])/60.0
        eta[i]   = (0.997*9.81*q[i]*h[i])/P_in[i]
        P_out[i] = eta[i]*P_in[i]

    return h, q, P_in, eta, P_out

##########################################################
#Printing Results

def print_Part1(h, q):
    print('######### PUMP HEAD #########')
    print(h)
    print('######### FLOW RATE #########')
    print(q)

def print_Part2(h, q, P_in, eta, P_out):
    print('######### PUMP HEAD #########')
    print(h)
    print('######### FLOW RATE #########')
    print(q)
    print('######### Power Input #########')
    print(P_in)
    print('######### Efficiency #########')
    print(eta)
    print('######### Power Output #########')
    print(P_out)


##########################################################
#Part 1

def turbineP1():
    p   = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    p_v = np.array([-1.75, -1.5, -0.5, 0.0, 0.0, 0.0])
    t   = np.array([9.7953, 11.538, 15.527, 23.024, 8.2605, 10e12])
    v   = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 0.0])

    print('######### Part 1: Turbine #########')
    h, q = part1_calc(p, p_v, v, t)
    print_Part1(h, q)
    return h, q

def centrifugalP1():
    p   = np.array([3.5, 3.75, 4.0, 4.25, 4.5, 5.25])
    p_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t   = np.array([7.162, 8.874, 13.452, 18.92, 21.465, 10e12])
    v   = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 0.0])

    print('######### Part 1: Centrifugal #########')
    h, q = part1_calc(p, p_v, v, t)
    print_Part1(h, q)
    return h, q

def gearP1():
    p   = np.array([2.0, 6.0, 10.0, 14.0, 18.0, 22.0])
    p_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t   = np.array([14.1603, 14.478, 15.018, 15.6943, 16.555, 10e12])
    v   = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 0.0])

    print('######### Part 1: Gear #########')
    h, q = part1_calc(p, p_v, v, t)
    print_Part1(h, q)
    return h, q

def axialP1():
    p      = np.array([0.6, 0.75, 0.9, 1.05, 1.2, 1.45])
    p_v    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    weir   = np.array([0.069, 0.062, 0.054, 0.042, 0.028, 10e-12])
    h = np.zeros(6)
    q = np.zeros(6)

    Cd = 0.6
    b  = 0.05
    g  = 9.81

    for i in range(6):
        h[i] = p[i] - p_v[i]
        q[i] = Cd*(2.0/3.0)*b*math.sqrt(2.0*g)*weir[i]**1.5*1000.0

    print('######### Part 1: Axial #########')
    print_Part1(h, q)
    return h, q


##########################################################
#Part 2

def turb1400_P2():
    p   = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    p_v = np.array([-1.75, -1.5, -0.5, 0.0, 0.0, 0.0])
    t   = np.array([10.344, 11.481, 15.248, 15.6943, 8.549, 10e12])
    v   = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 0.0])

    T   = np.array([0.61, 0.64, 0.74, 0.83, 0.91, 0.99])
    Ratio = 27.0/14.0
    N_m = 1400.0

    print('######### Part 2: Turbine 1400 #########')
    h, q, P_in, eta, P_out = part2_calc(p, p_v, v, t, N_m, Ratio, T)
    print_Part2(h, q, P_in, eta, P_out)
    return h, q, P_in, eta, P_out

def turb1300_P2():
    p   = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 22.5])
    p_v = np.array([-1.5, -1.0, -0.25, 0.0, 0.0, 0.0])
    t   = np.array([10.949, 12.816, 18.08, 6.761, 15.607, 10e12])
    v   = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 0.0])

    T   = np.array([0.53, 0.59, 0.67, 0.75, 0.86, 0.90])
    Ratio = 27.0/14.0
    N_m = 1300.0

    print('######### Part 2: Turbine 1300 #########')
    h, q, P_in, eta, P_out = part2_calc(p, p_v, v, t, N_m, Ratio, T)
    print_Part2(h, q, P_in, eta, P_out)
    return h, q, P_in, eta, P_out

def turb1200_P2():
    p   = np.array([0.0, 2.5, 7.5, 10.0, 12.5, 17.5])
    p_v = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t   = np.array([12.072, 12.443, 19.297, 24.617, 7.436, 10e12])
    v   = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 0.0])

    T   = np.array([0.46, 0.53, 0.60, 0.63, 0.68, 0.81])
    Ratio = 27.0/14.0
    N_m = 1200.0

    print('######### Part 2: Turbine 1200 #########')
    h, q, P_in, eta, P_out = part2_calc(p, p_v, v, t, N_m, Ratio, T)
    print_Part2(h, q, P_in, eta, P_out)
    return h, q, P_in, eta, P_out

##########################################################
#Plotting Figures

def plot_fig(fig_number, x1, y1, x2, y2, x3, y3, x4, y4, x_name, y_name, plot_name, a, b, c, d):
    fig = plt.figure(fig_number)
    xmax = max(np.max(x1), np.max(x2), np.max(x3), np.max(x4))
    xmin = min(np.min(x1), np.min(x2), np.min(x3), np.min(x4))

    ymax = max(np.max(y1), np.max(y2), np.max(y3), np.max(y4))
    ymin = min(np.min(y1), np.min(y2), np.min(y3), np.min(y4))

    #Plotting Without Curve Fitting
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2, 'g')
    plt.plot(x3, y3, 'b')
    plt.plot(x4, y4, 'y')

    #Plot Lengens, Titles, and Saved Images
    plt.legend([a, b, c, d])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.savefig(plot_name)
    fig.show()

##########################################################
#Executing Problems   

def problem1_execution():
    h1, q1 = turbineP1()
    h2, q2, = centrifugalP1()
    h3, q3 = gearP1()
    h4, q4 = axialP1()

    plot_fig(1, q1, h1, q2, h2, q3, h3, q4, h4, 'Flow Rate (L/s)', 'Pump Head (mH20)', 'Part1_Flow-vs-Head.png', 'Turbine', 'Centrifugal', 'Gear', 'Axial')

    input()

def problem2_execution():
    h1, q1, P_in1, eta1, P_out1 = turb1400_P2()
    h2, q2, P_in2, eta2, P_out2 = turb1300_P2()
    h3, q3, P_in3, eta3, P_out3 = turb1200_P2()

    plot_fig(1, q1, h1, q2, h2, q3, h3, 0.0, 0.0, 'Flow Rate (L/s)', 'Pump Head (mH20)', 'Part2_Flow-vs-Head.png', '1400 RPM', '1300 RPM', '1200 RPM', ' ')
    plot_fig(2, q1, P_in1, q2, P_in2, q3, P_in3, 0.0, 0.0, 'Flow Rate (L/s)', 'Power In (W)', 'Part2_Flow-vs-Power-In.png', '1400 RPM', '1300 RPM', '1200 RPM', ' ')
    plot_fig(3, q1, eta1, q2, eta2, q3, eta3, 0.0, 0.0, 'Flow Rate (L/s)', 'Efficiency', 'Part2_Flow-vs-Efficiency.png', '1400 RPM', '1300 RPM', '1200 RPM', ' ')

    input()

#problem1_execution()
problem2_execution()