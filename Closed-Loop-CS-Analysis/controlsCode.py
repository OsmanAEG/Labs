#Importing Numpy, Matplotlib, Math, and Pandas
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 

figure_counter = 0

#Extracting and formatting the data 
def extract_data(extension):
    path = 'Controls Lab Data/WED1 ' + extension + '.txt'
    n, m = len(open(path).readlines()), 4
    raw_data = np.empty((n - 3, m), dtype=np.dtype('U100'))
    n, m = 0, 0

    with open(path, 'r') as file:
        for line in file:
            m = 0
            if n > 1 and n < len(open(path).readlines()):
                for word in line.split():
                    if word != '[': 
                        raw_data[n - 3, m] = word
                        m += 1
            n += 1

    raw_data = np.char.strip(raw_data, ';')
    raw_data = np.char.strip(raw_data, ']')
    raw_data = raw_data.astype(np.float)
    return raw_data

def plot_freqData(fig_number, set_name, x, y):
    fig = plt.figure(fig_number)
    plt.semilogx(x, y)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Position (dB)')
    plt.savefig(set_name + '.png')
    fig.show()

def plot_regData(fig_number, set_name, x, y1, y2):
    fig = plt.figure(fig_number)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['Command Position', 'Encoder Position'])
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.savefig(set_name + '.png')
    fig.show()
                
#Processing the required datasets
def processing_freqDatasets():
    global figure_counter
    freq_names = np.array(['frequency response critically damped', 
                          'frequency response over damped',
                          'frequency response under damped'])

    for i in range(np.size(freq_names)):
        data = extract_data(freq_names[i])

        time, posCom, posEnc = data[:, 1], data[:, 2], data[:, 3]

        freq = 10**((1.0 - time/60.0)*math.log10(0.1) + (time/60.0)*math.log10(20.0))

        posDB = np.zeros(np.size(posEnc))
        posEnc = np.abs(posEnc)
        for j in range(np.size(posEnc)):
            if posEnc[j] > 1.0:
                posDB[j] = 20.0*math.log10(posEnc[j])
            else:
                posDB[j] = 20.0*math.log10(0.1)

        figure_counter += 1
        plot_freqData(figure_counter, freq_names[i], freq, posDB)
    
def processing_regDatasets():
    global figure_counter
    set_names = np.array(['P response Kp=2X',
                          'P response Kp=X',
                          'PD response critically damped',
                          'PD response over damped',
                          'PD response under damped',
                          'PID response Ki=2X',
                          'PID response Ki=X'])

    for i in range(np.size(set_names)):
        data = extract_data(set_names[i])
        time, posCom, posEnc = data[:, 1], data[:, 2], data[:, 3]
        figure_counter += 1
        plot_regData(figure_counter, set_names[i], time, posCom, posEnc)

processing_freqDatasets()
processing_regDatasets()
input()



