from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')


# make data
def showLabelDistribution(clientList):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar

    for i in range(1,11,1):
        IT = [client[i] for client in clientList]
        br1 = np.arange(len(IT))
        plt.bar(br1, IT, color='b', width=barWidth, edgecolor='grey', label=('label ' + str(i)))
    # Adding Xticks
    plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('Students passed', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(clientList))], ["client " + str(r) for r in range(len(clientList))])

    plt.legend()
    plt.show()
