### PLOTTING MAPS AND LOSSES ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels_csv = ['loss','classification','bbox_regression','bbox_ctrness','map','map_small','map_medium','map_large','map_50','map_75']

for n in [1,2]:
    data_bl = pd.read_csv(f'model-fcos_resnet50_fpn-DAWN-baseline-fp-weather-exclude-{n}.pth.csv')
    data_adv = pd.read_csv(f'model-fcos_resnet50_fpn-DAWN-adversarial-fp-notransfer-weather-exclude-{n}.pth.csv')

    for l in labels_csv:
        epochs = []
        bl_data = []
        adv_data = []

        for i in range(len(data_bl['epochs'])):
            epochs.append(data_bl['epochs'][i])
            bl_data.append(data_bl[f'{l}'][i])
            adv_data.append(data_adv[f'{l}'][i])

        fig, ax = plt.subplots(figsize=(10,5))

        plt.plot(epochs, bl_data, label="baseline")
        plt.plot(epochs, adv_data, label="adversarial")
        plt.legend()
        plt.title(f'{l}')
        plt.show()
        plt.close(fig)
        fig.savefig(f'plots/DAWN-fp-onlyrtransfer-exclude-{n}-{l}.png',bbox_inches='tight', dpi=150)
