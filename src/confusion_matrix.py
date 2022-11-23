import os
import torch
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def createConfusionMatrix(loader, net, device):
    y_pred = []  # save predction
    y_true = []  # save ground truth

    # iterate over data
    for _, inputs, labels, _ in loader:

        inputs = inputs.to(device)
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.to(device)
        output = output.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.to(device)
        labels = labels.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    audio_dir = "../data/audio_files"
    classes = tuple([name for name in os.listdir(
        audio_dir) if os.path.isdir(os.path.join(audio_dir, name))])

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
        
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))

    return sn.heatmap(df_cm, annot=True).get_figure()
