import pandas as pd
import argparse
import json
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os

def compute_eer(label, pred, name, save_dir=None, plot_id=0, logger=None):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred)
    fnr = 1 - tpr
    if logger:
        scatter2d = np.hstack(
        (np.atleast_2d(fnr).T, np.expand_dims(fpr, axis=1))
        )
            
        # report 2d scatter plot with lines
        logger.report_scatter2d(
            title=name,
            series="series_xy",
            iteration=plot_id,
            scatter=scatter2d,
            xaxis="False Negative Rate",
            yaxis="False Positive Rate",
        )
    if save_dir:

        plt.figure(figsize=(10, 10))
        plt.title(f"fn-fp curve for {name}")
        plt.plot(fnr, fpr, label=f"fn-fp curve for {name}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.title(f"fn-fp curve for plot {plot_id}")
        plt.savefig(os.path.join(save_dir, f"{name}_fn_fp_curve_{plot_id}.png"))
        plt.close()

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer