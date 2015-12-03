__author__ = 'tombocklisch'

from matplotlib import pyplot, cm
from pylab import *


def plot_examples(dataset, plot_ids):
    def label_of_example(i):
        return "'" + dataset.human_label_for(dataset.y_train[i]) + "'"

    fig = figure(figsize=(20,10))
    ax1 = subplot(311)
    setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('X - Acceleration')
    
    ax2 = subplot(312, sharex=ax1)
    setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('Y - Acceleration')
    
    ax3 = subplot(313, sharex=ax1)
    ax3.set_ylabel('Z - Acceleration')
    
    for i in plot_ids:
        c = np.random.random((3,))
    
        ax1.plot(range(0, dataset.num_features / 3), dataset.X_train[i,0:1200:3], '-o', c=c)
        ax2.plot(range(0, dataset.num_features / 3), dataset.X_train[i,1:1200:3], '-o', c=c)
        ax3.plot(range(0, dataset.num_features / 3), dataset.X_train[i,2:1200:3], '-o', c=c)
    
    legend(map(label_of_example, plot_ids))
    suptitle('Feature values for the first three training examples', fontsize=16)
    xlabel('Time')
    return fig
