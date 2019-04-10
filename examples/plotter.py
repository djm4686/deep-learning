from matplotlib import pyplot as plt


def plot_data_list(plots):
    for i, plot in enumerate(plots):
        plt.subplot(2, 1, i)
        plt.plot(plot[0], plot[1], 'b', label=plot[2])
    plt.show()
