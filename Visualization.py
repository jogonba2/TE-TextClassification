import matplotlib.pyplot as plt
import numpy as np


class Visualization:

    @staticmethod
    def visualize_attentions(attns, fig_width,
                             fig_height, rows,
                             columns, ticks, output_file = False,
                             save = False):

        if len(attns.shape) <= 2:
            attns = np.expand_dims(attns, axis=0)
        fig = plt.figure(figsize=(fig_width, fig_height))
        for i in range(1, columns * rows + 1):
            ind_head = i - 1
            ax = fig.add_subplot(rows, columns, i)
            if len(attns.shape) == 2:
                plt.imshow(attns, interpolation="nearest")
            else:
                plt.imshow(attns[ind_head], interpolation='nearest')
            plt.xticks(np.arange(len(ticks)), [x+ " " for x in ticks], rotation='vertical', fontsize=46)
            plt.yticks(np.arange(len(ticks)), ticks, fontsize=46)


        if not save:
            #plt.colorbar()
            plt.show()
        else:
            assert output_file
            fig.savefig(output_file, format="pdf")
