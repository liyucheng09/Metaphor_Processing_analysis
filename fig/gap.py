import matplotlib.pyplot as plt
import numpy as np

literal = [0.776, 0.796, 0.571, 0.591, 0.653, 0.693,]
meta = [0.594, 0.624, 0.551, 0.551, 0.591, 0.611,]
x1 = [1.5, 2.5, 4.5, 5.5, 7.5, 8.5,]

hmc = [0.55, 0.63]
vua = [0.801, 0.769]
x2 = [10.5, 11.5]

task = ['mt', 'qa', 'nli',]
models = [ 'Google', 'DeepL', 'RoBERTa', 'T5', 'RoBERTa', 'T5', 'Precision', 'Recall']


if __name__ == '__main__':

    plt.figure(figsize=(10, 3.5), dpi=120)

    plt.plot(x1, literal, 'x', color = 'red', markerfacecolor='none', label = 'Literal')
    plt.plot(x2, vua, '^', color = 'red', markerfacecolor='none', label = 'VUA')

    plt.plot(x1, meta, 's', color = 'blue', markerfacecolor='none')
    # plt.plot(x1, meta, 'o', color = 'blue', markerfacecolor='none', label = 'Metaphor')
    plt.plot(x2, hmc, 's', color = 'blue', markerfacecolor='none', label = 'HMC')

    plt.yticks(np.linspace(0.5,0.95,5))
    plt.xticks(x1+x2, models)

    plt.grid(axis='y')
    plt.xlim([0, 13])
    plt.legend()
    plt.show()
