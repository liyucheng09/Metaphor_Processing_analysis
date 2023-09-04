import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ['bert-base', 'bert-large', 'roberta-base', 'roberta-large']
vua = [0.64, 0.656, 0.674, 0.678]
novel = [0.58, 0.59, 0.61, 0.622]
ours = [0.56, 0.555, 0.58, 0.55]

# Create the plot
plt.figure(figsize=(6, 4), dpi=200)

# Plot the lines
plt.plot(tasks, vua, marker='o', label='VUA', color='violet')
plt.plot(tasks, novel, marker='s', label='Do Dinh', color='grey')
plt.plot(tasks, ours, marker='^', label='Ours', color='lime')

# Add titles and labels
plt.xlabel('Models')
plt.ylabel('Recall')
# plt.title('Performance Metrics by Task and Method')
# plt.ylim([0.45, 0.72])
# plt.yticks(np.arange(0.5, 0.72, 0.05))

# Add a legend
plt.legend( loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=10,)

# Show the plot
plt.show()
