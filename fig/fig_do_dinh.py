import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ['BERT-base', 'BERT-large', 'RoBERTa-base', 'RoBERTa-large']
vua = [0.64, 0.656, 0.674, 0.678]
novel = [0.58, 0.59, 0.61, 0.622]
ours = [0.56, 0.555, 0.58, 0.55]

# Create the plot
plt.figure(figsize=(6, 3.5), dpi=200)

# Plot the lines
plt.plot(tasks, vua, marker='o', label='VUA', color='violet')
plt.plot(tasks, novel, marker='s', label='Do Dinh et al. (2018)', color='grey')
plt.plot(tasks, ours, marker='^', label='Hard Metaphor (Ours)', color='lime')

# Add titles and labels
plt.xlabel('Models')
plt.ylabel('Recall')
# plt.title('Performance Metrics by Task and Method')
plt.ylim([0.52, 0.72])
plt.yticks(np.arange(0.52, 0.72, 0.03))

# Add a legend
plt.legend( loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=10,)

plt.tight_layout()
# Show the plot
plt.show()
