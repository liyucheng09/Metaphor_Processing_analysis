import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ['NLI', 'Translation', 'WSD', 'Sentiment', 'Preposition']
meta = [0.872, 0.922, 0.75, 0.927, 0.660]
literal = [0.888, 0.930, 0.78, 0.902, 0.679]

# Create a bar plot
x = np.arange(len(tasks))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 2.5), dpi=180)

# Plot bars
rects1 = ax.bar(x - width/2, meta, width-0.05, label='Metaphor', alpha = 0.7, color = 'lavender')
rects2 = ax.bar(x + width/2, literal, width-0.05, label='Literal', alpha = 0.7, color = 'lightpink')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by task and type')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize = 10)
ax.legend(fontsize=8, )

# Set y-axis limits and ticks
ax.set_ylim([0.5, 1.0])
ax.set_yticks(np.arange(0.5, 1.01, 0.1))

# Label with specially formatted floats
ax.bar_label(rects1, fmt='%.2f', label_type='edge', fontsize=8)
ax.bar_label(rects2, fmt='%.2f', label_type='edge', fontsize=8)

plt.show()
