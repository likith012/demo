import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv", header=None).T
t = np.arange(len(df))*30/3600
predict = df.to_numpy()
predicted = predict.squeeze()
# print (t.shape)
# print(t)
# print (predicted.shape)
# print (predicted)
fig, ax = plt.subplots(figsize=(12, 3))
# ax.plot(t, y_true[mask], label='True')
ax.plot(t, predicted, alpha=0.7)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'R'])
ax.set_xlabel('Time (h)')
ax.set_title('Hypnogram')
ax.legend()
plt.savefig('static/Hypnogram.png', dpi=600, bbox_inches='tight', papertype="a4")
# plt.show()