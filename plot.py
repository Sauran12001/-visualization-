#library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
draw_scatter()
#%%
df = pd.read_clipboard(sep=',')

label_encoder = LabelEncoder()
df['z_encoded'] = label_encoder.fit_transform(df['z'])

model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(df[['x', 'y']], df['z_encoded'])

x_min, x_max = df['x'].min() - 10, df['x'].max() + 10
y_min, y_max = df['y'].min() - 10, df['y'].max() + 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x', y='y', hue='z', palette="Set1", edgecolor='black')
plt.contourf(xx, yy, Z, alpha=0.3, cmap="Set1")
plt.title("PLOT")
plt.show()
