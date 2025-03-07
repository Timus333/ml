#ml
1) import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\sumit\\Downloads\\sales.csv")
df.head()
plt.figure(figsize=(8, 6))
plt.bar(df.Region,df.Sale,color=['r','g','b','y'], edgecolor='black')
plt.title('Simple Bar Chart Example', fontsize=16)
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.show()


2) import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:\\Users\\sumit\\Downloads\\sales.csv")
CF = df['Customer_Feedback'].value_counts()
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=CF.index, y=CF.values, hue=CF.index, palette='viridis', dodge=False)
leg = ax.get_legend()
if leg:
    leg.remove()
plt.scatter(df['Customer_Feedback'], df['Rating'], color='red', label='Avg Satisfaction Score', zorder=5)
plt.title('Customer Feedback Categories', fontsize=16)
plt.xlabel('Feedback Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()


4)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\sumit\
x = np.arange(len(data['Category']))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, data['Provider A'], width, label='Provider A', color='skyblue', edgecolor='black')
ax.bar(x + width/2, data['Provider B'], width, label='Provider B', color='salmon', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(data['Category'])
ax.set_xlabel('Satisfaction Categories')
ax.set_ylabel('Satisfaction Score (out of 10)')
ax.set_title('Customer Satisfaction Scores by Service Provider')
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


5)import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\sumit\Downloads\daily_customers.csv")
plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Day', y='Customers', data=df, hue='Day', palette='viridis', dodge=False)
legend = ax.get_legend()
if legend is not None:
    legend.remove()
plt.title("Daily Customer Visits")
plt.xlabel("Day")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.show()


6)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\sumit\Downloads\scatter_data.csv")
model = LinearRegression().fit(df[['X1', 'X2']], df['Y'])
x1, x2 = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 5, 20))
y = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X1'], df['X2'], df['Y'], color='red')
ax.plot_surface(x1, x2, y, alpha=0.5)
ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('Y')
plt.title("3D Scatter with Prediction Surface")
plt.show()


7)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
student_data = pd.read_csv(r"C:\Users\sumit\Downloads\student_performance.csv")
plt.figure(figsize=(6, 5))
sns.heatmap(student_data.corr(), annot=True, cmap='coolwarm',
linewidths=0.5)
plt.title("Correlation Matrix of Student Performance")
plt.show()


8)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
df_heatmap = pd.read_csv(r"C:\Users\sumit\Downloads\heatmap_data.csv")
G = nx.erdos_renyi_graph(10, 0.3)
plt.figure(figsize=(6, 6))
nx.draw(G, with_labels=True, node_color='lightblue',
edge_color='gray')
plt.title("Network Graph")
plt.show()
plt.figure(figsize=(6, 5))
sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu')
plt.title("Heat Map")
plt.show()
  

9)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_high_dim = pd.read_csv(r"C:\Users\sumit\Downloads\high_dim_scatter.csv")
sns.pairplot(data_high_dim)
plt.show()


10)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
data_cluster = pd.read_csv(r"C:\Users\sumit\Downloads\cluster_data.csv")
linked = linkage(data_cluster, 'ward')
plt.figure(figsize=(8, 5))
dendrogram(linked, labels=list(range(1, 11)), orientation='top')
plt.title("Dendrogram for Cluster Analysis")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
