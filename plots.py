import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('brainsize.txt', sep='\t', na_values='NA')
features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
men = df[df['Gender'] == 'Male']
women = df[df['Gender'] == 'Female']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, f in enumerate(features):
    axes[0, i].scatter(men[f], men['MRI_Count'], alpha=0.6, color='blue')
    axes[0, i].set_title(f'Male: {f}')
    axes[0, i].set_xlabel(f)
    axes[0, i].set_ylabel('MRI_Count')
    
    # Женщины — нижний ряд
    axes[1, i].scatter(women[f], women['MRI_Count'], alpha=0.6, color='red')
    axes[1, i].set_title(f'Female: {f}')
    axes[1, i].set_xlabel(f)
    axes[1, i].set_ylabel('MRI_Count')

plt.tight_layout()
plt.savefig('scatter.png', dpi=150)   
plt.show()                            

# 3. Heatmap корреляций
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

corr_men = men[features + ['MRI_Count']].corr()
corr_women = women[features + ['MRI_Count']].corr()

sns.heatmap(corr_men, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Male')

sns.heatmap(corr_women, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Female')

plt.tight_layout()
plt.savefig('heatmap.png', dpi=150)
plt.show()