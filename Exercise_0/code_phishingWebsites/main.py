import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.io import arff

data, meta = arff.loadarff('PhishingData.arff')
df = pd.DataFrame(data)

df = df.apply(pd.to_numeric, errors='ignore')

target_mapping = {-1: 'Legitimate', 0: 'Suspicious', 1: 'Phishing'}
df['Result_Label'] = df['Result'].map(target_mapping)

feature_cols = [col for col in df.columns if col not in ['Result', 'Result_Label']]

fig1 = plt.figure(figsize=(10, 8))
correlation_matrix = df[feature_cols].corr()

feature_name_mapping = {
    'SFH': 'SFH',
    'popUpWidnow': 'PopUp',
    'SSLfinal_State': 'SSL',
    'Request_URL': 'ReqURL',
    'URL_of_Anchor': 'Anchor',
    'web_traffic': 'Traffic',
    'URL_Length': 'URLLen',
    'age_of_domain': 'Age',
    'having_IP_Address': 'HasIP'
}
corr_renamed = correlation_matrix.rename(columns=feature_name_mapping, index=feature_name_mapping)

sns.heatmap(corr_renamed, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 8}, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('phishing_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(14, 5))

ax1 = plt.subplot(1, 2, 1)
target_counts = df['Result_Label'].value_counts()
ordered_labels = ['Legitimate', 'Suspicious', 'Phishing']
target_counts = target_counts.reindex(ordered_labels)
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax1.bar(range(len(target_counts)), target_counts.values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
ax1.set_xticks(range(len(target_counts)))
ax1.set_xticklabels(target_counts.index, fontsize=10)
ax1.set_xlabel('Website Type', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
ax1.set_title('Target Variable Distribution', fontsize=12, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2 = plt.subplot(1, 2, 2)

binary_features = ['having_IP_Address', 'SSLfinal_State', 'popUpWidnow', 'SFH']

feature_percentages = []
for feature in binary_features:
    pct_by_class = df.groupby('Result_Label')[feature].apply(
        lambda x: (x == 1).sum() / len(x) * 100 if feature in ['having_IP_Address', 'popUpWidnow']
        else (x == -1).sum() / len(x) * 100
    )
    feature_percentages.append(pct_by_class)

pct_df = pd.DataFrame(feature_percentages, index=binary_features).T
pct_df = pct_df.reindex(['Legitimate', 'Suspicious', 'Phishing'])

x = np.arange(len(pct_df.index))
width = 0.2
colors_features = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']

for i, feature in enumerate(binary_features):
    offset = (i - len(binary_features)/2 + 0.5) * width
    bars = ax2.bar(x + offset, pct_df[feature], width,
                   label=feature.replace('_', ' '),
                   color=colors_features[i], alpha=0.8, edgecolor='black')

ax2.set_xlabel('Website Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('Risky Feature Indicators by Website Type', fontsize=12, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(pct_df.index, fontsize=10)
ax2.legend(loc='upper left', fontsize=8, ncol=2)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('phishing_dataset_2plots.png', dpi=300, bbox_inches='tight')
plt.show()