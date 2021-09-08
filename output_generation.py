import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

#%%
df = pd.DataFrame()

for filname in os.listdir('synthetic_data'):
    if('participation_sat.npy' in filname):
        data = np.load(f'synthetic_data/{filname}')
        # print(filname)
        # print(np.mean(data, axis=0))
        # all_data.append(data)
        
        nf = pd.DataFrame(data, columns = ['GP-Copeland','runtime'])
        n, m = filname.split('-')[:2]
        nf['setting'] = f'n={n}, m={m}'
        
        df = df.append(nf)

df['log(runtime)'] = 
fig, ax = plt.subplots()
sns.catplot(x="setting", y="runtime", data=df, ax = ax)

sns.plot.show()
# g.fig.suptitle('runtime for brute force search \nCopeland-lexicographic MPSR tiebreaking')

#%%