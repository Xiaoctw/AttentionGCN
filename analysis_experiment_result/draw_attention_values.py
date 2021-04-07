import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
dataset='BlogCatalog'
file_path=Path(__file__).parent/('attention_value_'+dataset+'.txt')

f=open(file_path)
line1=f.readline()
adj_values,combine_values,ripple_values=[],[],[]
list1=line1.strip().split(':')[1].split(',')
for val in list1:
    adj_values.append(float(val))
line1 = f.readline()
list1=line1.strip().split(':')[1].split(',')
for val in list1:
    combine_values.append(float(val))

line1 = f.readline()
list1=line1.strip().split(':')[1].split(',')
for val in list1:
    ripple_values.append(float(val))

adj_values=np.array(adj_values)
combine_values=np.array(combine_values)
ripple_values=np.array(ripple_values)
print(adj_values.size)
id=np.arange(0,401,400/len(adj_values)+1)
print(id.size)
df=pd.DataFrame({'id':id,'adj':adj_values,'combine':combine_values,'ripple':ripple_values})

if __name__ == '__main__':
    plt.title(dataset)
    plt.xlabel('epochs')
    plt.ylabel('attention value')
    plt.ylim(0,0.8)
    plt.plot(id,adj_values,label='adj',linewidth=3)
    plt.plot(id,combine_values,label='combine',linewidth=3)
    plt.plot(id,ripple_values,label='ripple',linewidth=3)
    plt.legend(('adjacent', 'combine', 'ripple'), loc='upper left')
    plt.savefig(dataset+'.png')
    plt.show()





