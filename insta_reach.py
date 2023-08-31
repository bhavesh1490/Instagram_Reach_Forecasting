#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
import os 
for dirname, _, filenames in os.walk(''):
 for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


instadt=pd.read_csv('Instagram-Reach.csv', encoding = 'latin1')


# In[3]:


instadt.head()


# In[4]:


instadt.tail()


# In[5]:


instadt.info()


# In[6]:


instadt.describe()


# In[7]:


instadt.isnull().sum()


# In[8]:


instadt=instadt.dropna()


# In[9]:


instadt.info()


# In[10]:


instadt.nunique()


# In[11]:


instadt.describe().transpose()


# In[12]:


instadt['Date'] = pd.to_datetime(instadt['Date'])
instadt['Date']


# In[13]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=instadt['Date'],
                        y=instadt['Instagram reach'],
                        mode='lines', name='Instagram reach'))
fig.update_layout(title='Instagram Reach Trend', xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()


# In[14]:


fig = go.Figure()
fig.add_trace(go.Bar(x=instadt['Date'], 
                     y=instadt['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach by Day', 
                  xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()


# In[15]:


fig = go.Figure()
fig.add_trace(go.Box(y=instadt['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach Box Plot', 
                  yaxis_title='Instagram Reach')
fig.show()


# In[16]:


instadt['Day'] = instadt['Date'].dt.day_name()
instadt.head()


# In[17]:


import numpy as np

day_stats = instadt.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
day_stats.head()


# In[18]:


fig = go.Figure()
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['mean'], 
                     name='Mean'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['median'], 
                     name='Median'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['std'], 
                     name='Standard Deviation'))
fig.update_layout(title='Instagram Reach by Day of the Week', 
                  xaxis_title='Day', 
                  yaxis_title='Instagram Reach')
fig.show()


# In[19]:


from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = instadt[["Date", "Instagram reach"]]

result = seasonal_decompose(data['Instagram reach'], 
                            model='multiplicative', 
                            period=100)

fig = plt.figure()
fig = result.plot()

fig = mpl_to_plotly(fig)
fig.show()


# In[20]:


pd.plotting.autocorrelation_plot(data["Instagram reach"])


# In[21]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(instadt["Instagram reach"], lags = 100, method='ywm')


# In[22]:


p, d, q = 8, 1, 2

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())


# In[23]:


predictions = model.predict(len(data), len(data)+100)
trace_train = go.Scatter(x=data.index,
                         y=data["Instagram reach"],
                         mode="lines",
                         name="Training Data")
trace_pred = go.Scatter(x=predictions.index,
                        y=predictions,
                        mode="lines",
                        name="Predictions")
layout = go.Layout(title="Instagram Reach Time Series and Predictions", 
                   xaxis_title="Date", 
                   yaxis_title="Instagram Reach")

fig = go.Figure(data=[trace_train, trace_pred], layout=layout)
fig.show()

