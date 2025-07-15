#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go
import warnings

#choose the plots palette
sns.set_theme(
    style="whitegrid",
    palette="dark:steelblue_r")
pio.templates.default = "plotly_dark"


warnings.simplefilter(action='ignore', category=FutureWarning)
from data_clean_prep import return_df


# In[2]:


df = return_df()


# In[40]:


# Data frame info
print('\n📌Data fram info : \n')
display(df.info())
print()


# In[41]:


# Describe of numerical columns
print('\n📌Describe of numerical columns : \n')
display(df.describe().T)
print()


# In[42]:


# Describe of catgorical columns
print('\n📌Describe of catgorical columns : \n')
display(df.describe(include = "O").T)
print()


# # EDA

# ## Univariate Analysis

# In[3]:


# Vehicles identity	Analysis
fig_columns = ['brand', 'model', 'trim', 'body']

# Number of rows = number of columns, each row has one graph: countplot
fig, axes = plt.subplots(len(fig_columns), 1, figsize=(25, 30))
fig.suptitle('Vehicles Identity	Analysis', fontsize=22)

for i, col in enumerate(fig_columns):
    top_10 = df[col].value_counts().nlargest(15).index

    # countplot
    sns.countplot(x=col, data=df[df[col].isin(top_10)], ax=axes[i], order=top_10)
    axes[i].set_title(f'Countplot of {col} Top 15', fontsize=14, pad=40)
    axes[i].tick_params(axis='x', rotation=45)



plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# In[4]:


fig_columns = ['state','seller']

# Number of rows = number of columns, each row has two graphs (countplot + pie)
fig, axes = plt.subplots(len(fig_columns), 1, figsize=(25, 20))
fig.suptitle('Categorical Univariate Analysis', fontsize=22)

for i, col in enumerate(fig_columns):
    top_10 = df[col].value_counts().nlargest(10).index

    # countplot
    sns.countplot(x=col, data=df[df[col].isin(top_10)], ax=axes[i], order=top_10)
    axes[i].set_title(f'Countplot of {col} Top 10', fontsize=14, pad=40)
    axes[i].tick_params(axis='x', rotation=45)



plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# In[5]:


# Vehicle Attributes Analysis
fig_columns = ['model_year','color', 'interior','transmission']

# Number of rows = number of columns, each row has two graphs (countplot + pie)
fig, axes = plt.subplots(len(fig_columns), 2, figsize=(30, 40))
fig.suptitle('Vehicle Attributes Analysis', fontsize=22)

for i, col in enumerate(fig_columns):
    all_values = df[col].value_counts().index  

    # countplot
    sns.countplot(x=col, data=df, ax=axes[i, 0], order=all_values)
    axes[i, 0].set_title(f'Countplot of {col}', fontsize=14, pad=40)
    axes[i, 0].tick_params(axis='x', rotation=45)

    # pie chart
    axes[i, 1].pie(
        df[col].value_counts().values,
        labels=all_values,
        autopct='%1.1f%%',
        startangle=140,
        radius=1.4
    )
    axes[i, 1].set_title(f'Pie Chart of {col}', fontsize=14, pad=70)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[6]:


# Sales times analysis
fig_columns = ['sell_day_name','sell_month_name','sell_year','season', 'time_period']

# Number of rows = number of columns, each row has two graphs (countplot + pie)
fig, axes = plt.subplots(len(fig_columns), 2, figsize=(30, 40))
fig.suptitle('Sales Times Analysis', fontsize=22)

for i, col in enumerate(fig_columns):
    all_values = df[col].value_counts().index  

    # countplot
    sns.countplot(x=col, data=df, ax=axes[i, 0], order=all_values)
    axes[i, 0].set_title(f'Countplot of {col}', fontsize=14, pad=40)
    axes[i, 0].tick_params(axis='x', rotation=45)

    # pie chart
    axes[i, 1].pie(
        df[col].value_counts().values,
        labels=all_values,
        autopct='%1.1f%%',
        startangle=140,
        radius=1.4
    )
    axes[i, 1].set_title(f'Pie Chart of {col}', fontsize=14, pad=70)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[7]:


# Numerical Analysis
fig_columns = ['mmr','sellingprice','market_advantage','motor_mi']
fig, axes = plt.subplots(2, 2, figsize=(15, 20))  
fig.suptitle('Plots for Numerical Univariate Analysis', fontsize=22)

for i, col in enumerate(fig_columns):
    row = i // 2
    col_idx = i % 2
    ax = axes[row, col_idx]
    
    
    sns.histplot(x=col, data=df, ax=ax,kde=True) 
    
    ax.set_title(f'Histogram of {col}', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ## Bivariate Analysis

# In[8]:


# Check numerical correlation
print('\n🔗Numerical Correlation\n\n')
print(df.corr(numeric_only=True))


# In[9]:


# numerical correlation Plot
plt.figure(figsize=(8, 5)) # figer size 
sns.heatmap(data = df.corr(numeric_only=True))
plt.title('🔗Numerical Correlation')
plt.tight_layout()
plt.show()


# In[10]:


#1. Is there a direct or inverse relationship between motor_mi and sellingprice?
# Yes there is a inverse relationship
plt.figure(figsize=(15, 5)) # figer size 
sns.scatterplot(data=df, x='motor_mi', y='sellingprice') # scatter plot between motor_mi & sellingprice 
# add title and labels
plt.title('Scatter Plot: motor_mi vs Selling Price')
plt.xlabel("Motor Mi")
plt.ylabel('Selling Price')
# show fig
plt.grid(True) 
plt.tight_layout()
plt.show()


# In[11]:


#2. Are cars with a price lower than MMR in poor condition?
#Yes, cars that were sold at a price lower than their MMR tend to be in poorer condition on average.
sns.boxplot(x=df['market_advantage'] < 0, y='condition', data=df)
plt.xticks([0, 1], ['Above/Equal MMR', 'Below MMR'])
plt.title('Condition vs Market Advantage')
plt.show()

below_mmr = df[df['market_advantage'] < 0]
above_mmr = df[df['market_advantage'] >= 0]
print()
print("Average condition for cars below MMR:", below_mmr['condition'].mean())
print("Average condition for cars above MMR:", above_mmr['condition'].mean())
print()


# In[12]:


#3. Do newer cars (model_year) actually sell for a higher price?
#Yes, there is a positive relationship between model_year and selling price. On average, newer cars tend to sell for a higher price.

plt.figure(figsize=(12, 6))
sns.lineplot(x='model_year', y='sellingprice', data=df)
plt.title('Selling Price by Model Year')
plt.xticks(rotation=45)
plt.show()
print()
correlation = df['model_year'].corr(df['sellingprice'])
print(f"Correlation between model_year and sellingprice: {correlation}")
print()


# In[13]:


# 4. Are there clear differences between the selling price in different seasons?
# There is not much difference in prices with the different seasons.
season_avg_sell = df.groupby('season')['sellingprice'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='sellingprice', data=season_avg_sell)
plt.title('Average Selling Price by Season')
plt.xlabel('Season')
plt.ylabel('Selling Price')
plt.show()



# In[14]:


# 4. Are there clear differences between the Ravenu in different seasons?
# There is a clear difference. Ravenu is big in the winter.

season_ravenu = df.groupby('season')['sellingprice'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='sellingprice', data=season_ravenu)
plt.title('Ravenu by Season')
plt.xlabel('Season')
plt.ylabel('Selling Price')
plt.show()


# In[15]:


# 5-Which brand has the highest average sales?
# The brand has the highest average price is Rolls-Royce
brand_avg_sell = df.groupby('brand')['sellingprice'].mean().sort_values(ascending=False).reset_index().head(20)


plt.figure(figsize=(10, 6))
sns.barplot(y='brand', x='sellingprice', data=brand_avg_sell)
plt.title('Average Selling Price by brand')
plt.xlabel('Brand')
plt.ylabel('Selling Price')
plt.show()


# In[16]:


# 6- Does the transmission type (manual or automatic) affect the price?
# Yes, transmission type does affect the price, On average automatic cars are sold at higher prices compared to manual cars.
transmission_avg = df.groupby("transmission")['sellingprice'].mean().sort_values(ascending = True).reset_index()
plt.figure(figsize = (10,6))
sns.barplot(data = transmission_avg , x = 'transmission' , y = 'sellingprice')
plt.title("Average Selling price by transmission")
plt.xlabel('Transmission')
plt.ylabel("Selling price")
plt.show()


# In[17]:


# Model year and Value color by Transmission
YT = pd.crosstab(df["model_year"],df["transmission"])
px.line(YT,title= "Distribution of Model year preferance color by Transmission all cars")


# In[18]:


# Statewise Preferance color by Transmission 
ST = pd.crosstab(df["state"],df["transmission"])
px.line(ST, title= "Statewise Transmission preferance")


# In[19]:


# 7- Are there some states (states) where selling prices are higher than others?
# Oregon & Tennessee state selling prices are higher than others
state_avg = df.groupby("state")['sellingprice'].mean().sort_values(ascending = True).reset_index()
plt.figure(figsize = (12,6))
sns.barplot(data = state_avg , x = 'state' , y = 'sellingprice')
plt.title("Average Selling price by States")
plt.xlabel('States')
plt.ylabel("Selling price")
plt.show()


# In[20]:


# 8- Does the condition of the car greatly affect the selling price?
#Yes, there is a strong direct relationship between a car's condition and its selling price. The better the condition of the car, the higher its market value.
plt.figure(figsize=(10, 6))
sns.scatterplot(data = df ,x='condition', y='sellingprice' )
plt.title('Selling Price by Condition')

plt.show()
print()
correlation = df['condition'].corr(df['sellingprice'])
print(f"Correlation between condition and sellingprice: {correlation}")
print()


# In[21]:


#9-Are older cars in worse condition?
#Yes, there is a strong direct relationship between a car's condition and its model_year. 

plt.figure(figsize=(10, 6))
sns.scatterplot(data = df ,y='condition', x='model_year' )
plt.title('Selling Price by Condition')
plt.show()
print()
correlation = df['condition'].corr(df['model_year'])
print(f"Correlation between condition and model_year: {correlation}")
print()


# In[22]:


# show corr between price and mmr
plt.figure(figsize=(12,6))
plt.title('Selling Price vs MMR')
sns.regplot(x=df['sellingprice'], y=df['mmr'], marker='o', color=".3", line_kws=dict(color="r"))
plt.xlabel('Selling Price', fontsize=16)
plt.ylabel('MMR', fontsize=16)


# In[23]:


#10 -Are there certain sellers who always sell at above market price (MMR)?
# pdx auto wholesale llc seller have 41.5K above mmr in total of all sales
sell_above_mmr = df.groupby('seller')['market_advantage'].mean().sort_values(ascending = False).reset_index()
plt.figure(figsize = (12,6))
sns.barplot(data = sell_above_mmr.head(10) , y = 'seller' , x = 'market_advantage')
plt.title("sellers who always sell at above market price (MMR)")
plt.xlabel('States')
plt.ylabel("Market Advantage")
plt.show()


# In[24]:


#12- Is there a specific interior or color associated with higher prices?
# Certain exterior and interior car colors are associated with higher average selling prices, often due to public taste or luxury.

color_avg = df.groupby('color')['sellingprice'].mean().sort_values(ascending=False).reset_index()
interior_avg = df.groupby('interior')['sellingprice'].mean().sort_values(ascending=False).reset_index()
# Color plot
plt.figure(figsize=(12, 6))
sns.barplot(data=color_avg.head(10), x='color', y='sellingprice')
plt.title('Average Selling Price by Exterior Color (Top 10)', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Average Selling Price')
plt.show()

# Interior plot
plt.figure(figsize=(12, 6))
sns.barplot(data=interior_avg.head(10), x='interior', y='sellingprice')
plt.title('Average Selling Price by Interior Color (Top 10)', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Average Selling Price')
plt.show()



# In[25]:


# Interior plot
plt.figure(figsize=(12, 6))
sns.barplot(data=interior_avg.head(10), x='interior', y='sellingprice')
plt.title('Average Selling Price by Interior Color (Top 10)', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Average Selling Price')
plt.show()


# In[26]:


print("\nHandling Data type of saledate column")
# Unify dates and convert them to UTC time
df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)

# I wanted to remove the time and make the dates Naive
df['saledate'] = df['saledate'].dt.tz_convert(None)

# Verify column type
if isinstance(df['saledate'].dtype, pd.DatetimeTZDtype):
    df['saledate'] = df['saledate'].dt.tz_localize(None)
print('\nsaledate column data type handling, Done ✅\n')


# In[27]:


#Average Selling Price by Month
df['sale_month'] = df['saledate'].dt.to_period('M').dt.to_timestamp()

plt.figure(figsize=(18, 6))
plt.title('Average Selling Price per Month', fontsize=20)
sns.lineplot(x='sale_month', y='sellingprice', data=df)
plt.xlabel('Sale Month', fontsize=16)
plt.ylabel('Average Selling Price', fontsize=16)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Multivariate Analysis

# In[28]:


#Top 20 Expensive Trims by Brand
trim_price_avg = df.groupby('trim')['sellingprice'].mean().sort_values(ascending=False).head(50)
brand_trim_price = df.groupby(['brand', 'trim'])['sellingprice'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(14, 6))
sns.barplot(data=brand_trim_price.head(20), x='brand', y='sellingprice', hue='trim')
plt.xticks(rotation=45)
plt.title('Top 20 Expensive Trims by Brand')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[29]:


# Top 20 Expensive Trims by Model
model_trim_price = df.groupby(['model', 'trim'])['sellingprice'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(14, 6))
sns.barplot(data=model_trim_price.head(20), x='model', y='sellingprice', hue='trim')
plt.xticks(rotation=45)
plt.title('Top 20 Expensive Trims by Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[30]:


#1- How does the condition and model year together influence the selling price?

condition_price_impact = df.groupby(['model_year', 'condition'])['sellingprice'].mean().reset_index()

fig = px.line(condition_price_impact,
              x='condition',
              y='sellingprice',
              color='model_year',
              title='Condition and Model Year Impact on Selling Price',
              markers=True,height=800,width=1100)
fig.update_layout(xaxis_title='Condition', yaxis_title='Average Selling Price', legend_title='Model Year')
fig.show()


# In[31]:


#Is there a relationship between model, market_advantage, and the final sellingprice?

model_impact = df.groupby('model')[['market_advantage', 'sellingprice']].mean().reset_index()
top_models = df['model'].value_counts().nlargest(30).index
model_impact_top = model_impact[model_impact['model'].isin(top_models)]
import plotly.express as px

fig = px.scatter(model_impact_top,
                 x='market_advantage',
                 y='sellingprice',
                 text='model',
                 size='sellingprice',
                 color='model',
                 title='Market Advantage vs Selling Price by Model',
                 width=1000,
                 height=600)
fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)
fig.show()


# In[32]:


#Is there a relationship between model, market_advantage, and the final sellingprice?

model_impact = df.groupby('model')[['market_advantage', 'sellingprice']].mean().reset_index()

top_models = model_impact[model_impact['market_advantage'] > 0].sort_values('market_advantage', ascending=False).head(20)

import plotly.express as px

fig = px.scatter(top_models,
                 x='market_advantage',
                 y='sellingprice',
                 text='model',
                 size='sellingprice',
                 color='model',
                 title='Top 30 Models with Positive Market Advantage',
                 width=1000,
                 height=600)

fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)
fig.show()


# In[33]:


# How condition and Brand performace
top_makes = df['brand'].value_counts().nlargest(10).index

filtered_data = df[df['brand'].isin(top_makes)]

fig = px.box(filtered_data, x='brand', y='sellingprice',
             title="Selling Price Distribution by Brand for Top Brands",
             category_orders={"brand": top_makes.tolist()})
fig.update_layout(yaxis_title="Selling Price", xaxis_title="Brand", width=1100, height=600)
fig.show()


# In[34]:


# Condition Impact on Price within Top Brands
top_makes = ['Nissan', 'Ford', 'Chevrolet', 'Toyota', 'BMW','Lexus']

condition_price_impact = df[df['brand'].isin(top_makes)].groupby(['brand', 'condition'])['sellingprice'].mean().unstack()

fig = px.line(condition_price_impact.T, title='Condition Impact on Price within Top Brands', width=1100, height=400)
fig.update_layout(xaxis_title='Condition', yaxis_title='Average Selling Price', legend_title='Brand')
fig.show()


# In[35]:


# show top Brands by Total Revenue
model_revenue = (
    df.groupby(['brand', 'model','trim'])['sellingprice']
    .sum()
    .reset_index()
    .sort_values(by='sellingprice', ascending=False)
)

fig = px.treemap(
    model_revenue.head(80),  # Limit to top 80 for clarity
    path=['brand', 'model','trim'],
    values='sellingprice',
    title='Top Vehicle Models by Total Revenue'
)
fig.show(renderer='iframe')

We have some cars manufactured in 1982 this might result as having some Vintage Cars in the dataset.
Ford, Chevrolet, Nissan, Toyota, and Dodge are the top 5 brands by the count.
Sedan and SUV are the top 2 body types used in cars.
The selling price of automatic transmission car brands is highest besides of two brands Land Rover and Lotus. Lotus does not have car of automatic transmission while Rolls-Royce does not have car of manual transmission.
Our scatterplot shows that MMR and Selling Price is very much correlated but it also showing there are some of data points which are very much off in respect of other.
The distance travelled by car and selling price does have a negative relation but it is not vey much decisive in case of Selling price.
Cars manufactured after year 2000 shows and upward trend in respect of Selling price. While there is no trend between sale date and selling price, it just kind of random walk.
here is a positive relationship between model_year and selling price. On average, newer cars tend to sell for a higher price.
There is not much difference in prices with the different seasons.
There is a clear difference. Ravenu is big in the winter.
transmission type does affect the price, On average automatic cars are sold at higher prices compared to manual cars.
pdx auto wholesale llc seller have 41.5K above mmr in total of all sales



Market Trends:
About a third of sales are vehicles that are 4 years or younger, indicating that newer vehicles (4 years or younger) dominate the used market.
The most sold and profitable cars are the ones with max 50K miles: the quota of the sold cars quota is 49.75%, and the quota of the total USD gained is 67.82%
Customer Preferences:
Most Popular Makes and Models: Ford is the top-selling brand with 4 models in the top 6. The top-selling model is the Ford Fusion.
Vehicle Price and Condition: The correlation between vehicle condition and sales price is low, indicating that condition assessment does not have much impact on the final price.
Pricing Strategy:
The sold cars that are have a model age less than 3 years generate 48% of total amount of USD income.
The most profitable states for car sales are Tennessee, Colorado and Illinois, where median prices exceed the global average of $12,400.
Vehicle Depreciation:
Vehicles between 0 and 3 years old show less loss of value than older vehicles.
Some brands, such as BMW and Mercedes-Benz, tend to retain their value better, while other brands show higher rates of depreciation.