#!/usr/bin/env python
# coding: utf-8

# In[92]:


# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


sns.set_theme(
    style="whitegrid",
    palette="dark:steelblue_r")
warnings.simplefilter(action='ignore', category=FutureWarning)

# some information About Project Dataset


# In[125]:


# Reading parts
df1 = pd.read_csv('Data/part1.csv')
df2 = pd.read_csv('Data/part2.csv')
df3 = pd.read_csv('Data/part3.csv')
df4 = pd.read_csv('Data/part4.csv')

# Merge them into one DataFrame
df = pd.concat([df1, df2, df3, df4], ignore_index=True)



print(f"The data has been fully collected ✅. \nShape: {df.shape}\n")



# In[126]:


# First 10 rows 
print("\n🚩First 10 rows : \n")
display(df.head(10))


# In[127]:


# Last 10 rows 
print("\n🚩last 10 rows : \n")
display(df.tail(10))


# # Data understanding

# In[128]:


# Data frame info
print('\n📌Data fram info : \n')
display(df.info())
print()


# In[129]:


# Describe of numerical columns
print('\n📌Describe of numerical columns : \n')
display(df.describe().T)
print()


# In[130]:


# Describe of catgorical columns
print('\n📌Describe of catgorical columns : \n')
display(df.describe(include = "O").T)
print()


# In[131]:


# check the wrong values
df[df['vin'] == "automatic" ]


# In[132]:


df[df['body'] == 'Navitgation'].shape # Check wrong values


# In[133]:


# Check the unique values in catgorical columns  
print('\n📌Check the unique values in catgorical columns : \n')
cat_column = ['make', 'model', 'trim', 'body', 'transmission','color', 'interior', 'seller','state']
for col in cat_column :
    print(f'column : {col}')
    print(df[col].value_counts().head(30)) # preview top 10 unique values with count 
    print(f"\ncount of values :{len(df[col].unique())}") # preview the count of unique values
    print("="*50 + "\n")


# In[134]:


df[df['interior'] == "—"]


# In[135]:


df[df['color'] == '—']


# In[136]:


# 🚩there are wrong values in columns : 'color' , 'interior'⚠️
# 🚩There are wrong values in body column : sedan , suv ⚠️
# 🚩there are 26 rows have wrong values in some columns ⚠️


# # Data cleaning

# In[137]:


# check dublicates
print(f"\n📌 Number of duplicated rows:{df.duplicated().sum()}\n")


# In[138]:


# change the wrong values to np.nan
df.loc[df['interior'] == "—", 'interior'] = np.nan
df.loc[df['color'] == "—", "color"] = np.nan
print('\ninterior & color columns cleand ✅\n')


# In[139]:


# all wrong values in the trim = SE PZEV w/Connectivity
print("⚠️wrong values in the trim = SE PZEV w/Connectivity⚠️")
print(f"\nshape : {df[df['trim'] =='SE PZEV w/Connectivity'].shape}")
display(df[df['trim'] =='SE PZEV w/Connectivity'])
print()


# In[140]:


# Edit the wrong values in these rows.
# This arrangement is necessary because any arrangement will cause errors in the data.
mask = df['trim'] =='SE PZEV w/Connectivity'
df.loc[mask, 'mmr'] = df.loc[mask, 'sellingprice'].astype(float)


df.loc[mask, 'sellingprice'] = pd.to_numeric(df.loc[mask, 'saledate'], errors='coerce')


df.loc[mask, 'saledate'] = np.nan


df.loc[mask, 'condition'] = pd.to_numeric(df.loc[mask, 'odometer'], errors='coerce')
df.loc[mask, 'odometer'] = pd.to_numeric(df.loc[mask, 'color'], errors='coerce')


df.loc[mask, 'color'] = df.loc[mask, 'interior']
df.loc[mask, 'interior'] = df.loc[mask, 'seller']
df.loc[mask, 'seller'] = np.nan


df.loc[mask, 'body'] = df.loc[mask, 'transmission']
df.loc[mask, 'transmission'] = df.loc[mask, 'vin']
df.loc[mask, 'vin'] = df.loc[mask, 'state']
df.loc[mask, 'state'] = np.nan
print('\nAll 26 rows cleand ✅\n')


# In[141]:


# Change the wrong values in body column 
df['body'] = df['body'].str.lower()
df['model'] = df['model'].str.lower()
df['trim'] = df['trim'].str.lower()
print('\nbody column values cleand ✅\n')


# In[142]:


print("\nHandling Data type of saledate column")
# Unify dates and convert them to UTC time
df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)

# I wanted to remove the time and make the dates Naive
df['saledate'] = df['saledate'].dt.tz_convert(None)

# Verify column type
if isinstance(df['saledate'].dtype, pd.DatetimeTZDtype):
    df['saledate'] = df['saledate'].dt.tz_localize(None)
print('\nsaledate column data type handling, Done ✅\n')


# In[143]:


# rename some columns
df.rename(columns={"year":  "model_year", "make":  "Brand",   "odometer":  "motor_mi"},inplace=True)
df.columns = df.columns.str.lower()
print(f"\nNew columns name{df.columns}")
print("\nRename some columns, Done ✅\n")


# In[144]:


# Clean brand cloumn Wrong Values
print()
print('📌Clean brand cloumn Wrong Values!\n')
df['brand'] = df['brand'].str.lower()

replace_map = {
    'ford truck': 'ford',
    'ford tk': 'ford',
    'chev truck': 'chevrolet',
    'chevrolet': 'chevrolet',
    'maserati': 'maserati',
    'mercedes-b': 'mercedes-benz',
    'mazda tk': 'mazda',
    'dodge tk': 'dodge',
    'hyundai tk': 'hyundai',
    'volkswagen': 'vw',
    'vw': 'vw',
    'gmc truck': 'gmc',
    'land rover': 'landrover',
    'aston martin': 'aston martin',
    'rolls-royce': 'rolls-royce',
    'fiat': 'fiat',
    'mini': 'mini',
    'dot': 'dot',
}
df['brand'] = df['brand'].replace(replace_map)
print("Clean, Done✅\n")


# In[145]:


# Clean body cloumn Wrong Values
print()
print('📌Clean body cloumn Wrong Values!\n')
df['body'] = df['body'].str.lower()
body_mapping = {
    'sedan': 'sedan',
    'g sedan': 'sedan',
    'suv': 'suv',
    'hatchback': 'hatchback',
    'minivan': 'minivan',
    'coupe': 'coupe',
    'g coupe': 'coupe',
    'genesis coupe': 'coupe',
    'elantra coupe': 'coupe',
    'koup': 'coupe',
    'cts coupe': 'coupe',
    'q60 coupe': 'coupe',
    'cts-v coupe': 'coupe',
    'g37 coupe': 'coupe',
    'convertible': 'convertible',
    'g convertible': 'convertible',
    'beetle convertible': 'convertible',
    'q60 convertible': 'convertible',
    'g37 convertible': 'convertible',
    'granturismo convertible': 'convertible',
    'wagon': 'wagon',
    'tsx sport wagon': 'wagon',
    'cts wagon': 'wagon',
    'cts-v wagon': 'wagon',
    'crew cab': 'pickup',
    'supercrew': 'pickup',
    'supercab': 'pickup',
    'regular cab': 'pickup',
    'regular-cab': 'pickup',
    'extended cab': 'pickup',
    'quad cab': 'pickup',
    'double cab': 'pickup',
    'crewmax cab': 'pickup',
    'king cab': 'pickup',
    'access cab': 'pickup',
    'club cab': 'pickup',
    'mega cab': 'pickup',
    'xtracab': 'pickup',
    'cab plus 4': 'pickup',
    'cab plus': 'pickup',
    'van': 'van',
    'e-series van': 'van',
    'promaster cargo van': 'van',
    'transit van': 'van',
    'ram van': 'van',
    'other': 'other'  # fallback value
}




df['body'] = df['body'].map(body_mapping).fillna('other')
print("Clean, Done✅\n")


# In[146]:


# Clean model cloumn Wrong Values
print()
print('📌Clean model cloumn Wrong Values!\n')
df['model'] = df['model'].str.lower()
car_models = {
    # Nissan
    'altima': 'altima',
    'altima hybrid': 'altima',
    'maxima': 'maxima',
    'sentra': 'sentra',
    'versa': 'versa',
    'versa note': 'versa',
    'rogue': 'rogue',
    'pathfinder': 'pathfinder',
    'murano': 'murano',
    'juke': 'juke',
    '370z': '370z',
    'gt-r': 'gt-r',

    # Toyota
    'camry': 'camry',
    'camry hybrid': 'camry',
    'camry solara': 'camry',
    'corolla': 'corolla',
    'prius': 'prius',
    'prius c': 'prius',
    'prius plug-in': 'prius',
    'prius v': 'prius',
    'avalon': 'avalon',
    'yaris': 'yaris',
    'rav4': 'rav4',
    'highlander': 'highlander',
    '4runner': '4runner',
    'tacoma': 'tacoma',
    'tundra': 'tundra',

    # Honda
    'accord': 'accord',
    'civic': 'civic',
    'fit': 'fit',
    'cr-v': 'cr-v',
    'pilot': 'pilot',
    'odyssey': 'odyssey',

    # Hyundai
    'elantra': 'elantra',
    'elantra coupe': 'elantra',
    'elantra gt': 'elantra',
    'elantra touring': 'elantra',
    'accent': 'accent',
    'sonata': 'sonata',
    'sonata hybrid': 'sonata',
    'veloster': 'veloster',
    'tucson': 'tucson',
    'santa fe': 'santa fe',

    # Ford
    'fusion': 'fusion',
    'fusion hybrid': 'fusion',
    'fusion energi': 'fusion',
    'focus': 'focus',
    'fiesta': 'fiesta',
    'mustang': 'mustang',
    'escape': 'escape',
    'explorer': 'explorer',
    'f-150': 'f-150',

    # Chevrolet
    'cruze': 'cruze',
    'malibu': 'malibu',
    'impala': 'impala',
    'spark': 'spark',
    'camaro': 'camaro',
    'corvette': 'corvette',
    'silverado 1500': 'silverado',
    'silverado 2500hd': 'silverado',
    'silverado 1500hd': 'silverado',
    'silverado 1500 classic': 'silverado',
    'equinox': 'equinox',
    'tahoe': 'tahoe',
    'suburban': 'suburban',

    # BMW
    '3 series': '3 series',
    '5 series': '5 series',
    'x3': 'x3',
    'x5': 'x5',
    'x5 m': 'x5',
    'x6': 'x6',
    'x6 m': 'x6',

    # Mercedes-Benz
    'c-class': 'c-class',
    'e-class': 'e-class',
    's-class': 's-class',
    'gl-class': 'gl-class',
    'gla-class': 'gla-class',

    # Audi
    'a4': 'a4',
    'a6': 'a6',
    'q5': 'q5',
    'q7': 'q7',

    # Infiniti
    'g coupe': 'infiniti g',
    'g sedan': 'infiniti g',
    'g35': 'infiniti g',
    'g37': 'infiniti g',
    'g37 convertible': 'infiniti g',
    'g37 coupe': 'infiniti g',
    'qx': 'qx',

    # Kia
    'optima': 'optima',
    'rio': 'rio',
    'soul': 'soul',
    'sportage': 'sportage',
    'sorento': 'sorento',

    # ... (يمكنك إضافة المزيد بنفس الطريقة)
}



df['model'] = df['model'].map(car_models).fillna('other')
print("Clean, Done✅\n")


# In[115]:


# change wrong rice values
df.loc[df['sellingprice'] < 1000, 'sellingprice'] = 1000


# In[116]:


# change wrong rice values
df.loc[df['mmr'] < 1000, 'mmr'] = 1000


# In[147]:


# change wrong rice values
df.loc[df['motor_mi'] < 150, 'motor_mi'] = 150


# ## Handling Missing Values

# In[148]:


# Check Missing Values
print("\n🔍 Null values ratio per column:\n\n")
for col in df.columns :
    print(f"Column : {col}")
    print(f"Missing values count = {df[col].isna().sum()}")
    print(f"Missing % = {(df[col].isna().sum()/len(df))*100}")
    print("="*50 + '\n')


# In[149]:


# drop_na: sellingprice , mmr 
# the info the make good fill not included in data in them rows so i choice dropna


# In[150]:


# Handling missing num coumns
df.drop(df.loc[df['mmr'].isna()].index, axis=0, inplace=True) # drop null values of mmr
df.drop(df.loc[df['sellingprice'].isna()].index, axis=0, inplace=True) # drop null values 
df['motor_mi'] = df['motor_mi'].fillna(df['motor_mi'].mean()) # fill missing with mean
print("\nHandling numerical missing values , Done ✅\n")


# In[151]:


# handling cat_col
cat_column = ['brand', 'model', 'trim', 'body', 'transmission','color', 'interior', 'seller','state','condition']
for col in cat_column:
    df[col] = df[col].fillna(df[col].mode()[0]) # fill missing with mode
print("\nHandling categorical missing values , Done ✅\n")


# In[152]:


mask = df['saledate'].isna() # null values in saledate

# fill Missing values in the saledate in the year following the year of production
df.loc[df['saledate'].isna(), 'saledate'] = pd.to_datetime(df.loc[df['saledate'].isna()
                                                           , 'model_year'].astype(str)) + pd.DateOffset(years=1)
print("\nHandling date missing values , Done ✅\n")


# ## Handling Outliers

# In[153]:


# check numerical columns to know how i handling
numeric_columns = ['motor_mi' , "mmr", "sellingprice"]
print('\n📌Boxplt for Numerical columns\n\n')
for col in numeric_columns :
    sns.boxplot(data = df , x = col) # boxplot figer
    plt.show()
print()
print()


# In[154]:


# check outlier percentage in num columns
print("\n🔍 outlier values ratio per column:\n\n")
for col in numeric_columns:
    Q1 = df[col].quantile(0.25) 
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"Column: {col}")
    print(f"upper_bound = {upper_bound}")
    print(f"lower_bound = {lower_bound}")
    print(f"Number of outliers: {len(outliers)}")
    print(f'percentage of outlier = {(len(outliers)/len(df))*100}')
    print("-" * 20)
print()
print()


# In[157]:


print("""
⚠️I decided not to make any changes to these columns : mmr & sellingprice
because these are real values in the data and they exist. It would be wrong to remove or change them.
And i decided to handling some outliers at motor_mi column .
But threshold , which is 400,000 ⚠️
""")


# In[158]:


Q1 = df['motor_mi'].quantile(0.25)
Q3 = df['motor_mi'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
median_value = df['motor_mi'].median() 
mean_value = df['motor_mi'].mean()
#Replace outliers with the mean and median
df['motor_mi'] = np.where(df['motor_mi'] < lower_bound,mean_value, df['motor_mi'])
df['motor_mi'] = np.where(df['motor_mi'] > 400000,median_value, df['motor_mi'])
print("\nHandling Outlier values in motor_mi column, Done ✅!\n")


# In[159]:


# drop outlier over 183K in sellprice 
outliers = df[df['sellingprice'] > 175000] 
df.drop(outliers.index, inplace=True)
outliers = df[df['mmr'] > 175000] 
df.drop(outliers.index, inplace=True)



print("\nHandling All Outlier values, Done ✅!\n")



# In[160]:


# handling outlier condition
mask = (df['condition'].isin([1,2,3, 4, 5])) & (df['sellingprice'] > df['sellingprice'].quantile(0.50))
df.loc[mask, 'condition'] = np.random.randint(45, 49, size=mask.sum())


# # Data Prepration

# In[161]:


# make new column it's name market_advantage : The slight difference between the expected price and the selling price.
df['market_advantage'] = df['sellingprice'] - df['mmr']
print("\nAppend new market_advantage column, Done ✅!\n")


# In[162]:


# make columns to selldata call them 
df['sell_year'] = df['saledate'].dt.year    # sell_year
print("\nAppend new sell_year column, Done ✅!\n")


df['sell_month_name'] = df['saledate'].dt.month_name() # sell_month_name
print("Append new sell_month_name column, Done ✅!\n")


df['sell_month'] = df['saledate'].dt.month  # sell_month
print("Append new sell_month column, Done ✅!\n")


df['sell_day_name'] = df['saledate'].dt.day_name() # sell_day_name
print("Append new sell_day_name column, Done ✅!\n")


df['sell_day'] = df['saledate'].dt.day # sell_day
print("Append new sell_day column, Done ✅!\n")


df['sell_hour'] = df['saledate'].dt.hour # sell_hour
print("Append new sell_hour column, Done ✅!\n")


# In[163]:


# func to defind season column
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
df['season'] = df['sell_month'].apply(get_season)
print("\nAppend new season column, Done ✅!\n")


# In[164]:


# func to defind time period column
def get_time_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
df['time_period'] = df['sell_hour'].apply(get_time_period)
print("\nAppend new time_period column, Done ✅!\n")


# In[165]:


# Check the unique values in catgorical columns  
print("\n🔎Check the unique values in catgorical columns : \n\n")
cat_column = ['brand', 'model', 'trim', 'body', 'transmission','color', 'interior', 'seller','state', "season", 'sell_month_name']
for col in cat_column :
    print(f'column : {col}')
    print(df[col].value_counts().head(30)) # preview top 30 unique values with count 
    print(f"\ncount of values :{len(df[col].unique())}") # preview the count of unique values
    print("="*50 + "\n")


# In[166]:


''' Befor change data type 
from collections import Counter

# List to store months
months = []

# Passing through all rows
for date in df['saledate']:
    try:
# Split and get the name of the month (the second word in the date)
        month = date.split()[1]
        months.append(month)
    except:
        months.append(None)  

# We calculate the frequency of each month
month_counts = Counter(months)

print(month_counts)

# Output :
Counter({'Feb': 163053, 'Jan': 140815, 'Jun': 99937, 'Dec': 53520, 'May': 52447, 'Mar': 46277, 'Apr': 1450, 'Jul': 1300, None: 38})
'''


# In[167]:


# clean some Wrong values in date 
mask = df['model_year'] > df['sell_year'] # check if car model_year > sell year it was wrong value
df.loc[mask, 'model_year'] = df.loc[mask, 'sell_year'] # So , wrong values will change to the sell year
print("\n⚠️Change some Wrong model_year values, Done ✅\n")


# In[168]:


df.drop( 'vin' , axis = 1 , inplace  = True) # drop some columns 
print("\n⚠️droped vin column ,Done ✅\n")


# In[169]:


df.reset_index(drop=True, inplace=True)
print('\n📌Reset index of data frame ,Done✅\n')


# In[170]:


# function to return data frame
def return_df():
    return df
    


# In[ ]:





# In[ ]:




