# -*- coding: utf-8 -*-
"""
Reference Code

Here are a slew of reference code lines to explain code syntax, 
rules & oddities that I need to remember, and more
"""
#--------------- MODULES -----------------------------------------
# Built In modules
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple
from itertools import combinations # for creating combinations/permutations of things
from os
from sys
from math

# others
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rand # to use rand.randint() for random integer generator
import folium # Folium module can be used to create maps
import pathlib as path # Lets you use path.Path('C:/Users/mattc/path/') to set paths
from pandas_datareader.data import DataReader # Downloads financial data from google/yahoo Finance n more
from datetime import date # date& time funcionality. Related to DataReader
from datetime import datetime
from datetime import timedelta
from pytz import timezone
import pendulum
import csv
from scipy import stats
# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#--------------- SETUP / Workspace stuff -----------------------------------------
path.Path('C:/Users/mattc/path/') # to set paths. Works best with / slashes

# IMPORTING
    file = open('filename.csv','r'): # open link to file. mode = 'r' for read-only
         file.close() #close it afterwards. 
    # OR ... no need for file.close() with this method
    with open(casepath+casefile,'r') as file:
        caselist = file.read() 
    
    # CSV
    csv.reader(file)
    # TXT
    file.read()     # read & return whole .txt file as a string
    file.readline() # read & return 1 line from .txt file
    
    data = np.loadtxt(filename, delimiter = ',', skiprows=1, usecols=[0,3],dtype=str) # Not good for mixed data types
    data = np.genfromtxt() # very similar to loadtxt but more powerful. Handles mixed datatypes
    data = np.recfromcsv() # like genfromtxt() but with default values delim=',', names=True, dtype=None

%timeit # Use timeit on only ONE line of code
%%timeit -r2 -n10 # used to measure execution time of code segments. -r = # of runs. -n = # of loops
times = %%timeit -o #Store the output
    
#--------------- TIME / Timezones -----------------------------------------
time_inc = datetime.strptime(date_var,'%m/%d/%Y') # Converts 'MM/DD/YYYY' to time instance
time_inc = datetime.strftime(date_var,'%m/%d/%Y') # Converts time instance to 'MM/DD/YYYY' string

datetime.now() # local time
datetime.utcnow() # global time
cent = timezone('US/Central') # set timezone object cent to US central time zone

timedelta(days=90) # use a time delta to add/subtract time for looking forwards/backwards

    # Methods
        .astimezone('US/Pacific') # returns a converted time in the requested time zone
time_inc.replace(tzinfo = 'US/Central') # use replace() to insert TZ info to naive object

    # Pendulum objects/methods
pendulum.parse('time_string',tz='US/Eastern') # return a pendulum datetime object from a date/time string
                strict = False # for datetime string that's NOT ISO8601 format
        .in_timezone('Asia/Tokyo') # convert pendulum datetime object to a particular timezone
        .in_words() # provide a time delta in easy to understand wording
        .in_days()
        .in_hours()
        .to_iso8601_string() # convert time object to ISO8601 standard string



#--------------- ARRAYS -----------------------------------------
# Appending / adding row or column to array
at = np.array([[1,5],[2, 6],[3, 7],[4, 8]]) # a 4x2 array
at2=np.append(at,[[5, 9]],axis=0)
# axis=0 argument adds the row. axis=1 adds a column
at3=np.append(at2,[[10],[11],[12],[13],[14]],axis=1)

# Importing
    file = 'digits.csv'
    digits = np.loadtxt(file, delimiter = ',') # ONLY GOOD FOR TEXT
    digits = np.genfromtxt(file, delimiter='\t', dtype=None, names=True) # preferred over np.loadtxt
                # Good for data with different types in different columns. 
                # dtype = None tells it to figure out each column's type
                # names = True tells it to expect a header in the first row to remove & use
    digits = np.recfromcsv(file) # Similar to genfromtxt, but it uses defaults for delimiter, names, and dtype

# Methods
    .insert(my_array,row,colc) # insert a value into array

# Logic and booleans. 
# Use np.logical_and() / np.logical_or() / np.logical_not() on arrays


#--------------- SQL DATABASES -----------------------------------------
# Process:
    #1. import create_engine'
    #2. create an engine
    #3. use that engine to connect to database
    #4. query database. store result
    #5. disconnect the database

from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite') #string that indicates type of database and its name
# Connect to the Database
con = engine.connect()
# query the database
rs = con.execute("SELECT * FROM Orders") # Orders is a table name in the db
df = pd.DataFrame(rs.fetchall()) # fetchall fetches all rows of rs
df.columns = rs.keys() # to get column headers
# close connect
con.close()
# Can also do this with:
    with engine.connect() as con:
        rs = con.execute("SELECT OrderID, OrderDate FROM Orders") # SELECT [columns] FROM table_name
        df = pd.DataFrame(rs.fetchmany(5)) # fetchmany fetches a given number of rows of rs
        # etc. Here do all the query & storing
# OR shortcut it to
df = pd.read_sql_query('SELECT * FROM Orders', engine)

# Inner Joins and retrieving columns from different tables in a database
df = pd.read_sql_query('SELECT OrderID, CompanyName FROM Orders INNER JOIN Customers on Orders.CustomerID = Customers.CompanyName', engine)
# OrderID, CompanyName are desired columns
# Orders, Customers are table_names linked using INNER JOIN
# joing is done "on" connection made between each table's common column: Orders.CustomerID = Customers.CustomerID

# Methods
    engine.table_names() # list of table names contained in the linked database engine

#--------------- DATAFRAMES -----------------------------------------
# BASICS
# Use 1 bracket df[] to pull a series from the dataframe
# Use 2 brackets df[[]] to pull a subset dataframe from the dataframe

# IMPORTING
new_df = pd.read_json()
new_df = pd.read_sql()
new_df = pd.read_csv('filename.csv',index_col=(0)) # where 0 means use the first column to as labels in the dataframe
    # Modifications
    chunksize = 4 # using chunksize creates generator/iterable that reads in X many rows each time its called
    comments = '#' # Tell it which character indicates comments embedded in data
    header = None # Tell it there's no header row in the data
    na_values = 'n/a' # to replace n/a values. Cleans up data type detected in DF columns
                      # converts a string to np.nan, (Not A Number)
                      # n/a can be replace to 'NAN' or some other error code present
    nrows = 20        # Import the provided number of rows
    parse_dates=['Date Col'] # Or other column name corresponding to a date. 
                             # Parses data therein as formamt datetime64.
    sep = '\t' # pandas version of delimiter
newdf = pd.read_excel('filename.xls')
    # Modifications
    sheetname = 0; # Grab first sheet. OR. sheetname= ['sheet1', 'sheet3']
                   # Loading multiple sheets into "newdf" makes newdf into a dictionary
                   # for each sheet where the key = each sheetname, value = sheet's dataframe
    na_values = '......' # Same as for read_csv

xls = pd.ExcelFile('file.xlsx') # Find out about an excel file
sheets = xls.sheet_names    #sheet_names is an attribute of the file

stock_data = DataReader('ticker',data_source,start, end)
        ticker = 'GOOG', 'TSLA', etc
        data_source = 'google' # or some other source from an available list
        start = date(2015,1,17) # Jan 17 2015
        end = date(2021, 8, 29) # Aug 29 2021
# EXPORTING
DataFrame.to_csv('output_name.csv')


# ATTRIBUTES
    .dtype # provides a series list of the DF's columns and data type contained therein
# Creating DF
df= pd.DataFrame(list_of_dicts) # creates a DF row-by-row for each dictionary in the list. (Dictionary keys become column headers)
df= pd.DataFrame(dict_of_lists) # creates a DF column-by-column for each value(list) with each key as column header

# FUNCTIONS
newdf = pd.concat([df1, df2, df3],axis=0) #Concatenate or stack dataframes
                                          # Axis = 0 to stack vertical; 1 for horizontal
dummies=pd.get_dummies(df['column']) # creates dummy variables in dataframe for categorical values in 'column'. One for each unique value (gas/diesel/etc)                                # Axis = 0 to stack vertical; 1 for horizontal
newdf = pd.merge(df1, df2, on='shared_col_name') # merges dataframe based on column reference provided that they share

# PLOTTING
    df.plot()
        subplot = True # for plotting dataframe's columns in multiple charts/subplots
        secondary_y = 'Column name' # For a secondary Y-axis scale
        kind = 'scatter' # bar/ barh / line / hist etc.
    df.plot.bar() 
           .scatter(x='col_name', y='col2_name')
           .hist() #.... etc
    
            
# METHODS
    .any() # check whether any value is True, potentially over an axis
    .apply(func, axis = 0 ) # apply function to 1 row/column at a time of dataframe. Direction is required. axis = 0 to apply on columns, =1 to apply on rows
        # Similar to a generator function. Needs to be called to store result: baseball['RD'] = run_diffs_apply.... where run_diffs_apply = baseball_df.apply(func, axis=1)
    .astype('int')    # used to convert data types. Input to method is the intended target datatype
    .corr() # create a correlation matrix (size of (columns x columns)) with correlation between different columns
    .cut(df['price'],bins,labels = ['Low','Medium','High'],include_lowest=True) # Break up numerical data into bins
        bins = np.linspace(0,100,4) # min value = 0, max value = 100, 4 numbers requested thus creating 3 bins defined by those 4 boundary values
    .describe() # list statistical summary (mean, std, quartiles, min, max)
        include='all' # provides full, more comprehensive, summary statistics    
    .div(otherDF)  # divide primary DF by otherDF
    .dropna() # to get rid of missing values' rows (axis=0) or columns (axis=1)
        .dropna(inplace = True) # directly update the dataframe
    .drop() # Delete a row or column
        .drop(labels=["column name 1", 'column name 2'], # Indices/labels for rows. Labels for columns
          axis = 1 # Axis 1 for columns; Axis 0 for rows
          inplace=False) # Alter DF directly (inplace=True) or return a copy (False)
        .drop(labels=[0],axis=0) # Delete row of units
        .reset_index(drop = True) # Reset indices to start at 0 again. Does not insert old indices into DataFrame
    .dt.component # Extract date information from a date-type column. component = year/month/day/etc
    .get_group('group_or_row_var')['price'] # subsets a specific group from a dr.groupby() dataframe
    .groupby(['col_name'],as_index=False)[['price']] # returns dataframe. Can use single brackets around target column for series
        as_index = False # doesn't force the input col_name to be index of output DF
    .idxmax('Column name') # Similar for idxmin. Return index of max/min value in the referenced column
df[col].isin(['list','of','possibilities'])
    .isna() # detect and identify missing values (creates DF of booleans)
    .isnull()     # checks for missing values. Opposite of notnull()
    .iterrows() # iterate along rows of DF. Produces a tuple of (index, row's data as a series)
    .itertuple() # iterate along rows of DF as tuples. Returns a named tuple, which attribute lookup for columns
    .mul()  # multiply primary DF by otherDF or value
    .notnull()
    .pivot_table(values = 'unit_cost', index = ['department', 'product_line'], columns = 'products', aggfunc = np.sum, fill_value=0) # columns not required. 
        # pivot tables similar to .groupby() but it can apply several functions across columns.
        # pivot table is an index-sorted dataframe. Use .loc[] functions for slicing
        values=['D', 'E'], index=['A', 'C'], aggfunc={'D': np.mean,'E': [min, max, np.mean]})
        fill_value = 0 # Replace NaN values with a default value you determine
        margins = True # Include an extra row & column averaging all the data in that row
        # Pivot tables have their own methods, & method-arguments:
            df.pivot_table.mean(axis='columns')  # calculate mean across columns of pivot table
    .pct_change() # calculate percent-change in consecutive rows of dataframe column
    .rename(columns = {'old_name': 'new_col_name'}, inplace=True)
    .replace('old',np.NaN) # or 'new' or some other replacement value in the whole DF
    .set_index('Column Name') # set a particular column as Index label
    .sort_index(level = ['col1, col2'],ascending = [True, False])
    .sort_values(['Col1, Col2'],ascending = [False, True]) # brackets not needed for singular column
    .sub(otherDF)  # subtract otherDF from primary dataframe columns
    .unique() # Return unique values of a dataframe as a numpy array, when used on a column df['column'].unique()
    .value_counts() # Returns a tally of how many times a particular value appears in a column
    
# Slicing/subsetting
    # Best done after sorting dataframe by index df_srt = df.sort_index()
    subset = df_srt.loc['index_label' : 'other_index_label'] # subset row to row with 1 index
    subset = df_srt.loc[('index1_label','index2_label') : ('other_index1_label','other_index2_label')] # subset row to row with multi-index
    subset = df._srt.loc[('index1_label','index2_label') : ('other_index1_label','other_index2_label'), 'col_label1' : 'col_label2'] # subset multi-index row-to-row AND specific column to column

#--------------- DICTIONARIES-----------------------------------------

# Counters
    eatery_count_by_types = Counter(eatery_types) # return dictionary with types: qty of that type
        .most_common(3) # give top 3 most common keys, and their values
# Default Dicts - when populating a dictionary, if the desired key is not-present you can create it with a default chosen value, or empty
    eatery_by_park = defaultdict(list)
    eatery_contact = defaultdict(int)
# Ordered Dicts - dictionaries kept in a sorted order
    eatery_permits=OrderedDict()
    eatery_permits[eatery['end_date']]=eatery # 'end_date' is column you wish to sort by
        .popitem() # method returns the items in reverse insertion order. or .popitem(last=False) to return in insertion order

# Functions
    del()    # will throw KeyError is key requested does not exist. use .pop() to safely delete without errors
    sorted(my_dict, reverse = False) # ascending sort
    key in my_dict # returns boolean after checking if requested 'key' is in my_dict 

# Importing 
    import csv; file=open('csvfile.csv','r')
    csv.DictReader(f, args, kwds)

# Methods
    .append()    
    .get('desired key', 'Not Found') # safely access a key without error or exception handling
           # Returns None if a key is not in the dictionary, or some default you want "Not Found"
    .items()
    .pop(key, 'default if error') # safely return & remove keys & associated values from dictionary
    .update({new_dict}) # combine new_dict with original dictionary   






#--------------- FUNCTIONS -----------------------------------------

global   # make reference to a global variable defined in main body of script
nonlocal # make reference to a nonlocal variable within a nested function from a parent function 

interact(func, input1, input2) # found in ML Jupyter notebook. need investigating

# LAMBDA
# quickly define a function in 1 
rand_multiple = lambda k: k * rand.randint(2,5)
    # call rand_multiple(a) to multiply a by a random integer between 2,5


# Maps - apply function on all elements of a sequence (like a list)
nums = [48, 6, 9, 21, 1]
square_all = map(lambda kk: kk**2, nums)
print(list(square_all))  # Have to put list() function to not print <map_object>



#--------------- GENERATORS -----------------------------------------
# variable equals the function in parentheses

case = (j for i, j in enumerate([5,6,7,8]))
    



#--------------- LISTS -----------------------------------------
x = ['a', 'b', 'c']
y = list(x) # use this notataion to copy x's list contents into Y, 
            # so you can't accidentally update x while updating y
# Combine lists as columns using zip()

# Functions
sorted(list_name) 
enumerate(list_name, 1) # begin indexing at 1, or some other number, instead of zero.
zip(list1, list2) # where both are 1-D lists

# Methods
    .append('value') # add one value to a list
    .extend(list_name) # append a list with a whole nother list
    .index('value') # Return index location of desired value
    .pop([3]) # Remove item in list belong to index 3



#----------------- Machine Learning: Regression----------------------------
from sklearn.model_modelselection import train_test_split
from sklearn.model_modelselection import cross_val_score
                                         cross_val_predict

# Randomly split up datasets (x, y=target) into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.10, random_state=1)
    test_size = percentage of data to designate as test data
    random_state = seed for random number generation


# Cross Validation
Rcross = cross_val_score(lr_obj,x_data[['feature']]),y_data, cv = 4) # cv = # of folds. Performed on total data set to try different permutations of train/test data
    # Arguments:
        # lr_obj is the model/regression object. Followed by X_data features desired, then Y/Target, and folds
        scoring = 'neg_mean_squared_error' # Change the method of scoring from R-squared default to negative mean squared error
    # Can also be used for predictions:
   yhat = cross_val_predict(lr_object, x_data[['feature']], y_data, cv=2)


######### Models: Simple Linear Regression, Multi Linear Regress, Polynomial Regression
          Ridge Regression (& grid search)
# Linear Regression Model
lm = LinearRegression()
lm.fit(X_train[['feature']], Y) # Train the model on the intended features "X"
             # X_train = df[['highway-mpg']];       Y = df['price']
                   # DF for independent var      Series for Target
             # lm.intercept_ (b0) is an attribute of lm now
             # lm.coef_ (b1) is also an attribute of lm now
lm.predict(X2[['feature']]) # Obtain a prediction    
lm.score(Xtest[['feature']], Ytest)  # Calculate R-squared score of the model on test data, still based on same X features used in fit/training:  X = df[['highway-mpg']]

# Multiple Linear Regression - mostly the same as linear
lm.fit(X, Y) # X = df[['highway-mpg','horsepower','engine-size','curb-weight']]; 

# Singular Polynomial Regression (only 1 independent variable)
f = np.polyfit(x,y,3)
p = np.poly1d(f)        # p can be printed out in symbolic form

# Multi-dimensional Polynomial Regression
    # Test several different degrees at once with a for-loop. Storing/appending the scores into a list
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
                           import StandardScaler     # used for normalizing features
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
pr = PolynomialFeatures(degree=2) 
    # Arguments
    include_bias = False # Default is True.  Dunno what this is yet
    interaction_only = True # Default is False. Dunno what this is yet
x_poly = pr.fit_transform(x[['horsepower','curb_weight']])  # Transform DF features into polynomial feature

SCALE = StandardScaler()
SCALE.fit(x[['horsepower','curb_weight']])  # train the scaler object
x_scale = SCALE.transform(x[['horsepower','curb_weight']])   # assign and store transformed/normalized x_data for use in regression

# Ridge Regression
from sklearn.linear_model import Ridge
Ridge_Obj = Ridge(alpha=.1)
Ridge_Obj.fit(x_train, y_train)

# Grid Search - Used to test several different hyperparameter combinations
from sklearn.linear_model import GridSearchCV
parameters = [ {'alpha': [.001, .01, .1, 1, 10, 100, 1000],
               'normalize': [True, False]} ] # list of dictionary keys
    # Other possible parameters
    normalize = True
    copy_X= True
    fit_intercept = True
    max_iter = None
    random_state = None # or a seed integer
    solver = 'auto'
    tol = 0.001
Grid_Obj = GridSearchCV(Ridge_Obj, parameters, cv = 4) # cv = # of folds desired
    # Arguments
    iid=None # avoids a deprecation warning due to the iid parameter
Grid_Obj.fit(x_train, y_train)
BestRR = Grib_Obj.best_estimator_   # Returns RidgeRigression Object with the best values for the tested hyperparameters
    BestRR.score(x_test, y_test)

# PIPELINES
# Pipeline constructor: list of tuples: (name_of_estimator_model, model_constructor)
pipe = Pipeline(Input) #see Input below
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('mode',LinearRegression()))]

pipe.fit(df[['highway-mpg','horsepower','engine-size','curb-weight']], y )  # Training the pipeline object
yhat = pipe.predict(X[['highway-mpg','horsepower','engine-size','curb-weight']])   
    

# Error checking / Evaluation
mean_squared_error(df['price'],Y_predict_simple_fit)
    # Rsquared Linear regression   
    lm.score(df[['highway-mpg']],df['price'])   # Rsquared (R squared R2)
    # Rsquared Polynomial regression
    from sklearn.metrics import r2_score
    r_squared = r2_score(y, p(x)) # y = actuals. p(x) = prediction of x inputs
    
    
# Plots
    # Residual Plot
    sns.residplot(df['highway-mpg'],df['price'])
    # Distribution Plot
    sns.distplot(df['price'],hist=False, color = 'r', label='Actual Value') # hist=False makes it normalized to a curved distribution
    sns.distplot(yhat,hist=False, color = 'b', label='Predict Value')

    # Pre-built distribution plot functions!!!
    # Linear Reg plot:
    def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
        width = 12
        height = 10
        plt.figure(figsize=(width, height))
        ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
        ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)
        plt.title(Title)
        plt.xlabel('Price (in dollars)')
        plt.ylabel('Proportion of Cars')
        plt.show()
        plt.close()
    # Polynomial Reg plot:
    def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
        # lr:  linear regression object 
        # poly_transform:  polynomial transformation object 
        width = 12
        height = 10
    
        xmax=max([xtrain.values.max(), xtest.values.max()])
        xmin=min([xtrain.values.min(), xtest.values.min()])
        x=np.arange(xmin, xmax, 0.1)
        plt.plot(xtrain, y_train, 'ro', label='Training Data')
        plt.plot(xtest, y_test, 'go', label='Test Data')
        plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
        plt.ylim([-10000, 60000])
        plt.ylabel('Price')
        plt.legend()


#----------------- Machine Learning: Clustering----------------------------
from sklearn.cluster import KMeans

# Kmeans algorithmn
kmeans = KMeans(4, random_state=8) # (# of desired clusters, random seed)
Y_hat = kmeans.fit(X).labels_ #Training PLUS an attribute for the labels created after clustering
mu = kmeans.cluster_centers_ # Means of clusters

#--------------- PLOTS -----------------------------------------
# Basics: Define all characteristics of a plot (plt.label(), plt.xscale(), etc.)
#         and THEN show plot with plot.show().   
#         plt.clf() clears the plot

# Magic function: inline plots. Dont need a plt.show() command to plot charts 
# inline in Jupyter notebook
%matplotlib inline

# Plotting/inserting text into chart
plt.text(2.0, 5.5, 'Chart Text')   # format (x,y,text_string)

plt.tight_layout() # Reduces whitespace and looks cleaner

plt.legend() # add legend

# Subplots
plt.subplot(3,2,1) # Activate plot 1 of a grid of 3x2 plots

# STYLES!!! - use pre-established chart format styles
print(plt.style.available) # To view available styles
plt.style.use('fivethirtyeight') # Use fivethirtyeight website's style format for chart

import seaborn as sns
sns.set() # then just process & produce the plots as normal. no other style inputs needed

# Bee Swarm Plot - similar to Histogram but with extra fidelity of categorical groupings (% of vote Obaama got vs the 3 swing states)
    # Data along the y-axis is quantitative
    # Data is spread along x-axis (categories) to make them visible
    # may require Seaborn to be imported and set as style
sns.swarmplot(x=df'state',y=df'democratic_vote_pcnt', data = dataframe)
    # Dataframe as input. X, Y set equal to columns


# Bar Chart
plt.bar()   # Classic vertical bar chart
plt.barh()  # Horizontal bar chart
    # Modifications:
    yerr = df.error # Plot error bars for each plotted category in bar chart
# Stacked bar charts are built from the bottom up. Plot the bottom data (dog), 
# then the next layer
    plt.bar(df.precinct, df.dog) # Use as bottom
    plt.bar(df.precinct, df.cat, bottom = df.dog) #Stacked ontop of dog


# Box Plot
sns.boxplot(x="body-style", y="price", data=df)


# Line Plot
plt.plot()
    # Modifications:
    data["Year"], data["Los Angeles Police Dept"] 
    label="Los Angeles"  # name for a particualr data series / line
    xlabel = 'X Axis Name'
    linestyle = ':' # Dotted line. Also '-' '--' '-.'
    marker='s' # For square marker. Also try 'o' (circle)   'd' (diamond)
    color = 'DarkCyan'  # Google "web colors" for other valid color names
    rot = 45 # Rotate x-axis labels by provided angle
    # ECDF - Empirical Cumulative Distribution Function
    # Quantitative data is along x-axis (e.g. % of Obama votes)
    # Cumulative distribution on Y-axis says: 'n % of counties had less than x-axis value(e.g. % of obama vote)
    x = np.sort(df['dem_share'])
    y = np.arange(1, len(x)+1) / len(x) # Must be normalized to go from 0 to 1
    plt.plot(x,y,marker='.',linestyle='none')
    plt.margins(0.02) # keeps data off plot edges

# Scatter Plot
plt.scatter()
    # Modifications: (many of the same ones from line plots work here)
    alpha = .3 # Alpha takes a number from 0 - 1 to adjust transparency. 0 = transparent
# Scatter Plot with regression Line
sns.regplot(x='engine-size',y='price',data=df)
plt.ylim(0,)

# Heat Map
plt.pcolor(df_pivot, cmap = 'RdBu') #cmap probably means the reference color of map
plt.colorbar() # akin to a legend or gradient scale

# Histogram
plt.hist()   # Classic histogram with a default of 10 bins assumed
    # Modifications:
        bins = nbins
        range = (xmin, xmax)
        density = True # Normalize the histograms (for different series) to 
                       # plot them cleaner alongside each other
 
    
# Maps - using folium
phone_map = folium.Map()    
companies = [
    {'loc': [37.4970,  127.0266], 'label': 'Samsung: 20.5%'},
    {'loc': [37.3318, -122.0311], 'label': 'Apple: 14.4%'},
    {'loc': [22.5431,  114.0579], 'label': 'Huawei: 8.9%'}]
for company in companies:
    marker = folium.Marker(location=company['loc'], popup=company['label'])
    # Creates a "marker" for each data point/location from the dictionaries in 'companies' list
    marker.add_to(phone_map) # Feeds the marker parameters into the map object


#
# Variable line thickness in line plots
import numpy as np
from matplotlib.collections import LineCollection
#import matplotlib.pyplot as plt
x = np.linspace(0,4*np.pi,10000)
y = np.cos(x)
lwidths=1+x[:-1]
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, linewidths=lwidths,color='blue')
fig,a = plt.subplots()
a.add_collection(lc)
a.set_xlim(0,4*np.pi)
a.set_ylim(-1.1,1.1)
fig.show()




#--------------- SETS -----------------------------------------
# BASICS
# Sets only store unique items. Generally made from lists
new_set = set(['cookie', 'other_cookie', 'cookie'])

# Methods
    .add() # adds single element
    .difference(other_set) # finds data present in original set but not in other_set
    .discard() # safely delete value from set
    .intersection(other_set) # returns the shared elements between 2 sets
    .pop() # remove & return a value from set
    .symmetric_difference() # all elements in exactly 1 set
    .union(other_set) # returns a new set with all unique elements from both sets
    .update() # merges in another set or list
    
    


#--------------- TUPLES -----------------------------------------    

# namedtuple - Conventionally the first character is capitalized
Eatery = namedtuple('Eatery',['name', 'location', 'park_id', 'type_name']) # (tuple_name, [fieldnames])
    # Each field is now accessible as an attribute of Eatery.fieldname
    
    



#--------------- REGULAR EXPRESSIONS -----------------------------------------    




    
    
#--------------- OUTPUT / WRITING -------------------------------  
with open('output.txt','a') as output:
    print()
    # Mode arguments
    'w' # open for writing
    'a' # open for writing, appending to the end of the file if it exists
    '+' # open a disk file for updating (reading & writing)
# CSVs
DataFrame.to_csv('output_name.csv')

#--------------- OPEN PUBLIC DATASETS -------------------------------
http://datacatalogs.org/
http://www.kaggle.com/datasets
http://data.un.org/ (UN)
http://www.data.gov (US)
http://www.europeandataportal.eu/en/
http://datasetsearch.research.google.com/
Data Asset eXchange 


(MAX) Model Asset Exchange for machine learning/deep learning models
http://developer.ibm.com/exchanges/models

https://github.com/jupyter/jupyter/wiki     Jupyter Notebook Gallery