"""
Python script using using fuzzy clustering to perform the Environmental
Classification mode of operation.

Script reads data from ThingSpeak over a given time period (inputted by user)
and performs a fuzzy clustering classification of the environment. From this
classification the plant data base is read to give a list of plants which suit
the environment. From this a Tweet can be generated to give user feedback.

Plots can also be generated (and appended to tweet) in order to display data.

Daniel Harrington
"""
# Import standard packages
import numpy as np                  # arithmetic
import pandas as pd                 # dataframes
import matplotlib.pyplot as plt     # plots
import random                       # random numbers
from random import randint          # random integer
import re                           # regular expressions for string matching 

# 3D plots
from mpl_toolkits.mplot3d import axes3d, Axes3D

import statistics # allow for statistical operations

# Import python's fuzzy logic package
import skfuzzy as fuzz

# Import datetime to allow for conversion from string to datetime data
from datetime import datetime

# Import literal eval to allow for a dictionary to be created from data string
from ast import literal_eval

# Import urllib to allow for access to ThingSpeak
import urllib.request

# Import csv to read csv file (Plant Database)
import csv

# Import tweepy to allow tweets to be sent and define channel keys as global
# variables. Also define get_afg function
import tweepy


# Change plot font to Arial for consistency
plt.rc('font', family='Arial') 
plt.rc('font', serif='Arial') 
plt.rc('text', usetex='false') 

# Define Twitter API keys in a dictionary

cfg = { 
        'consumer_key'        : '2IbY8DGtk5XGM3fkQqzFdtdSP',
        'consumer_secret'     : 'r6JdTuOzTgXfGwbComf29EYVwzDrict0x6NzvXNoXt7tVt7xXV',
        'access_token'        : '1106072529674924035-wKNE8OcATQQ8CP6snyfPqZAzPNodH2',
        'access_token_secret' : 'nENWsldwv7oxwEsHQ0j1RABb58fGsrLnri1yU0He9W05v'
        }


def get_api(cfg):
    """
    Function which authorises the Twitter API
    """
    auth = tweepy.OAuthHandler(cfg['consumer_key'], cfg['consumer_secret'])
    auth.set_access_token(cfg['access_token'], cfg['access_token_secret'])
    return tweepy.API(auth)


api = get_api(cfg) # get the API key from the cfg dictionary 


def GenMembershipFuncs():
    """
    Function to generate membership functions

    Outputs:

    LuxMF, HumMF, TemMF, OP_MF - dictionaries which define Membership Functions for
    all levels in light, humidity, temperature and Output (Condition Indictator)

    """
    # Generate arrays
    LuxArray = GenArray(50000,0)
    HumArray = GenArray(100,0)
    TemArray = GenArray(50,0)
    OP_Array = GenArray(100,0)
    
    # Create membership functions and store in dictionaries
    
    LuxMF = {
            'Shade'  :GenT_MF(LuxArray,(0,0,400,800)),
            'S_Shade':GenT_MF(LuxArray,(600,1000,1800,2200)),
            'Bright' :GenT_MF(LuxArray,(1600,2600,4400,5400)),
            'D_Sun'  :GenT_MF(LuxArray,(5000,6000,9000,10000)),
            'F_Sun'  :GenT_MF(LuxArray, (8000,12000,50000,50000))
             }
    
    HumMF = {
            'Low' :GenT_MF(HumArray,(0,0,30,40)),
            'Med' :GenG_MF(HumArray, 45, 7.5),
            'Hi'  :GenT_MF(HumArray,(50,60,100,100))
            }
    
    TemMF = {
            'VCold' :GenT_MF(TemArray,(0,0,5,10)),
            'Cold'  :GenT_MF(TemArray, (5,10,12.5,17.5)),
            'Avrg'  :GenT_MF(TemArray, (15,17.5,22.5,25)),
            'Warm'  :GenT_MF(TemArray, (22.5,27.5,30,35)),
            'Hot'   :GenT_MF(TemArray,(30, 35, 50,50))
            }
    
    OP_MF = {
            'VBad'    :GenT_MF(OP_Array,(0,0,10,20)),
            'Bad'     :GenT_MF(OP_Array, (12.5, 22.5, 32.5, 42.5)),
            'Okay'    :GenG_MF(OP_Array, 50, 7.5),
            'Good'    :GenT_MF(OP_Array, (57.5, 67.5, 77.5, 87.5)),
            'Optimum' :GenT_MF(OP_Array, (80, 90, 100, 100))
            }
    
    return LuxMF, HumMF, TemMF, OP_MF
    
    
def GenArray(MaxVal,MinVal):
    """
    Function to generate a numpy array

    Inputs:
    - MaxVal/ MinVal - Maximum and minimum values for array/ int

    Outputs:
    - array - numpy array 
    """  
    array = np.linspace(MinVal,MaxVal,num = (10*MaxVal)+1)
    
    return array


def GenT_MF(array,vector):
    """
    Generate a trapezium Membership Function.

    Inputs:
    - array - array to be used
    - vector- vector of values which define shape of trapezium

    Outputs:
    - MF - membership function/ array
    """
    MF = fuzz.membership.trapmf(array, vector)
    
    return MF


def GenG_MF(array, mean, sigma):
    """
    Generate a gaussian Membership Function

    Inputs:
    - array - array to be used
    - mean  - mean value/ float
    - sigma - std dev of distribution

    Outputs:
    - MF - membership function/ array
    """
    MF = fuzz.membership.gaussmf(array, mean, sigma)

    return MF
    
    
def GetData(channelID, readKEY, samples):
    """
    Function which reads Data from ThingSpeak channel

    Inputs:
    - channelID - Channel ID number/ int
    - readKEY   - API read key/ str
    - samples   - number of samples to be read

    Outputs:
    - Data - Dictionary of data
    """
    # Define the base URL
    baseURL = 'https://api.thingspeak.com/channels/'+str(channelID)+'/feeds.json?api_key='+ readKEY +'&results='
    
    # Request the URL for data and read - reads as a binary array which is
    # difficult to work with
    with urllib.request.urlopen(baseURL + str(samples)) as TS_Data:
        DataString = TS_Data.read()
        
    # Decode using the utf-8 encoding to get a string and then evaluate
    # literally to get a python dictionary 
    string    = str(DataString,'utf-8')
    Data      = literal_eval(string)
    
    return Data

def EvalData(Data, Start, Stop):
    """
    Evaluate the data set and separate into time, lux, humidity and temperature
    channels

    Inputs:
    - Data - dictionary of data

    Outputs:
    - Lux_array, Hum_array, Tem_array - arrays (lists) containing lux, humidity
      and temp levels.
    - Time - list of time data/ datetime
    """
    
    # Initialise variable arrays
    Lux_array = []
    Hum_array = []
    Tem_array = []
    Time      = []

    # Find start and stop times
    for i in range(len(Data['feeds'])):

        # Use a regular expression match to find the start and stop times 
        Start_test = re.match(Start, Data['feeds'][i]['created_at'])
        Stop_test  = re.match(Stop, Data['feeds'][i]['created_at']) 

        # if there is a match then set a Start/ Stop index of that loop
        if Start_test != None:
            StartIndex = i
        elif Stop_test != None:
            StopIndex = i
        else:
            pass

    # Loop between start and stop indexes
    for i in range(StartIndex,StopIndex):
        
        # Access data and assign values
        LoopData = Data['feeds'][i]
        Tem      = float(LoopData['field1'])
        Hum      = float(LoopData['field2'])
        Lux      = float(LoopData['field3'])
        
        # Access the string with time information, convert to datetime and
        # append to Time
        T_strng = LoopData['created_at']
        TStamp = datetime.strptime(T_strng, '%Y-%m-%dT%H:%M:%SZ')
        Time.append(TStamp)

        # Create arrays
        Lux_array.append(Lux)
        Hum_array.append(Hum)
        Tem_array.append(Tem)
        
    return Lux_array, Hum_array, Tem_array, Time


def GenClusterData(MF_Dict, key):
    """
    Function to generate cluster data for use in fuzzy clustering algorithm.
    Algorithm uses membership functions to generate a probability distribution
    of points i.e. a point with a Membership value of 1 has a higher probability
    of being kept than a value with a value of 0.5.

    Inputs:
    - MF_Dict - Membership function dictionary containing all MFs for each channel
    - key     - key to determine which channel is being used/ str

    Outputs:
    - CombinedArray - array of data points for the cluster/ np.array 
    """
    # Change array based on key used from each channel
    if key == 'Lux':
        TestArray = GenArray(50000,0)
    elif key == 'Hum':
        TestArray = GenArray(100,0)
    else:
        TestArray = GenArray(50,0)

    # Generate the list of keys from the Membership Function Dict
    keys = list(MF_Dict)

    # Initialise an array which will store all results
    CombinedArray = []

    # Loop for all membership functions
    for i in range(len(MF_Dict)):

        # Initialise Output data array
        DataArray = []
        # Get the key for the loop
        LoopKey = keys[i]
        # Get MF array
        MFarray = MF_Dict[LoopKey]
        # Initialise the range list and reset the flag
        Range = []
        flag = 0
        # Define the loop range which will be iterated over
        Loop = range(len(MFarray))

        # Loop to find range of MF - i.e. the point at which it becomes non-zero
        # and the point when it returns to 0
        for j in Loop:
            # If flag is 0 - looking for the first non-zero term and append
            if flag == 0:
                if MFarray[j] > 0:
                    Range.append(TestArray[j])
                    flag = 1 # increment flag

            # If the flag is 1 then non-zero term has been found - now looking
            # for either the next 0 term or the end of the array (if last point
            # in MF has been reached)
            elif flag == 1:
                if MFarray[j] == 0 or j == Loop[-1]:
                    Range.append(TestArray[j])
                    flag = 2 # increment to break loop 
            else:
                break
            
      
  
        # Loop to generate 10000 random data points
        while len(DataArray) < 10000:
            # Generate a random number in the range 
            DataRand = random.uniform(Range[0],Range[1])

            # Find corresponding value in the membership function 
            Level = fuzz.interp_membership(TestArray, MF_Dict[LoopKey] , DataRand)

            # Assert whether level is greater than a random number between 0 and
            # 1 and append - higher Level gives a greater probability of being
            # kept.
            if Level > random.uniform(0,1):
                DataArray.append(DataRand)

        # Append the array to the overall results array
        CombinedArray.append(DataArray)
    
    # Convert to a numpy array 
    CombinedArray = np.asarray(CombinedArray)

    return CombinedArray

def CombineCluster(LuxData,HumData,TemData):
    """
    Function to combine cluster data into a single output

    Inputs:
    - LuxData,HumData,TemData - lists containing channel data

    Outputs:
    - ClusterData - list of np arrays
    """
    # Initiliase Cluster data
    ClusterData = []

    # For each cluster in Lux, Hum and Tem
    for x in range(len(LuxData)):
        for y in range(len(HumData)):
            for z in range(len(TemData)):
                # create np array with a list containing channel values
                Output = np.array([LuxData[x],HumData[y],TemData[z]])
                # append to cluster data
                ClusterData.append(Output)
                
    return ClusterData

def GenCentres(ClusterData):
    """
    Function to generate cluster centres from cluster data

    Inputs:
    - ClusterData - list of np arrays

    Outputs:
    - centres - np array containing centre locations 
    - 
    """
    # Intialise centre array
    centres = []

    # Loop for all clusters, find cluster centre using fuzzy c-means and append
    for z  in range(len(ClusterData)):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(ClusterData[z], 1, 2, error=0.005, maxiter=1000, init=None)
        centres.append(cntr)
    
   # Convert centres to a np array which can be used by fuzzy c-means predict
    for i in range(len(centres)):
        centres[i] = centres[i][0]

    centres = np.asarray(centres)

    return centres

def GenSearchArray():
    """
    Function to return an array of L/H/T values to represent cluster centres
    
    Outputs:
    - search - list of lists representing the 75 cluster centres e.g. [4,1,3]
    """
    search = []
    for x in range(5):
        for y in range(3):
            for z in range(5):
                search.append([x,y,z])
    return search

def PlantDF(search, cluster_membership, DBase):
    """
    Function to return data frame slice of plants which suit the given cluster
    membership calculated.

    Inputs:
    - search - list of lists representing the 75 cluster centres
    - cluster_membership - the value of the cluster membership based on input data
    - DBase - the database of plants/ pandas dataframe

    Outputs:
    - Result - slice of dataframe containing plants which suit cluster 
    """
    # Obtain L/H/T values based on the cluster membership obtained
    Search = [0]*3
    for i in range(3):
        Search[i] = search[cluster_membership][i]


    # Generate matching strings for search criteria in the dataframe
    
    String = [0]*3 # initialise string list 

    # Light 
    if Search[0] == 0:
        String[0] = ['3-4']
    elif Search[0] == 1:
        String[0] = ['3','2-4']
    elif Search[0] == 2:
        String[0] = ['2-3']
    elif Search[0] == 3:
        String[0] = ['2','1-3']
    else:
        String[0] = ['1','1-2']

    # Humidity
    if Search[1] == 0:
        String[1] = ['3','2-3']
    elif Search[1] == 1:
        String[1] = ['2']
    else:
        String[1] = ['1','1-2']

    # Temperature
    if Search[2] == 0:
        String[2] = ['1']
    elif Search[2] == 1:
        String[2] = ['1-2']
    elif Search[2] == 2:
        String[2] = ['2']
    elif Search[2] == 3:
        String[2] = ['2-3']
    else:
        String[2] = ['3']


    # Initialise Find  - list to find matches for each channel
    # Initialise Index - list to get the index values of the matches
    Find =  [0]*3
    Index = [0]*3

    # Loop for each channel 
    for j in range(3):
        if j == 0:
            col = 'L'
        elif j == 1:
            col = 'H'
        else:
            col = 'T'

        # Find matches and indexes of matches
        Find[j]  = DBase[col].isin(String[j])
        Index[j] = [i for i, val in enumerate(Find[j]) if val] 

    # Find where all 3 channel matches intersect 
    Intersection = list(set(Index[0]) & set(Index[1]) & set(Index[2]))

    # Get dataframe slice for the matches and return
    Result = DBase.iloc[Intersection,:]

    # Reset the index to obtain new dataframe with ordered index
    Result = Result.reset_index(drop=True)

    return Result

def GenClusterTest(Lux_array,Hum_array,Tem_array):
    """
    Function to generate cluster data to be classified

    Inputs:
    - Lux_array,Hum_array,Tem_array - arrays containing data from each channel

    Outputs:
    - ClusterTest - array of lists containing data points
    """
    # Intitialise list 
    ClusterTest = []

    # For all points 
    for x in range(len(Lux_array)):
        # Create loop array
        LoopArray = np.array([Lux_array[x],Hum_array[x],Tem_array[x]])
        # Append to test cluster
        ClusterTest.append(LoopArray)

    # Save as a np array
    ClusterTest = np.asarray(ClusterTest)

    return ClusterTest

def TweetGenerator(Result):
    """
    Function to generate a tweet based on the result of the identfied plants

    Inputs:
    - Result - data frame slice containing potential plants

    Outputs:
    - Tweet - string containing a tweet

    """
    # If the dataframe of results is greater than 1 pick a random index, if it
    # has length 1 then set the index to 0, otherwise no plants suit the
    # environment.

    if len(Result) > 1:
        Index = randint(0,len(Result)-1)
    elif len(Result) == 1:
        Index = 0
    else:
        Tweet = 'No plants suit this environment!'
        return Tweet
        
    # Get the common and botanical name 
    CommonName = str(Result['Common Name'][Index])
    Botanical  = str(Result['Botanical Name'][Index])

    # Define linguistic descriptions of the levels for all three channels and
    # watering advice.

    Light = {
            '1'  : 'very sunny',
            '1-2': 'sunny',
            '1-3': 'high light',
            '2'  : 'high to medium light',
            '2-3': 'medium light',
            '2-4': 'medium to low light',
            '3'  : 'low light',
            '3-4': 'shaded'
            }

    Humidity = {
            '3'  : 'very high humidity',
            '2-3': 'high humidity',
            '2'  : 'medium humidity',
            '1-2': 'medium to low humidity',
            '1'  : 'low humidity'
            }
    
    Temperature = {
            '3'  : 'very warm temperature',
            '2-3': 'warm temperature',
            '2'  : 'medium temperature',
            '1-2': 'medium to low temperature',
            '1'  : 'low temperature'
            }

    Water = {
            '3'  : 'This plant is very hardy and does not need much water. Soil can become dry (but don\t forget about it completely!).',
            '2-3': 'Water occasionally, but don\t worry if the soil gets a bit dry.',
            '2'  : 'Allow the surface of the soil mix to dry before watering.',
            '1-2': 'This plant needs occasional watering. Ensure soil surface does not become too dry.',
            '1'  : 'This plant likes a lot of water so keep the soil mix moist.'
    }

    Lstrng = Light[Result['L'][Index]]
    Hstrng = Humidity[Result['H'][Index]]
    Tstrng = Temperature[Result['T'][Index]]
    Wstrng = Water[Result['W'][Index]]

    Tweet = 'This environment would suit a ' + CommonName + ' (' + Botanical + ')' \
    ' very well as it is a ' + Lstrng + ' spot with ' + Hstrng + ' and a ' + \
    Tstrng + '. ' + Wstrng

    return Tweet

def PlotResults(centres, cluster_membership, Lux_array, Hum_array, Tem_array):
    """
    Function to plot results in 3D space

    Inputs:
    - centres            - cluster centres
    - cluster_membership - the calculated membership/ classification
    - Lux_array,Hum_array,Tem_array - arrays containing data from each channel

    """
    
    # Define font size and intialise figure
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure()
    ax = Axes3D(fig)

    # Plot predicted centre in red and all other centres in blue
    for i in range(75):
        if i == cluster_membership:
            ax.scatter(centres[i][0],centres[i][1],centres[i][2],s=100, color = 'r')

        else:
            ax.scatter(centres[i][0],centres[i][1],centres[i][2],s=20, color = 'b')

    # Overlay the data from the channel in green 
    ax.scatter(Lux_array,Hum_array,Tem_array,color = 'g')

    ax.set_title('Environmental Classification')
    plt.show()
    plt.savefig('FuzzyCluster.png') # save figure locally


# Main program
def main():

    
    # Generate Membership Functions
    LuxMF, HumMF, TemMF, OP_MF = GenMembershipFuncs()

    # Define ThingSpeak channel variables including channel ID, API key and
    # Start/ Stop times.

    channelID = 736819
    readKEY   = '0NFUKVRECOGUZFB8'
    samples   = 1000
    Start     = re.compile('^2019-03-23T08:45')
    Stop      = re.compile('^2019-03-23T09:19')

    # Read plant database into workspace as a pandas dataframe
    DBase = pd.read_csv('Plant_Database_new.csv')

    # Get Data from ThingSpeak
    Data = GetData(channelID, readKEY, samples)
    
    # Evaluate the data and extract channel data
    Lux_array, Hum_array, Tem_array, Time = EvalData(Data, Start, Stop)

    # Generate random data points for each membership function 
    LuxData = GenClusterData(LuxMF, 'Lux')
    HumData = GenClusterData(HumMF, 'Hum')
    TemData = GenClusterData(TemMF, 'Tem')

    # Create cluster data for each type of environment
    ClusterData = CombineCluster(LuxData,HumData,TemData)

    # Calculate cluster centres
    centres = GenCentres(ClusterData)

    # Generate a cluster to be classified from the input data 
    ClusterTest = GenClusterTest(Lux_array,Hum_array,Tem_array)

    # Perform prediction using the cluster centres and the test cluster to
    # obtain a cluster membership value (i.e. predicted class)
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(ClusterTest.T, centres, 2, error=0.005, maxiter=1000)
    cluster_membership = statistics.mode(np.argmax(u, axis= 0))

    # Generate a search array that is used to identify clusters
    search = GenSearchArray()

    # Produce a list of plants which would suit the environment 
    Result  = PlantDF(search, cluster_membership, DBase)

    # Export result to excel database
    Result.to_excel('Results.xlsx')

    # Generate a tweet string based on the result and print 
    Tweet = TweetGenerator(Result)
    print(Tweet)

    # Connect to twitter and tweet
    status = api.update_status(status=Tweet) 

    # Plot result
    PlotResults(centres, cluster_membership, Lux_array, Hum_array, Tem_array)
    


if __name__ == "__main__":
    main()      


