#!/bin/bash
"""
Script to analyse the environment for a given species over a period of 1 hour
which the user defines using time-stamps

Version:
- Reads data from ThingSpeak
- Reads the plant database
- Analyses the environment using Fuzzy Logic
- Produces expert feedback in the form of a tweet

Daniel Harrington
"""
# Import standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random
import re

# Import python's fuzzy logic package
import skfuzzy as fuzz

# Import datetime to allow for conversion from string to datetime data
from datetime import datetime

# Import literal eval to allow for a dictionary to be created from data string
from ast import literal_eval

# Import urllib to allow for access to ThingSpeak
import urllib.request

# Import csv to read database
import csv

# Import tweepy to allow tweets to be sent and define channel keys as global
# variables. Also define get_api function
import tweepy

# Change font for consistence across all plots
plt.rc('font', family='Arial') 
plt.rc('font', serif='Arial') 
plt.rc('text', usetex='false') 
# plt.rcParams.update({'font.size':22})

# Twitter API keys
cfg = { 
        'consumer_key'        : '2IbY8DGtk5XGM3fkQqzFdtdSP',
        'consumer_secret'     : 'r6JdTuOzTgXfGwbComf29EYVwzDrict0x6NzvXNoXt7tVt7xXV',
        'access_token'        : '1106072529674924035-wKNE8OcATQQ8CP6snyfPqZAzPNodH2',
        'access_token_secret' : 'nENWsldwv7oxwEsHQ0j1RABb58fGsrLnri1yU0He9W05v'
        }

# Authorise
def get_api(cfg):
  auth = tweepy.OAuthHandler(cfg['consumer_key'], cfg['consumer_secret'])
  auth.set_access_token(cfg['access_token'], cfg['access_token_secret'])
  return tweepy.API(auth)

api = get_api(cfg)


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
    Generate a trapezium Membership Function
    """
    MF = fuzz.membership.trapmf(array, vector)
    
    return MF


def GenG_MF(array, mean, sigma):
    """
    Generate a gaussian Membership Function
    """
    MF = fuzz.membership.gaussmf(array, mean, sigma)

    return MF
    
    
def GetData(channelID, readKEY, samples):
    """
    Function which reads Data from ThingSpeak channel and returns as a
    dictionary.

    Inputs: 
    channelID - string containing the channel number 
    readKEY   - API Read key 
    samples   - number of samples to import

    Outputs: 
    Data - data from channel stored as a dictionary
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


        Start_test = re.match(Start, Data['feeds'][i]['created_at'])
        Stop_test  = re.match(Stop, Data['feeds'][i]['created_at']) 

        if Start_test != None:
            StartIndex = i
        elif Stop_test != None:
            StopIndex = i
        else:
            pass

    
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


def FuzzyLogicControl(array, MF_Dict, Database, OP_MF, key, ConditionDict):
    """
    Function to interpret membership functions, implicate fuzzy rules and
    defuzzify outputs to give a crisp output.

    Inputs: 
    array   - list containing data 
    MF_Dict - dictionary containing the membership functions 
    OP_MF   - the output membership function dictionary 
    key - key to show which channel is being calculated for 
    ConditionDict -Dictionary containing species preferred levels

    Outputs: 
    Condition_Indicator_Output - list containing condition indicator
    levels for specified period.
    """
    if key == 'Lux':
        TestArray = GenArray(50000,0)
    elif key == 'Hum':
        TestArray = GenArray(100,0)
    else:
        TestArray = GenArray(50,0)

    # Define output implication array
    OP = [0]*5

    # Create array for defuzzification 
    OP_Array = GenArray(100,0)

    # Initialise condition output indicator
    Condition_Indicator_Output = []

    # For all values in array
    for i in range(len(array)):

        # Initialise Level array
        Levels = [0]*len(MF_Dict)

        # For all MFs in dictionary
        for j in range(len(MF_Dict)):

            # Generate the levels by interpreting the membership functions 
            Levels[j] = fuzz.interp_membership(TestArray, MF_Dict[str(ConditionDict[key][0][j])] , array[i])
        
        # Apply fuzzy rules
        AR = FuzzyRules(Levels, key)

        # Implicate on output
        OP[0] = np.fmin(AR[0], OP_MF['Optimum']) 
        OP[1] = np.fmin(AR[1], OP_MF['Good']) 
        OP[2] = np.fmin(AR[2], OP_MF['Okay']) 
        OP[3] = np.fmin(AR[3], OP_MF['Bad']) 
        OP[4] = np.fmin(AR[4], OP_MF['VBad']) 



        # Aggregate output
        aggregated = np.fmax(OP[0], np.fmax(OP[1], np.fmax(OP[1], np.fmax(OP[2], np.fmax(OP[3], OP[4])))))


        # Append to output
        Condition_Level = fuzz.defuzz(OP_Array, aggregated, 'centroid')
        Condition_Indicator_Output.append(Condition_Level)

    return Condition_Indicator_Output


def DatabaseRead(species, Database):
    """
    Function to read database and return preferred levels for a given species.

    Inputs:
    species  - string containing common name of species
    Database - string containing name of database to be read

    Outputs:
    Dict - dictionary containing ordered lists of preferred plant levels

    """

    # Read CSV file containing plant data
    DBase = pd.read_csv(Database)

    # Find the relevant species in the Data Base
    SpeciesList = list(DBase.loc[:,'Common Name'])

    # Find location of species of interest 
    Location = SpeciesList.index(species)
    
    # Initialise arrays
    L = []
    T = []
    H = []
    # Read the preferred levels from the database
    L_val = DBase.iloc[Location][2]
    T_val = DBase.iloc[Location][3]
    H_val = DBase.iloc[Location][4]

    # Rank preferred conditions based on database read

    # Light 
    if L_val == '1':
        L.append(['F_Sun','D_Sun','Bright','S_Shade','Shade'])
    if L_val == '2'or L_val == '1-2':
        L.append(['D_Sun','Bright','F_Sun','S_Shade','Shade'])
    if L_val == '2-3':
        L.append(['Bright','D_Sun','S_Shade','F_Sun','Shade'])
    if L_val == '3' or L_val == '2-4':
        L.append(['S_Shade','Bright','D_Sun','F_Sun','Shade'])
    if L_val == '3-4':
        L.append(['Shade','S_Shade','Bright','D_Sun','F_Sun'])

    # Temperature
    if T_val == '3':
        T.append(['Warm','Hot','Avrg','Cold','VCold'])
    if T_val == '2-3':
        T.append(['Warm','Avrg','Hot','Cold','VCold'])
    if T_val == '2':
        T.append(['Avrg','Warm','Cold','Hot','VCold'])
    if T_val == '1-2':
        T.append(['Avrg','Cold','Warm','VCold','Hot'])
    if T_val == '1':
        T.append(['Cold','Avrg','VCold','Warm','Hot'])

    # Humidity
    if H_val == '1':
        H.append(['Hi','Med','Low'])
    if H_val == '1-2':
        H.append(['Med','Hi','Low'])
    if H_val == '2':
        H.append(['Med','Low','Hi'])
    if H_val == '2-3':
        H.append(['Low','Hi','Med'])
    if H_val == '3':
        H.append(['Low','Med','Hi'])


    # Zip into a dictionary and return 
    Labels = ['Lux','Hum','Tem']
    Dict = dict(zip(Labels, (L, H, T))) 

    return Dict
     

def FuzzyRules(Levels, channel):
    """
    Function to generate the fuzzy rules based on the channel and the preferred levels for a given species.
    
    Inputs:
    Levels  - list of ranked levels preferred by species
    channel - channel string to show which channel is being evaluated

    Outputs:
    AR - list of outputs form activation rules
    """
    # For Lux channel
    # Intialise Activation Rule vector
    AR = [0]*5

    if channel == 'Lux' or channel == 'Tem':

        AR = Levels

    else:

        AR[0] = Levels[0]
        AR[1] = min(Levels[0],Levels[1])
        AR[2] = Levels[1]
        AR[3] = min(Levels[1],Levels[2])
        AR[4] = Levels[2]

    
    return AR 

def Performance(IParray):
    """
    Function to determine the performance of environment for a given species 
    """
    # Convert output to a numpy array
    OParray = np.array(IParray)
    VGood   = len(OParray[OParray>75])/len(OParray)
    Good    = len(np.where(np.logical_and(OParray>55, OParray<=75))[0])/len(OParray)
    Ok      = len(np.where(np.logical_and(OParray>45, OParray<=55))[0])/len(OParray)
    Bad     = len(np.where(np.logical_and(OParray>25, OParray<=45))[0])/len(OParray)
    VBad    = len(OParray[OParray<25])/len(OParray)

    Dict = {
            'Very Good':VGood,
            'Good'     :Good,
            'Okay '    :Ok,
            'Bad'      :Bad,
            'Very Bad' :VBad
    }

    return Dict

def PlotOutputs(Lux_array, Hum_array, Tem_array, OP_L, OP_H, OP_T, Time, ConditionDict):
    """
    Function to generate plots of outputs and save figures
    """
    # Define best levels
    LuxG = ConditionDict['Lux'][0][0]
    HumG = ConditionDict['Hum'][0][0]
    TemG = ConditionDict['Tem'][0][0]
    # Define worst levels
    LuxB = ConditionDict['Lux'][0][4]
    HumB = ConditionDict['Hum'][0][2]
    TemB = ConditionDict['Tem'][0][4]

    # Define min and max for shaded regions to show on plots
    # Light
    LuxDict = {'F_Sun':[12000,50000],'D_Sun':[6000,9000],'Bright':[2600,4400],'S_Shade':[1000,1800],'Shade':[0,400]}
    HumDict = {'Hi':[54,90],'Med':[40,50],'Low':[10,30]}
    TemDict = {'Hot':[35,50],'Warm':[27.5,30],'Avrg':[17.5,22.5],'Cold':[10,12.5],'VCold':[0,5]}

    # Plot results including red and green shaded areas for good/ bad regions
    # Light
    f, (ax1,ax2) = plt.subplots(2, sharex=True)

    # Axis 1 - Measured Lux
    ax1.plot(Time, Lux_array)
    ax1.set_title('Measured Light')
    ax1.set_ylabel('Lux')
    ax1.axhspan(LuxDict[LuxG][0], LuxDict[LuxG][1], color = 'g', alpha = 0.35)
    ax1.axhspan(LuxDict[LuxB][0], LuxDict[LuxB][1], color = '#9B7680', alpha = 0.35)
    ax1.grid()
    # Axis 2 - Condition Indicator
    ax2.plot(Time,OP_L)
    ax2.set_title('Light Output')
    ax2.set_ylabel('Condition Indicator')
    ax2.set_ylim([0,100])
    ax2.grid()
    plt.savefig('LightOutput.png') # Save figure 

    # Humidity
    f, (ax1,ax2) = plt.subplots(2, sharex=True)

    # Axis 1 - Measured Humidity
    ax1.plot(Time,Hum_array)
    ax1.set_title('Measured Humidity')
    ax1.set_ylabel('Percentage Humidity')
    ax1.axhspan(HumDict[HumG][0], HumDict[HumG][1], color = 'g', alpha = 0.35)
    ax1.axhspan(HumDict[HumB][0], HumDict[HumB][1], color = '#9B7680', alpha = 0.35)
    ax1.grid()
    # Axis 2 - Condition Indicator
    ax2.plot(Time,OP_H)
    ax2.set_title('Humidity Output')
    ax2.set_ylabel('Condition Indicator')
    ax2.set_ylim([0,100])
    ax2.grid()
    plt.savefig('HumidityOutput.png') # Save figure

    # Temperature
    f, (ax1,ax2) = plt.subplots(2, sharex=True)

    # Axis 1 - Measured Temperature
    ax1.plot(Time,Tem_array)
    ax1.set_title('Measured Temperature')
    ax1.set_ylabel('Temperature/ deg C')
    ax1.axhspan(TemDict[TemG][0], TemDict[TemG][1], color = 'g', alpha = 0.35)
    ax1.axhspan(TemDict[TemB][0], TemDict[TemB][1], color = '#9B7680', alpha = 0.35)
    ax1.grid()
    # Axis 2 - Condition Indicator
    ax2.plot(Time,OP_T)
    ax2.set_title('Temperature Output')
    ax2.set_ylabel('Condition Indicator')
    ax2.set_ylim([0,100])
    ax2.grid()
    plt.savefig('TemperatureOutput.png') # Save figure

    #plt.show()

def TweetGeneration(PerformanceList, species, ConditionDict):
    """
    Function to generate tweet text from performance output

    Inputs:
    Performance - List of performance dictionaries

    Outputs:
    Tweet - string containing tweet
    """
    # Analyse performances
    PerformanceOP = [max(PerformanceList[i], key=PerformanceList[i].get) for i in range(len(PerformanceList))]

    # Ideal conditions
    Lux = ConditionDict['Lux'][0][0]
    Hum = ConditionDict['Hum'][0][0]
    Tem = ConditionDict['Tem'][0][0]

    # Generate advice
    Advice = []

    # Lighting Advice
    if (Lux == 'F_Sun' or Lux == 'D_Sun') and (PerformanceOP[0] == 'Bad' or PerformanceOP[0] == 'Very Bad'):
        Advice.append('Your plant needs some more light; show it some love and move it to a sunnier spot like a south-facing window.')
    elif (Lux == 'S_Shade' or Lux == 'Shade') and (PerformanceOP[0] == 'Bad' or PerformanceOP[0] == 'Very Bad'):
        Advice.append('Your plant is getting too much light; try moving it away from the window.')
    
        
    # Humidity advice
    if (Hum == 'Hi') and (PerformanceOP[1] == 'Bad' or PerformanceOP[1] == 'Very Bad'):
        Advice.append('The humidity is a bit low - try misting the plant or group it with some friends!')
    elif (Hum == 'Low') and (PerformanceOP[1] == 'Bad' or PerformanceOP[1] == 'Very Bad'):
        Advice.append('The humidity is a bit high - try opening a window or buy a dehumidifier.')

        
    # Temperature advice 
    if (Tem == 'Hot' or Tem == 'Warm') and (PerformanceOP[2] == 'Bad' or PerformanceOP[2] == 'Very Bad'):
        Advice.append('Brrr your plant is cold! Show it some love and switch the heating on or move it away from the window.')
    elif (Tem == 'VCold' or Tem == 'Cold') and (PerformanceOP[2] == 'Bad' or PerformanceOP[2] == 'Very Bad'):
        Advice.append('It\'s getting hot in here! Try moving the plant away from the radiator or open the window.')
        

    # If generally bad performance tell user to relocate
    if (PerformanceOP.count('Very Bad') + PerformanceOP.count('Bad')) > 1 and Advice == []:
        AdviceString = 'Location is very bad, reconsider placement and retry analysis'
    
    # If empty environment is good
    elif Advice == []:
        AdviceString = 'Environment is well suited to species, good job!'
    
    # Otherwise select a random piece of advice
    else:
        AdviceString = random.choice(Advice)


    # Create Tweet from strings
    Tweet = 'Hourly environmental summary for your ' + species + ':\n' \
            'Lighting was ' + PerformanceOP[0] + '.\n' \
            'Humidity was ' + PerformanceOP[1] + '.\n' \
            'Temperature was ' + PerformanceOP[2] + '.\n' \
            + AdviceString
            
    return Tweet

# Main program
def main():

    # Generate Membership Functions
    LuxMF, HumMF, TemMF, OP_MF = GenMembershipFuncs()

    # Define ThingSpeak channel
    channelID = 728110
    readKEY   = 'SBZHVHAX0H0U1ZIG'
    samples   = 1800 # Equivalent of 1 hour of data (3600 secs / 20 sec intervals)
    Start     = re.compile('^2019-03-13T13:00')
    Stop      = re.compile('^2019-03-13T14:00')

    # Define species of interest and database
    species  = 'Philodendron Pertusum'
    Database = 'Plant_Database_new.csv'

    # Generate dictionary based on species
    ConditionDict = DatabaseRead(species, Database)
    
    # Get Data from ThingSpeak
    Data = GetData(channelID, readKEY, samples)
    
    # Evaluate the data and extract channel data
    Lux_array, Hum_array, Tem_array, Time = EvalData(Data, Start, Stop)

    """ Test data (comment/uncomment as required) """

    Lux_array = np.linspace(0, 50000, 1000)
    Hum_array = np.linspace(0, 100, 1000)
    Tem_array = np.linspace(0, 50, 1000)
    Time      = np.linspace(0, 10000, 1000)

    # Calculate outputs arrays for light, humidity and temperature 
    OP_L = FuzzyLogicControl(Lux_array, LuxMF, Database, OP_MF, 'Lux', ConditionDict)
    OP_H = FuzzyLogicControl(Hum_array, HumMF, Database, OP_MF, 'Hum', ConditionDict)
    OP_T = FuzzyLogicControl(Tem_array, TemMF, Database, OP_MF, 'Tem', ConditionDict)
    
    # Generate Performance indicators
    PerformanceL = Performance(OP_L)
    PerformanceH = Performance(OP_H)
    PerformanceT = Performance(OP_T)
    PerformanceList = [PerformanceL,PerformanceH,PerformanceT]

    # Print outputs
    print(PerformanceL)
    print(PerformanceH)
    print(PerformanceT)

    # Plot the outputs and save 
    PlotOutputs(Lux_array, Hum_array, Tem_array, OP_L, OP_H, OP_T, Time, ConditionDict)

    # Generate Tweet
    Tweet = TweetGeneration(PerformanceList, species, ConditionDict)

    # Append to log for historical analysis
    with open('ConditionLog.csv', 'a') as csvfile:
        fieldnames = ['Channel','T_Start', 'T_Stop','Species','PerformanceL','PerformanceH','PerformanceT']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Channel':channelID,'T_Start': Time[0], 'T_Stop': Time[-1], 'Species': species, 'PerformanceL': PerformanceL, 'PerformanceH':PerformanceH, 'PerformanceT':PerformanceT})

    # Append images of plots to Tweet
    images = ('LightOutput.png', 'HumidityOutput.png', 'TemperatureOutput.png')
    media_ids = [api.media_upload(i).media_id_string for i in images]
    api.update_status(status=Tweet, media_ids=media_ids)

if __name__ == "__main__":
    main()      


