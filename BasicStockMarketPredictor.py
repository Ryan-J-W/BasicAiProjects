#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#open formatted data flile and save it in the variable dataFile
dataFile = open('DataPoints_StockMarket_basic.txt')

#create lists to store each data point in the x and y axis
dataPntNbs = []
dataPointVals = []

#read the first line of the data file
line = dataFile.readline()


#save each data point in the data file in dataPntNbs and its corresponding value in dataPointVals
while line:
    dataPntNbs.append(line.split()[0])  #splits the line between the first and second part based on the space.  Take the first part(data nmbr and save it to dataPntNbs
    dataPointVals.append(line.split()[1]) #saves second part to dataPointVals
    line = dataFile.readline() #read the next line of the data file


#converts each data point from string to int
for i in range(len(dataPntNbs)):
    dataPntNbs[i] = int(dataPntNbs[i])
    dataPointVals[i] = float(dataPointVals[i])
    
#print data arrays if needed to check for formatting issues
#print(dataPntNbs)
#print(dataPointVals)


#convert data arrays to numpy arrays and save them as the x and y axis accordingly
x_axis = np.array(dataPntNbs) #the data point numbers as time passes
y_axis = np.array(dataPointVals) #the data point value of each time frame


#plot the arrays on a matplotlib graph for visualization
plt.plot(x_axis, y_axis, 'o') 

#sets m and b(the coefficients) to best fit the data points
m,b = np.polyfit(x_axis, y_axis, 1)

#plots the new line that fits the data onto the graph
plt.plot(x_axis, x_axis*m + b)

#display the graph and line for the user 
plt.show()
