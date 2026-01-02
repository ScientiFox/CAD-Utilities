###
#
# Script to validate batch NME checker on known files
#
###

#Standards
import math,time,random

#File handling
import glob

#Load known good and bad models
f_good = glob.glob("Good Models//*.stl")
f_bad = glob.glob("Bad//*.stl")

#Grab marked files
f_marked_bad = glob.glob("known_bad//*.stl")
f_marked_good = glob.glob("to_examine//*.stl")

#Create file lists
f_good = [fil.split("\\")[-1] for fil in f_good]
f_bad = [fil.split("\\")[-1] for fil in f_bad]
f_marked_good = [fil.split("\\")[-1] for fil in f_marked_good]
f_marked_bad = [fil.split("\\")[-1] for fil in f_marked_bad]

#Read in file report
report = open('report.txt','r')
rep_data = report.readlines()
report.close()

#Grab report data
rep_data = [l.split(":") for l in rep_data]
rep_data = [[l[0][:-1],int(l[1]),int(l[2])] for l in rep_data]

#Lists of different classes
True_Positive = []
True_Negative = []
False_Positive = []
False_Negative = []

#Looping over data
for dat in rep_data:
    fil = dat[0] #Get sample file

    #If good
    if fil in f_good:
        #Annotate if correctly or incorrectly sorted
        if fil in f_marked_bad:
            False_Negative+=[(dat[1],dat[2])]
        if fil in f_marked_good:
            True_Positive+=[(dat[1],dat[2])]

    #If bad, do the same
    if fil in f_bad:
        if fil in f_marked_bad:
            True_Negative+=[(dat[1],dat[2])]
        if fil in f_marked_good:
            False_Positive+=[(dat[1],dat[2])]

#Report validation records
print("TP: ",len(True_Positive))
print("TN: ",len(True_Negative))
print("FP: ",len(False_Positive))
print("Fn: ",len(False_Negative))
print("&&&&&&&&")

#Output all files by class for further investigation
print("True Positive:")
for a in True_Positive:
    print(a)
print("--------")
print("True Negative:")
for a in True_Negative:
    print(a)
print("--------")
print("False Positive:")
for a in False_Positive:
    print(a)
print("--------")
print("False Negative:")
for a in False_Negative:
    print(a)
print("--------")



