import csv
import numpy as np
import pandas as pd


#loads the raw data
def parser(path):
    return pd.read_csv(path, sep=",",quotechar='"', engine="python", encoding="latin-1")

#removes unused rows and special char
def cleanup(array):
    frame = pd.DataFrame(array)
    frame = frame[["v1","v2"]]
    #will remove special char one by one
    frame.v2.apply(lambda txt: 
                           (''.join([char for char in txt if char.isalnum()])).lower()
                           )
    return frame 

