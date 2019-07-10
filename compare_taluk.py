#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:01:30 2019

@author: ispluser
"""

from osgeo import gdal
from osgeo import ogr
import numpy as np
import math
from scipy.stats import mode, zscore

def calculateAcreage(crop,filepath):
    print("Analysing",crop,"data!")
    dataset = gdal.Open(filepath)
    band = dataset.GetRasterBand(1)

    rast_array = np.array(band.ReadAsArray())
    tcount = 0
    count = 0

    withoutNoData = []

    for row in rast_array:
        for element in row:
            tcount = tcount + 1
            if math.isnan(element) == False and element != 0.0:
                count = count+1
                withoutNoData.append(element)

    '''
    minval = band.ComputeStatistics(0)[0]
    maxval = band.ComputeStatistics(0)[1]
    meanval = band.ComputeStatistics(0)[2]
    sdval = band.ComputeStatistics(0)[3]
    '''

    minval = min(withoutNoData)
    maxval = max(withoutNoData)
    meanval = np.mean(withoutNoData)
    medianval = np.median(withoutNoData)
    modeval = float(mode(withoutNoData)[0])
    modefreq = int(mode(withoutNoData)[1])
    sdval = np.std(withoutNoData)
    varianceval = np.var(withoutNoData)
    rangeval = maxval - minval
    coefvariation = sdval * 100 / meanval
#    zs = zscore(withoutNoData)
#    threshold = 3
#    outliers = np.where((zs) > threshold)
    
    print("Min Value:", minval)
    print("Max Value:", maxval)
    print("Mean :", meanval)
    print("Median :", medianval)
    print("Mode :", modeval)
    print("Mode frequency :", modefreq)
    print("Variance :", varianceval)
    print("Standard Deviation :", sdval)
    print("Range :", rangeval)
    print("Coefficient of Variation :", coefvariation)
#    print("Zscore:", zs)
#    print("unique values:", np.unique(zs,return_counts = True))
#    print("outliers:", outliers)
    area = count * 9/1000000

    #tarea = tcount * 9/1000000
    #print("Total pixels:", tcount)
    #print("Total", crop ,"pixels:",count)
    #print("Total area:", tarea, "sqkm")

    print("Total", crop ,"area:", area, "sqkm")
    print("------------------------------------")
    return(varianceval,area)

shpfile1 = ogr.Open("/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/Ahm_taluk_shp/Dholka.shp")
shape1 = shpfile1.GetLayer(0)
feature1 = shape1.GetFeature(0)
talukArea1 = feature1.geometry().GetArea()/1000000
#
shpfile2 = ogr.Open("/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/Ahm_taluk_shp/Saanand.shp")
shape2 = shpfile2.GetLayer(0)
feature2 = shape2.GetFeature(0)
talukArea2 = feature2.geometry().GetArea()/1000000

#paddyacreage = calculateAcreage("Paddy", "/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/NDVI_Mask_tiff/dholka_wheat_tiff.tif")
variance1, wheatacreage_1 = calculateAcreage("Wheat", "/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/NDVI_Mask_tiff/dholka_wheat_tiff.tif")
variance2, wheatacreage_2 = calculateAcreage("Wheat", "/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/NDVI_Mask_tiff/sanand_wheat_tiff.tif")

taluka_variances =[variance1,variance2] 
print("Area of taluka1:" , talukArea1, "sqkm")
#print("Area of taluka2:" , talukArea2, "sqkm")
print("------------------------------------")

#print("Percentage area of Paddy:", 100 * paddyacreage / villageArea, "%")
print("Percentage area of Wheat:", 100 * wheatacreage_1 / talukArea1, "%")
print("Percentage area of Wheat:", 100 * wheatacreage_2 / talukArea2, "%")

print("The taluka with maximum variance is {}".format(np.argmax(taluka_variances)))


# compare variability across taluka :

def compare_taluka(shapefile_path_list, imagefile_path_list):
    acreage = []
    variability = []
    for i in range(len(shapefile_path_list)):
        shpfile = ogr.Open(shapefile_path_list[i])
        shape = shpfile.GetLayer(0)
        feature = shape.GetFeature(0)
        talukArea = feature.geometry().GetArea()/1000000
        variance, cropAcreage = calculateAcreage("Wheat", imagefile_path_list[i])
        acreage.append(cropAcreage)
        variability.append(variance)
    max_val , min_val = np.argmax(variability), np.argmin(variability)
    min_region_name  = imagefile_path_list[min_val].split('/')[-1].split('.')[0].split('_')[0]
    max_region_name = imagefile_path_list[max_val].split('/')[-1].split('.')[0].split('_')[0]
    return min_region_name , max_region_name
        
shape_list = ['/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/Ahm_taluk_shp/Dholka.shp','/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/Ahm_taluk_shp/Saanand.shp']
image_list = ['/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/NDVI_Mask_tiff/dholka_wheat_tiff.tif','/home/ispluser/Dimple/Wheat_Paddy_ahm/Tutorial1/NDVI_Mask_tiff/sanand_wheat_tiff.tif']


minimum_region , maximum_region = compare_taluka(shape_list, image_list)
print("The Taluka with minimum variability is {}".format(minimum_region))
print("The Taluka with maximum variability is {}".format(maximum_region))


########
def data(imagefile_path_list):
    conditioned_data = []
    for i in range(len(imagefile_path_list)):
        dataset = gdal.Open(imagefile_path_list[i])
        band = dataset.GetRasterBand(1)
        
        rast_array = np.array(band.ReadAsArray())
        tcount = 0
        count = 0
        
        withoutNoData = []
        
        for row in rast_array:
            for element in row:
                tcount = tcount + 1
                if math.isnan(element) == False and element != 0.0:
                    count = count+1
                    withoutNoData.append(element)
        conditioned_data.append(withoutNoData)              
            
    return conditioned_data
conditioned_data = data(image_list)

def grouping(data_list):
    classified_list = []
    conditioned = []
    for i in range(len(data_list)):
        data = data_list[i]
        classified = {}
        counts = {'Poor':0,"Medium": 0, "Good": 0, "Excellent": 0}
        for i in range(len(data)):
            if 0 < data[i] <= 0.25:
                classified[data[i]] = 'Poor';
                counts['Poor'] += 1
            elif 0.26 < data[i] <= 0.50:
                classified[data[i]] = 'Medium';
                counts['Medium'] += 1
            elif 0.51 < data[i] <= 0.75: 
                classified[data[i]] = 'Good';
                counts['Good'] += 1
            else:
                classified[data[i]] = 'Excellent'
                counts['Excellent'] += 1
        classified_list.append(classified)
        conditioned.append(counts)
    return classified_list,conditioned

categorized, count_dict = grouping(conditioned_data)

#%%
#histogram of conditional data:
import matplotlib.pyplot as plt

condition = count_dict[0].keys()

plt.plot(condition, count_dict[0].values(), color = 'b', label = 'Taluka1')#, width = 0.25)

plt.plot(condition, count_dict[1].values(), color = 'r', label = 'Taluka2')
plt.legend()#, width = 0.35)
plt.show()
