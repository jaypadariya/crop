
# task 1: visualize a satellite image and shapefile using python
# task 2: descriptive analysis and acreage calculation

# import required libraries

from osgeo import gdal
from osgeo import ogr
import numpy as np
import math
from scipy.stats import mode
import os
import matplotlib.pyplot as plt


# task 1: # visualize a satellite image and shapefile using python

img = plt.imread(r'C:\Users\student\Desktop\tutor\Tutorials_data (1)\Tutorials_data\Tutorial1\NDVI_Mask_tiff\ndvi_mask_paddy_zero.tif')
plt.imshow(img)
plt.show()


# task 2: descriptive analysis and acreage calculation

def calculateAcreage(crop,filepath):
    print("Analysing",crop,"data!")
    print(filepath)
    dataset = gdal.Open(filepath)
    if dataset:
        band = dataset.GetRasterBand(1)
    else:
        print("Cannot open file:")
        exit()

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

    area = count * 9/1000000

    print("Total", crop ,"area:", area, "sqkm")
    print("------------------------------------")
    return(withoutNoData, area)

def img_stats(withoutNoData):
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

    return ([minval, maxval, meanval, medianval, modeval, modefreq, sdval, varianceval, rangeval, coefvariation])

shapfile_path = r'C:\Users\student\Desktop\tutor\Tutorials_data (1)\Tutorials_data\Tutorial1\Ahm_taluk_shp\Dholka.shp'
shpfile = ogr.Open(shapfile_path)
shape = shpfile.GetLayer(0)
feature = shape.GetFeature(0)
villageArea = feature.geometry().GetArea()/1000000



tif_path = r'C:\Users\student\Desktop\tutor\Tutorials_data (1)\Tutorials_data\Tutorial1\NDVI_Mask_tiff'
paddy_tif = r'ndvi_mask_paddy.tif'
wheat_tif = r'ndvi_mask_wheat.tif'

paddywithoutnodata, paddyacreage = calculateAcreage("Paddy", os.path.join(tif_path,paddy_tif))
img_stats(paddywithoutnodata)

wheatwithoutnodata, wheatacreage = calculateAcreage("Wheat", os.path.join(tif_path,wheat_tif))
img_stats(wheatwithoutnodata)

print("Area of village:" , villageArea, "sqkm")
print("------------------------------------")
print("Percentage area of Paddy:", 100 * paddyacreage / villageArea, "%")
print("Percentage area of Wheat:", 100 * wheatacreage / villageArea, "%")

#-----------------------------------------------------------------------------------------

# Data Analysis (Histogram, ZScore, Outliers)

# find ouotliers using zscore for wheat data 

def detect_outlier(data_1):
    outliers = []
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    z_score = []
    for y in data_1:
        z_score_ = (y - mean_1) / std_1
        z_score.append(z_score_)
        if np.abs(z_score_) > threshold:
            outliers.append(y)
    print("Function block")
    print("Zscore", z_score)
    print("Outliers", outliers)
    return z_score, outliers


z_score, outliers = detect_outlier(wheatwithoutnodata)


# histogram:
def histogram(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(data, color='purple')

    ax.set(xlabel='Pixels',
           ylabel='Frequency',
           title="Distribution of NDVI Mask Values");


histogram(wheatwithoutnodata)


# cumulative histograms of data:
def cumulative_histogram(data):
    mu = 200
    sigma = 25
    n_bins = 50
    # x = NDVI_mask_hist
    x = data
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',
                               cumulative=True, label='Empirical')

    # Add a line showing the expected distribution.
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    y = y.cumsum()
    y /= y[-1]

    ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

    # Overlay a reversed cumulative histogram.
    ax.hist(x, bins=bins, density=True, histtype='step', cumulative=-1,
            label='Reversed emp.')

    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('pixels')
    ax.set_ylabel('Likelihood of occurrence')

    plt.show()


cumulative_histogram(wheatwithoutnodata)


# group array into categories:
def grouping(data):
    classified = {}
    counts = {'Poor': 0, "Medium": 0, "Good": 0, "Excellent": 0}
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
    return classified, counts


categorized, count_dict = grouping(wheatwithoutnodata)

# histogram of conditional data:
plt.bar(count_dict.keys(), count_dict.values())