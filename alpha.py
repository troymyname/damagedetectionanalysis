# ----// Alpha // -----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
import cv2, glob, math, matplotlib.pyplot as plt, numpy as np, pandas as pd
import sys, seaborn as sns, time
# ---------------------------------------------------------------------------------------
from numpy import linalg as la
from pandas import DataFrame
# ---------------------------------------------------------------------------------------
from CALCULATE_mean_stddev import *
from CALCULATE_prop import *
from PERFORM_PCA import *
# ---------------------------------------------------------------------------------------
Q_distance = [10]	# VARIABLE // GLCM Calculation, Inter-Pixel Distance
Q_angle = [np.pi/4]	# VARIABLE // GLCM Calculation, Offset Angle
levels = 256		# VARIABLE // Intensity Levels for GLCM Calculation

eVals_thresh = 0.5 	# VARIABLE // Eigenvalue Threshold for Selection of M Principal Components
# ---------------------------------------------------------------------------------------
standardize = 1		# SWITCH   // Standardize Data 
			# [ROW > NUMBER OF FEATURES, COLUMN > NUMBER OF SAMPLES]
# ---------------------------------------------------------------------------------------
# NOTE: This code reads Train (or Set A) image data, uses custom functions to calculate image
#	properties and to perform principal component analysis calculations, and finally
#	generates output data files and plots.
# ---------------------------------------------------------------------------------------

start = time.clock()

# CALCULATE IMAGE PROPERTIES AND STANDARDIZE VALUES FOR PRINCIPAL COMPONENT ANALYSIS ----
# ---------------------------------------------------------------------------------------
train_files = glob.glob('files/train_image/*')
# GLOB // LOCATE REFERENCE MATERIAL IMAGE FILES TO CREATE MODEL


glcm_matrix = np.array([])
# ---------------------------------------------------------
intensity = np.array([])
# ---------------------------------------------------------
contrast = np.array([])
dissimilarity = np.array([])
homogeneity = np.array([])
energy = np.array([])
correlation = np.array([])
entropy = np.array([])
# NUMPY // INITIALIZE EMPTY NUMPY ARRAYS


for i in range(len(train_files)):

        train_im = cv2.imread(train_files[i]).astype(np.uint8)
        # OPENCV // READ IMAGE AS UNASSIGNED INTEGER (WITHIN 0 TO 255 PIXEL VALUES)

	mn, con, dis, hom, ene, cor, ent = CALCULATE_prop(train_im, Q_distance, Q_angle, levels)
	# CALCULATE_prop // EXTRACT FEATURE VALUES FROM ONE IMAGE	
	
	intensity = np.append(intensity, mn)
	contrast = np.append(contrast, con)
	dissimilarity = np.append(dissimilarity, dis)
	homogeneity = np.append(homogeneity, hom)
	energy = np.append(energy, ene)
	correlation = np.append(correlation, cor)
	entropy = np.append(entropy, ent)
	# NUMPY // COLLECT GREY-LEVEL PROPERTY VALUES IN ROWS
# END


mean_intensity_data = np.mean(intensity, axis = 0)[np.newaxis]
# NUMPY // CALCULATE MEAN OF INTENSITY PROPERTY


org_texture_data = np.vstack((contrast, dissimilarity, homogeneity, energy, correlation, entropy))
# NUMPY // STACK TEXTURE PROPERTY VALUES IN SEQUENCE VERTICALLY [ROW > NUMBER OF FEATURES, COLUMN > NUMBER OF SAMPLES]


mean_texture_data = np.mean(org_texture_data, axis = 1)[np.newaxis]
mean_texture_data = np.transpose(mean_texture_data)
# NUMPY // CALCULATE MEAN OF ORGANIZED TEXTURE PROPERTY VALUES DATA AND TRANSPOSE [ROW > NUMBER OF FEATURES, COLUMN > 1]


# ---------------------------------------------------------
std_texture_data = np.zeros(org_texture_data.shape)

if standardize == 1:
	for i in range(org_texture_data.shape[0]):
		data_row_mean, data_row_stddev = CALCULATE_mean_stddev(org_texture_data[i])
		
		for j in range(org_texture_data.shape[1]):
			std_texture_data[i][j] = (org_texture_data[i][j] - data_row_mean)/float(data_row_stddev)
		# END
	# END	
# END

std_texture_data = np.nan_to_num(std_texture_data)
# NUMPY // CONVERT NOT-A-NUMBER (NAN) DATA POINTS TO ZERO
# ---------------------------------------------------------

# ---------------------------------------------------------------------------------------



# PERFORM PRINCIPAL COMPONENT ANALYSIS CALCULATIONS -------------------------------------
# ---------------------------------------------------------------------------------------
projection, eVals, eVecs, eVecs_PC = PERFORM_PCA(std_texture_data, eVals_thresh)
# PERFORM_pca // PERFORM PRINCIPAL COMPONENT ANALYSIS


eVals = eVals[np.newaxis]
eVals = np.transpose(eVals)
# NUMPY // TRANSPOSE MATRIX


eVals_sum = np.sum(eVals)
# NUMPY // CALCULATE SUM OF EIGENVALUES

temp = 0
proportion = np.array([])
# NUMPY // INITIALIZE ARRAY

for j in range(len(eVals)):
	temp = (eVals[j])/(eVals_sum)
 	proportion = np.append(proportion, temp)
	# NUMPY // CALCULATE PROPORTION VALUES
# END

cumulative = np.cumsum(proportion)
# NUMPY // CALCULATE CUMULATIVE SUM VALUES
# ---------------------------------------------------------------------------------------



# GENERATE "AVERAGE IMAGE" FOR ANOVA CALCULATIONS ---------------------------------------
# ---------------------------------------------------------------------------------------
# train_im_array = np.zeros((800, 800), np.float)
# NUMPY // INITIALIZE NUMPY ARRAY OF ZEROS

# for i in range(len(train_files)):

# 	train_im = cv2.imread(train_files[i]).astype(np.uint8)
       	# OPENCV // READ IMAGE AS UNASSIGNED INTEGER (WITHIN 0 TO 255 PIXEL VALUES)

#	train_im = cv2.cvtColor(train_im, cv2.COLOR_BGR2GRAY)
	# OPENCV // CONVERT IMAGE TO GRAYSCALE

#	train_im_array = train_im_array + train_im
# END

# train_im_avg = train_im_array/len(train_files)
# train_im_avg = np.array(np.round(train_im_avg), dtype = np.uint8)
# NUMPY // CALCULATE AVERAGE INTENSITY VALUES FROM TRAIN IMAGE DATA
# ---------------------------------------------------------------------------------------



# GENERATE OUTPUT DATA AND CONSTRUCT PLOTS ----------------------------------------------
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------
eVecs_df = pd.DataFrame(eVecs)
eVecs_df.to_csv('eigenvectors.csv')

eVals_df = pd.DataFrame(eVals)
eVals_df.to_csv('eigenvalues.csv')

eVecs_PC_df = pd.DataFrame(eVecs_PC)
eVecs_PC_df.to_csv('principal_component_eigenvectors.csv')

data_org_mean_df = pd.DataFrame(intensity)
data_org_mean_df.to_csv('organized_intensity_train_data.csv')

data_org_texture_df = pd.DataFrame(org_texture_data)
data_org_texture_df.to_csv('organized_texture_train_data.csv')

data_mean_texture_df = pd.DataFrame(mean_texture_data)
data_mean_texture_df.to_csv('mean_texture_train_data.csv')

data_mean_intensity_df = pd.DataFrame(mean_intensity_data)
data_mean_intensity_df.to_csv('mean_intensity_train_data.csv')

# proportion_df = pd.DataFrame(proportion)
# proportion_df.to_csv('proportion.csv')

# cumulative_df = pd.DataFrame(cumulative)
# cumulative_df.to_csv('cumulative.csv')

# cv2.imwrite("train_image_average.jpg", train_im_avg)
# ---------------------------------------------------------


# ---------------------------------------------------------
# plt.figure(0)
# plt.plot((1 + np.arange(len(eVals))), eVals, 'k-')
# plt.title('Scree Plot')
# plt.xlabel('Component Number')
# plt.ylabel('Eigenvalue')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# n = 1 + np.arange(len(org_texture_data[0]))

# plt.figure(1)
# plt.plot(n, intensity, 'bo')
# plt.title('Variation of Intensity in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Intensity Values of Train Image Data')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(2)
# plt.plot(n, org_texture_data[0], 'ro')
# plt.title('Variation of Contrast in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Contrast')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(3)
# plt.plot(n, org_texture_data[1], 'ro')
# plt.title('Variation of Dissimilarity in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Dissimilarity')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(4)
# plt.plot(n, org_texture_data[2], 'ro')
# plt.title('Variation of Homogeneity in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Homogeneity')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(5)
# plt.plot(n, org_texture_data[3], 'ro')
# plt.title('Variation of Energy in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Energy')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(6)
# plt.plot(n, org_texture_data[4], 'ro')
# plt.title('Variation of Correlation in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Correlation')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(7)
# plt.plot(n, org_texture_data[5], 'ro')
# plt.title('Variation of Entropy in Train Set or Set A')
# plt.xlabel('Number of Observations')
# plt.ylabel('Entropy')
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# plt.figure(8)
# plt.scatter(projection[0,:], projection[1,:], color = "green")
# plt.title('Plot of New Projected Data Using 1st & 2nd Principal Components')
# plt.xlabel("Projected Data Using 1st Principal Component")
# plt.ylabel("Projected Data Using 2nd Principal Component")
# plt.grid(b = True, which = 'major', color = 'k', linestyle = '-')
# plt.grid(b = True, which = 'minor', color = 'k', linestyle = '-')
# plt.show()

# ---------------------------------------------------------
# data_std_texture_df = DataFrame(np.transpose(std_texture_data), columns = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'Entropy'])

# plt.figure(9)
# plt.title('Pair Plot for Comparison of Standardized Texture Properties')
# sns.pairplot(data_std_texture_df, size = 3, diag_kind = "kde")
# plt.show()

# plt.figure(10)
# plt.title('Pair Plot for Comparison of Standardized Texture Properties')
# sns.pairplot(data_std_texture_df)
# plt.show()

# plt.figure(11)
# plt.title('Joint Plot for Comparison of Standardized Texture Properties')
# sns.jointplot(x = "Contrast", y = "Entropy", data = data_std_texture_df, size = 5)
# plt.show()

# plt.figure(12)
# plt.title('Box Plot for Comparison of Standardized Texture Properties')
# sns.boxplot(data_std_texture_df)
# plt.show()
# ---------------------------------------------------------

# ---------------------------------------------------------------------------------------

end = time.clock()

print "// COMPUTATION TIME:", (end - start), "seconds"


