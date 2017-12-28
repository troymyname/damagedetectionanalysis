# ----// Beta //-------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
import cv2, glob, math, matplotlib.pyplot as plt, matplotlib.font_manager, numpy as np, pandas as pd
import sys, seaborn as sns, time
# ---------------------------------------------------------------------------------------
from pandas import DataFrame
from sklearn import svm
# ---------------------------------------------------------------------------------------
from CALCULATE_mean_stddev import *
from CALCULATE_prop import *
from APPLY_grid import *
# ---------------------------------------------------------------------------------------
Q_distance = [10]	# VARIABLE // GLCM Calculation, Inter-Pixel Distance
Q_angle = [np.pi/4]	# VARIABLE // GLCM Calculation, Offset Angle
levels = 256		# VARIABLE // Intensity Levels for GLCM Calculation

cell_dim = 400          # VARIABLE // Vertical/Horizontal Pixel Dimension of Cropped Test Image Cells/Blocks

# ---------------------------------------------------------------------------------------
OCSVM_kernel = 'rbf'	# KERNEL   // Kernel Type to be Used In Algorithm
			#          // TYPES: 'linear', 'poly', 'rbf', 'sigmoid'

OCSVM_nu = 0.01		# VARIABLE // An Upper Bound on the Fraction of Training Errors and
			#	   // a Lower Bound on the Fraction of Support Vectors in the
			#	   // Interval (0, 1]

OCSVM_degree = 2	# VARIABLE // Degree of polynomial kernel function 'poly'
OCSVM_gamma = 0.01	# VARIABLE // Kernel Coefficient for 'rbf', 'poly', 'sigmoid'
# ---------------------------------------------------------------------------------------
standardize = 1		# SWITCH   // Standardize Data 
			# [ROW > NUMBER OF FEATURES, COLUMN > NUMBER OF SAMPLES]
# ---------------------------------------------------------------------------------------
d = 10			# VARIABLE // Diameter of Each Pixel Neighbor that is Used During Filtering
sigmaColor = 150	# VARIABLE // Filter Sigma in Color Space, Large Value of the Parameter
			#          // Means Farther Colors within the Pixel Neighborhood will be
			#	   // Mixed Together Resulting in Larger Areas of Semi-Equal Color
sigmaSpace = 150	# VARIABLE // Filter Sigma in the Coordinate Space, Large Value of the Parameter
			#          // Means Farther Pixels will Influence Each Other as long as Their
			#          // Colors are Close Enough 
# ---------------------------------------------------------------------------------------
thresh = 75		# VARIABLE // Threshold Value for Binary Thresholding
maxValue = 125		# VARIABLE // Maximum Value for Binary Thresholding
# ---------------------------------------------------------------------------------------
# NOTE: This code reads output from STEP_ABC.py, and performs calculations to classify blocks
#	in the image from Set B as match or mismatch. To accomplish this task, the code implements
#	a One-Class Support Vector Machine (or One-Class SVM). Based on the result, the code
#	removes information from the block of the image that the classifier predicts as a
#	mismatch. Finally, the code detects and predicts a quantative value for the damage.
# ---------------------------------------------------------------------------------------



start = time.clock()



# READ OUTPUT DATA FROM BUILD MODEL FUNCTION, AND SELECT NUMBER OF EIGENVECTORS ---------
# ---------------------------------------------------------------------------------------
eVecs = pd.read_csv('eigenvectors.csv')
eVecs = np.array(eVecs)
eVecs = eVecs[:,1:(eVecs.shape[1])]

eVals = pd.read_csv('eigenvalues.csv')
eVals = np.array(eVals)
eVals = eVals[:,1:(eVals.shape[1])]

eVecs_PC = pd.read_csv('principal_component_eigenvectors.csv')
eVecs_PC = np.array(eVecs_PC)
eVecs_PC = eVecs_PC[:,1:(eVecs_PC.shape[1])]

org_int_train_data = pd.read_csv('organized_intensity_train_data.csv')
org_int_train_data = np.array(org_int_train_data)
org_int_train_data = org_int_train_data[:,1:(org_int_train_data.shape[1])]

org_tex_train_data = pd.read_csv('organized_texture_train_data.csv')
org_tex_train_data = np.array(org_tex_train_data)
org_tex_train_data = org_tex_train_data[:,1:(org_tex_train_data.shape[1])]

mean_tex_train_data = pd.read_csv('mean_texture_train_data.csv')
mean_tex_train_data = np.array(mean_tex_train_data)
mean_tex_train_data = mean_tex_train_data[:,1:(mean_tex_train_data.shape[1])]

mean_int_train_data = pd.read_csv('mean_intensity_train_data.csv')
mean_int_train_data = np.array(mean_int_train_data)
mean_int_train_data = mean_int_train_data[:,1:(mean_int_train_data.shape[1])]
# ---------------------------------------------------------------------------------------



# CALCULATE TRAIN WEIGHT VECTOR (OR TRAIN "PATTERN" VECTOR) AND STANDARDIZE -------------
# ---------------------------------------------------------------------------------------
omega_train = (np.transpose(eVecs_PC)).dot(org_tex_train_data - mean_tex_train_data)


omega_train_mean = np.zeros([omega_train.shape[0], 1])
omega_train_stddev = np.zeros([omega_train.shape[0], 1])
# ---------------------------------------------------------
std_omega_train = np.zeros(omega_train.shape)

if standardize == 1:
	for i in range(omega_train.shape[0]):
		omega_train_mean[i], omega_train_stddev[i] = CALCULATE_mean_stddev(omega_train[i])
		
		for j in range(omega_train.shape[1]):
			std_omega_train[i][j] = (omega_train[i][j] - omega_train_mean[i][0])/float(omega_train_stddev[i][0])
		# END
	# END	
# END

std_omega_train = np.nan_to_num(std_omega_train)
std_omega_train = np.transpose(std_omega_train)
# NUMPY // CONVERT NOT-A-NUMBER (NAN) DATA POINTS TO ZERO AND TRANSPOSE
# ---------------------------------------------------------

# ---------------------------------------------------------------------------------------



# TRAIN WITH ONE-CLASS SVM CLASSIFIER ---------------------------------------------------
# ---------------------------------------------------------------------------------------
tex_classifier = svm.OneClassSVM(kernel = OCSVM_kernel, nu = OCSVM_nu, degree = OCSVM_degree, gamma = OCSVM_gamma)
tex_classifier.fit(std_omega_train)
# SKLEARN // DEFINE A TYPE OF ONE-CLASS SUPPORT VECTOR MACHINE FOR TEXTURE CLASSIFICATION

tex_train_predict = tex_classifier.predict(std_omega_train)
tex_train_error = tex_train_predict[tex_train_predict == -1].size
# SKLEARN // PERFORM REGRESSION ON TRAIN SAMPLES AND FIND ERRORS

# --

int_train_mean, int_train_stddev = CALCULATE_mean_stddev(org_int_train_data)
std_int_train_data = np.divide((org_int_train_data - int_train_mean), float(int_train_stddev))
# CALCULATE_mean_stddev // STANDARDIZE ...

int_classifier = svm.OneClassSVM(kernel = OCSVM_kernel, nu = OCSVM_nu, degree = OCSVM_degree, gamma = OCSVM_gamma)
int_classifier.fit(std_int_train_data)
# SKLEARN // DEFINE A TYPE OF ONE-CLASS SUPPORT VECTOR MACHINE FOR TEXTURE CLASSIFICATION

int_train_predict = int_classifier.predict(std_int_train_data)
int_train_error = int_train_predict[int_train_predict == -1].size
# SKLEARN // PERFORM REGRESSION ON TRAIN SAMPLES AND FIND ERRORS
# ---------------------------------------------------------------------------------------



# CLASSIFY IMAGE CELLS/BLOCKS AS MATCH OR MISMATCH --------------------------------------
# ---------------------------------------------------------------------------------------
test_files = glob.glob('files/test_image/*')
# GLOB // LOCATE TEST IMAGE FILES


for i in range(len(test_files)):
	
	test_im = cv2.imread(test_files[i]).astype(np.uint8)
        # OPENCV // READ IMAGE AS UNASSIGNED INTEGER (WITHIN 0 TO 255 PIXEL VALUES)

	cell_count, grid_coordinates = APPLY_grid(test_im.shape[0], test_im.shape[1], cell_dim, cell_dim)
	# APPLY_grid // CALCULATE NUMBER OF CELLS AND GRID COORDINATES

	glcm_matrix = np.array([])
	mn = np.array([])
	con = np.array([])
	dis = np.array([])
	hom = np.array([])
	asm = np.array([])
	ene = np.array([])
	cor = np.array([])
	ent = np.array([])
	# NUMPY // INITIALIZE EMPTY NUMPY ARRAYS

	elimination = np.ones([cell_count, 1])
	# NUMPY // INITIALIZE ARRAY WITH ONES

	for i in range(0,cell_count):
        	w1 = grid_coordinates[i][0]
        	h1 = grid_coordinates[i][1]
        	w2 = grid_coordinates[i][2]
        	h2 = grid_coordinates[i][3]

        	test_im_cropped = test_im[h1:h2,w1:w2]

        	mn, con, dis, hom, ene, cor, ent = CALCULATE_prop(test_im_cropped, Q_distance, Q_angle, levels)
		# CALCULATE_prop // EXTRACT PROPERTY VALUES FROM ONE IMAGE
		
        	im_tex_test_data = np.vstack((con, dis, hom, ene, cor, ent))
		# NUMPY // STACK TEXTURE VALUES IN SEQUENCE VERTICALLY [ROW > NUMBER OF FEATURES, COLUMN > 1]

		omega_test = ((np.transpose(eVecs_PC)).dot(im_tex_test_data - mean_tex_train_data))
		omega_test = np.reshape(omega_test, (len(omega_test),1))
		std_omega_test = np.divide((omega_test - omega_train_mean), omega_train_stddev)
		
		std_omega_test = np.transpose(std_omega_test)
		# NUMPY // CALCULATE TEST WEIGHT VECTOR, RESHAPE THE VECTOR AND STANDARDIZE THE VECTOR
	
		std_mn = (mn - int_train_mean)/float(int_train_stddev)
		# CALCULATE_mean_stddev // STANDARDIZE ...

		# ---------------------------------------------------------

		tex_predict = tex_classifier.predict(std_omega_test)
		int_predict = int_classifier.predict(std_mn)
		
		if tex_predict == -1 or int_predict == -1:
			elimination[i][0] = 0
		# END
	# END
# END

# ---------------------------------------------------------------------------------------



# DETECT AND QUANTIFY DAMAGE ------------------------------------------------------------
# ---------------------------------------------------------------------------------------
damage_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
# OPENCV // CONVERT IMAGE TO GRAYSCALE

damage_im = cv2.bilateralFilter(damage_im, d, sigmaColor, sigmaSpace)
# OPENCV // APPLY BILATERAL FILETERING

thresh, damage_im = cv2.threshold(damage_im, thresh, maxValue, cv2.THRESH_BINARY)
# OPENCV // APPLY BINARY THRESHOLDING


for i in range(0,cell_count):
	if elimination[i][0] == 0:
		w1 = grid_coordinates[i][0]
        	h1 = grid_coordinates[i][1]
        	w2 = grid_coordinates[i][2]
        	h2 = grid_coordinates[i][3]

		damage_im[h1:h2,w1:w2] = maxValue
	# END
# END


cv2.imwrite('results/damage_image.jpg', damage_im)
# OPENCV // GENERATE IMAGE

damage_quantity = ((damage_im.shape[0])*(damage_im.shape[1]) - cv2.countNonZero(damage_im))
damage_percent = (damage_quantity/float((damage_im.shape[0])*(damage_im.shape[1])))*100
# OPENCV // CALUCLATE DAMAGE PERCENTAGE

# ---------------------------------------------------------------------------------------



end = time.clock()



# PRINT RESULTS -------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
print "----------------------------------------------------------------------------------"
print "COMPUTATION TIME: ", (end - start), "seconds."
print "----------------------------------------------------------------------------------"
print "RESULTS:"
print elimination[elimination == 0].size, "out of", cell_count, "blocks were classified as mismatch." 
print "Image has approximately %0.2f" %damage_percent,"% damage."
print "----------------------------------------------------------------------------------"
# ---------------------------------------------------------------------------------------



