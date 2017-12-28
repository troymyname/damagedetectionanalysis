# ----// CALCULATE IMAGE PROPERTIES // --------------------------------------------------
# ---------------------------------------------------------------------------------------
import cv2, math, numpy as np
# ---------------------------------------------------------
from skimage.feature import greycomatrix

# ---------------------------------------------------------------------------------------
clip_limit = 2.0	# VARIABLE // Clip Limit for CLAHE
title_grid_size = (8,8)	# VARIABLE // Grid Size for CLAHE

# ---------------------------------------------------------------------------------------
# NOTE: This function receives image data and calculates properties such as mean, contrast
#	dissimilarity, homogeneity, angular second moment, energy, correlation

# ---------------------------------------------------------------------------------------



# CALCULATE PROPERTIES (MEAN, CONTRAST, DISSIMILARITY, HOMOGENEITY, ASM, ENERGY, CORRELATION, ENTROPY)
# ---------------------------------------------------------------------------------------
def CALCULATE_prop(im, Q_distance, Q_angle, levels):
	
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# OPENCV // CONVERT IMAGE TO GRAYSCALE

	clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = title_grid_size)
	im = clahe.apply(im)
	# OPENCV // PERFORM CONTRAST LIMITED ADAPTIVE HISTOGRAM EQUILIZATION
	
	mn = cv2.mean(im)
	mn = mn[0]
        # OPENCV // CALCULATE MEAN (INTENSITY) OF IMAGE AND USE SLICING TO EXTRACT MEAN VALUES

	glcm = greycomatrix(im, Q_distance, Q_angle, levels, symmetric = True, normed = True)
	# SKIMAGE // CALCULATE GREY LEVEL CO-OCCURANCE MATRIX
	# ---------------------------------------------------------	
	# 'im' represents the image [array] from which GLCM has to be calculated
	# 'Q_distance' represents the distance offset [array] between the reference and neighbor pixels
	# 'Q_angle' represents the angle (in radians) [array] between the reference and neighbor pixles
	# 'levels' represents the number of possible intensity levels with highest value of 256
	# 'symmetric'
	# 'normed'
	# ---------------------------------------------------------
	
	
	# ---------------------------------------------------------
	glcm_matrix = np.array(glcm[:,:,0,0])
	# NUMPY // EXTRACT i AND j COMPONENTS TO MAKE A 2D ARRAY FROM A 4D ARRAY OUPUT FROM greycomatrix
	
	con = 0
	dis = 0
	hom = 0
	ene = 0
	ent = 0
	# VARIABLE RESET
	
	glcm_matrix = np.nan_to_num(glcm_matrix)
		
	for i in range(levels):
		for j in range(levels):
			temp = (i - j)
			
			con = con + ((glcm_matrix[i][j])*(temp**2))
			# CALCULATE CONTRAST PROPERTY				
	
			dis = dis + ((glcm_matrix[i][j])*abs(temp))
			# CALCULATE DISSIMILARITY PROPERTY
	
			hom = hom + ((glcm_matrix[i][j])/(((temp)**2) + 1))
			# CALCULATE HOMOGENEITY PROPERTY
	
			ene = ene + ((glcm_matrix[i][j])**2)
			# CALCULATE ANGULAR SECOND MOMENT PROPERTY

			if glcm_matrix[i][j] == 0:
				ent = 0
			elif glcm_matrix[i][j] != 0:
				ent = ent + (glcm_matrix[i][j])*(np.log2(glcm_matrix[i][j]))
			# END
		# END
	# END

	ent = -ent
	# CALCULATE ENTROPY PROPERTY	
	# ---------------------------------------------------------
	
        
        # ---------------------------------------------------------
        cor = 0

	mu_i = 0
	mu_j = 0

	var_i = 0
	var_j = 0

	std_i = 0
	std_j = 0

        for i in range(levels):
               	for j in range(levels):
                       	mu_i = mu_i + (i)*(glcm_matrix[i][j])				
			# CALCULATE MEAN BASED ON REFERENCE PIXELS i
	
			mu_j = mu_j + (j)*(glcm_matrix[i][j])
			# CALCULATE MEAN BASED ON NEIGHBOR PIXELS j
           
		# END
        # END

	for i in range(levels):
               	for j in range(levels):
                       	var_i = var_i + (glcm_matrix[i][j])*(((i) - mu_i)**2)
			# CALCULATE VARIANCE BASED ON REFERENCE PIXELS i

			var_j = var_j + (glcm_matrix[i][j])*(((j) - mu_j)**2)
			# CALCULATE VARIANCE BASED ON NEIGHBOR PIXELS j
			
                # END
        # END

	std_i = math.sqrt(var_i)
	# MATH // CALCULATE STANDARD DEVIATION BASED ON REFERENCE PIXELS i

	std_j = math.sqrt(var_j)
	# MATH // CALCULATE STANDARD DEVIATION BASED ON NEIGHBOR PIXELS j

	for i in range(levels):
                for j in range(levels):
			cor = cor + (glcm_matrix[i][j])*((((i) - mu_i)*(((j) - mu_j)))/math.sqrt((var_i*var_j)))
			# MATH // CALCULATE CORRELATION PROPERTY
                # END
        # END
	# ---------------------------------------------------------


	return mn, con, dis, hom, ene, cor, ent
# ---------------------------------------------------------------------------------------


