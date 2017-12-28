# ----// GAMMA // -----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
import glob, numpy as np, sys, matplotlib.pyplot as plt, pandas as pd, PIL, seaborn as sns

from numpy import linalg as la
from pandas import DataFrame
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree, NearestNeighbors
# ---------------------------------------------------------------------------------------
n_neighbors = 10	# // NUMBER OF NEIGHBORS FOR NEAREST-NEIGHBORS ALGORITHM

curv_thresh = 2		# // CURVATURE THRESHOLD FOR DIVISION BETWEEN POSITIVE, NEGATIVE AND FLAT SURFACE CURVATURE VALUES
lumi_thresh = 245	# // LUMINANCE THRESHOLD FOR SELECTION OF DARK PIXELS


PF_thresh = 0.003	# // POSITIVE-TO-FLAT RATIO THRESHOLD
NF_thresh = 0.003	# // NEGATIVE-TO-FLAT RATIO THRESHOLD
FA_thresh = 0.995	# // FLAT-TO-ALL RATIO THRESHOLD

PHIp_thresh = 300	# // THRESHOLD FOR NUMBER OF POSITIVE CURVATURE POINTS FOR SELECTED CURVATURE
PHIn_thresh = 300	# // THRESHOLD FOR NUMBER OF NEGATIVE CURVATURE POINTS FOR SELECTED CURVATURE
PHIf_thresh = 99550	# // THRESHOLD FOR NUMBER OF POINTS REPRESENTING NEARLY FLAT SURFACES FOR SELECTED CURVATURE
# ---------------------------------------------------------------------------------------



# READ .PLY FILE AND EXTRACT DATA -------------------------------------------------------
# ---------------------------------------------------------------------------------------
PTCL_data = PlyData.read('odm_georeferenced_model.ply')
# PLYFILE // READ POINT CLOUD DATA

MESH_data = PlyData.read('odm_mesh.ply')
# PLYFILE // READ MESH DATA

x_coordinate = MESH_data.elements[0].data['x']
y_coordinate = MESH_data.elements[0].data['y']
z_coordinate = MESH_data.elements[0].data['z']

R_channel = PTCL_data.elements[0].data['red']
G_channel = PTCL_data.elements[0].data['green']
B_channel = PTCL_data.elements[0].data['blue']

x_normal = PTCL_data.elements[0].data['nx']
y_normal = PTCL_data.elements[0].data['ny']
z_normal = PTCL_data.elements[0].data['nz']
# ---------------------------------------------------------------------------------------



# CURVATURE/LUMINANCE ANALYSIS ----------------------------------------------------------
# ---------------------------------------------------------------------------------------
data = np.vstack((x_coordinate, y_coordinate, z_coordinate)).T
# NUMPY //

neighbors = NearestNeighbors((n_neighbors + 1), algorithm = 'kd_tree').fit(data)
# SKLEARN // FIT THE NEAREST NEIGHBORS MODEL USING 'data' AS TRAINING DATA

distances, indices = neighbors.kneighbors(data)
# SKLEARN // EXTRACT DISTANCES AND INDICES OF NEIGHBORS WITHIN INDICATED THRESHOLD IN 'n_neighbors'

distances = distances[:,1:]
# NUMPY // REMOVE SELF, AND KEEP DISTANCES TO NEIGHBORS

indices = indices[:,1:]
# NUMPY // REMOVE SELF, AND KEEP INDICES TO NEIGHBORS

p = np.matlib.repmat(data, n_neighbors, 1) - data[np.ravel(indices), 0:]
p = np.reshape(p, (len(data), n_neighbors, 3))
# NUMPY // REPEAT ARRAY OR MATRIX, AND RESHAPE

C = np.zeros((len(data), 6))
# NUMPY // INITIALIZE COVARIANCE MATRIX

C[:,0] = np.sum(p[:,:,0]*p[:,:,0], axis = 1)
C[:,1] = np.sum(p[:,:,0]*p[:,:,1], axis = 1)
C[:,2] = np.sum(p[:,:,0]*p[:,:,2], axis = 1)
C[:,3] = np.sum(p[:,:,1]*p[:,:,1], axis = 1)
C[:,4] = np.sum(p[:,:,1]*p[:,:,2], axis = 1)
C[:,5] = np.sum(p[:,:,2]*p[:,:,2], axis = 1)
# NUMPY // CALCULATE ELEMENTS OF COVARIANCE MATRIX


curvature = np.zeros(len(data))
luminance = np.zeros(len(data))
# NUMPY // INITIALIZE CURVATURE AND LUMINANCE


neg_curv = []
pos_curv = []
flt_curv = []

neg_count = 0
pos_count = 0
flt_count = 0
drk_count = 0

for i in range(len(data)):
	
	C_MAT = np.array([[C[i,0], C[i,1], C[i,2]], [C[i,1], C[i,3], C[i,4]], [C[i,2], C[i,4], C[i,5]]])
	# NUMPY // POSITION COVARIANCE MATRIX ELEMENTS INTO AN ARRAY
	
	eVals, eVecs = la.eig(C_MAT)
	# NUMPY // FIND EIGENVALUES AND EIGENVECTORS OF COVARIANCE MATRIX

	diag = np.diag(eVecs)
	# NUMPY // EXTRACT A DIAGONAL ARRAY
	
	lmda = np.min(diag)
	# NUMPY // RETURN MINIMUM
	
	
	# COUNT POINT CLOUDS THAT HAVE CURVATURE WITHIN A DESIRED RANGE
	# ---------------------------------------------------------------------
	curvature[i] = lmda/float(np.sum(diag))
	# NUMPY //
	
	if curvature[i] < -curv_thresh:
		neg_curv = np.append(neg_curv, curvature[i])

		neg_count += 1
	# END
	
	elif curvature[i] > curv_thresh:
		pos_curv = np.append(pos_curv, curvature[i])

		pos_count += 1
	# END
	
	else:
		flt_curv = np.append(flt_curv, curvature[i])

		flt_count += 1
	# END
	# ---------------------------------------------------------------------
	
	
	# COUNT POINT CLOUDS THAT HAVE LUMINANCE ABOVE A DESIRED THRESHOLD
	# ---------------------------------------------------------------------
	luminance[i] = 0.2126*float(R_channel[i]) + 0.7152*float(G_channel[i]) + 0.0722*float(B_channel[i])
	# MATH //
	
 	if luminance[i] > lumi_thresh:
		drk_count += 1
	# END
	# ---------------------------------------------------------------------
# END

# ---------------------------------------------------------------------------------------



# WRITE DATA ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# neg_curv_df = pd.DataFrame(neg_curv)
# neg_curv_df.to_csv('negative_curvature.csv')
# PANDAS //

# pos_curv_df = pd.DataFrame(pos_curv)
# pos_curv_df.to_csv('positive_curvature.csv')
# PANDAS //

# flt_curv_df = pd.DataFrame(flt_curv)
# flt_curv_df.to_csv('flat_curvature.csv')
# PANDAS //

# lumi_df = pd.DataFrame(luminance)
# lumi_df.to_csv('luminance245.csv')
# PANDAS //
# ---------------------------------------------------------------------------------------



# CLASSIFY SURFACE ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
CRK_point = 0
DEP_point = 0

# TEST POSITIVE-TO-FLAT RATIO
# -----------------------------------------------
if (pos_count/float(flt_count)) < PF_thresh:
	CRK_point += 1
else:
	DEP_point += 1
# -----------------------------------------------

# TEST NEGATIVE-TO-FLAT RATIO
# -----------------------------------------------
if (neg_count/float(flt_count)) < NF_thresh:
	CRK_point += 1
else:
	DEP_point += 1
# -----------------------------------------------

# TEST FLAT-TO-ALL RATIO
# -----------------------------------------------
if (flt_count/float(len(data))) > FA_thresh:
	CRK_point += 1
else:
	DEP_point += 1
# -----------------------------------------------

# TEST NUMBER OF POSITIVE CURVATURE POINTS
# -----------------------------------------------
if pos_count < PHIp_thresh:
	CRK_point += 1
else:
	DEP_point += 1
# -----------------------------------------------

# TEST NUMBER OF NEGATIVE CURVATURE POINTS
# -----------------------------------------------
if neg_count < PHIn_thresh:
	CRK_point += 1
else:
	DEP_point += 1
# -----------------------------------------------

# TEST NUMBER OF FLAT REGION POINTS
# -----------------------------------------------
if flt_count > PHIf_thresh:
	CRK_point += 1
else:
	DEP_point += 1
# -----------------------------------------------

# ---------------------------------------------------------------------------------------



# PRINT RESULTS -------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
print "--CURVATURE------------------------------------"
print "Number of Positive Curvature Regions: ", pos_count
print "Number of Negative Curvature Regions: ", neg_count
print "NUmber of Flat Regions: ", flt_count
print "-----------------------------------------------"

print "--LUMINANCE------------------------------------"
print "Number of Dark Luminance Regions above 245: ", drk_count
print "-----------------------------------------------"

print "--CURVATURE RATIOS-----------------------------"
print "PF Ratio: ", pos_count/float(flt_count)
print "NF Ratio: ", neg_count/float(flt_count)
print "FA Ratio: ", flt_count/float(len(data))
print "-----------------------------------------------"

print "--CLASSIFICATION RESULT------------------------"
print "Points for Presence of Surface Crack     : ", CRK_point
print "Points for Presence of Surface Depression: ", DEP_point
print "-----------------------------------------------"
# ---------------------------------------------------------------------------------------




