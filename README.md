# damagedetectionanalysis

For my Thesis project, I worked on the problem of damage detection (or quantification) on any given infrastruce surface. I also attempted damage analysis (for identification of the type of damage). I developed the entire Python code in a Linux environment.

![](https://raw.githubusercontent.com/troymyname/damagedetectionanalysis/edit/master/images/sample.jpg)

STEPS:
1. Use alpha.py to read train images from a local directory and calculate features, perform PCA, and store the data in a local directory.
2. Use beta.py to read the stored data from Step 1 and then, calulate train weight vectors, train a One-Class SVM (OCSVM) classifier, use the trained OCSVM to classify test image blocks for match/mismtach, and finally quantify damage.
3. Use OpenDroneMap toolkit to generate point cloud and mesh models from multiple overlapping image sets. This process generates .ply files.
4. Use gamma.py to read the .ply files to calculate curvature and luminance properties, and finally use the properties to classify the type of damage. CAUTION: This section needs improvement.


PRE-REQUISITES:
- Python 2.7X
--- numpy
--- cv2
--- matplotlib
--- pandas
--- seaborn
--- PIL
--- plyfile
--- sklearn
- OpenDroneMap Toolkit from GitHub
- MeshLab (to read/display .ply files)

If you require the raw data that I used for my thesis project, please create an issue.
