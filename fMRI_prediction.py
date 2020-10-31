import glob
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_img
from nilearn import image
import nilearn.datasets as ds
import nibabel as nib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
from matplotlib import pylab as plt



print("Extracting demographic data")

np.random.seed(0)

######################################
##### EXTRACTING DEMOGRAPHIC DATA ####
######################################

# read excel doc as df
df = pd.read_excel("DATA_IZKF_Version.xlsx")
df = df.set_index('No.')
# drop index 87 as we don't have MRI data
df = df.drop([87])

# Outcome= sexual orientation.
# Group:  1 hetero, 2 homo
Y = df["Group"]
# Group:  1 hetero, 0 homo
Y[Y == 2] = 0 
# 41 homo, 45 hetero in total
Y = Y.values # df to array


# add path to structural imgs in df
df["fMRI_path"] = ['Function/filtered_func_data_warped_{}.nii.gz'.format(i) for i in df.index]






# function to extract correlation matrix results per ROI
# return the values per ROI (must be squared matrix)
def extract_ROIconn(matrix):
	output_col = []
	for column in range(len(matrix)-1):
		output_col.append(matrix.T[column][column+1:])
	output_row = []
	for row in range(1, len(matrix)):
		output_row.append(matrix[row][:row])
	output = []
	for nb in range(len(output_col)+1):
		if nb == 0:
			output.append(output_col[nb])
		elif nb == len(output_col):
			output.append(output_row[nb-1])
		else:
			out = np.concatenate((output_row[nb - 1], output_col[nb]))
			output.append(out)
	return np.array(output)

# function to substract TS of t + 1
def subtract(TS):
	TS_ = np.zeros((121, 6)) # last columns has no t+1
	for i in range(121):
		if i == 120:
			TS_[i] = TS[i]
		else:
			TS_[i] = TS[i+1] - TS[i] # transpose because substract column wise
	return TS_


print("Extracting motion parameter data")
# Extract motion parameters from .PAR 
# save it as .xlsx doc
MP_paths = glob.glob('MP/*.par')
for MP in MP_paths:
	df_mp1 = pd.DataFrame(columns=[1, 2, 3, 4, 5, 6], 
						index=np.arange(121)) 
	df_mp2 = pd.read_csv(MP, header=None)  
	for row in df_mp2.index:
		line = df_mp2.iloc[row].values[0]
		splits = line.split(' ')
		ind = 1
		for split in splits:
			if split != '':
				df_mp1.iloc[row][ind] = np.float(split)
				ind += 1
	df_mp1.to_excel("MP/first/{}.xlsx".format(MP[3:-4]))

print("Computing proprocess of motion parameter")

MP_paths = glob.glob('MP/first/*.xlsx')
for MP in MP_paths:
	df_mp1 = pd.read_excel(MP)
	TS = df_mp1[df_mp1.columns[1:]].values
	# shape (121, 6)
	# substract neighbor in time 
	cur_FS = subtract(TS)
	# shape (121, 6)
	for i in range(6):
		df_mp1[i+7] = cur_FS.T[i]
	# shape (120, 12) pour le df
	data = df_mp1[df_mp1.columns[1:]].values**2
	df_mp1 = pd.DataFrame(columns=np.arange(12), 
					data=data)
	df_mp1.to_excel("MP/second/{}.xlsx".format(MP[9:-5]))



print("Masking")
###########################
### Preparing masking #####
###########################

# define masker
atlas = ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
first_nii = nib.load(df.fMRI_path.values[0])
cur_ratlas_nii = resample_img(
    	atlas.maps, target_affine=first_nii.affine, interpolation='nearest')
masker = NiftiLabelsMasker(labels_img=cur_ratlas_nii)
masker.fit()


print("Starting the ROI time series extraction")
###################################
### Preprocessing per subject #####
###################################

# extract TS per ROI, preprocess and compute correlation matrix
# loop through 86 subjects
# extract TS per ROI, preprocess and compute correlation matrix
# loop through 86 subjects
FS = []
for i_nii, nii_path in enumerate(df.fMRI_path.values):
	# each participant has a 4D nii of shape (91, 109, 91, 121)
	print("Currently doing sub", i_nii + 1, "/86")
	# EXTRACT TIME SERIES PER roi
	cur_FS = masker.transform(nii_path)
	# shape (121, 100)
	# deconfounding
	# upload df with motion parameter values
	confounds = pd.read_excel('MP/second/prefiltered_func_data_mcf_{}.xlsx'.format(i_nii+1))
	confounds = confounds[confounds.columns[1:]].values
	cur_FS = clean(signals=cur_FS, confounds=confounds, detrend=False, standardize=True)
	# shape (121, 100)
	# compute cross correlation
	sub_cross_corr = np.corrcoef(cur_FS.T)
	# shape (100, 100)
  	# extract results per ROI
	sub_cross_corr_per_ROI = extract_ROIconn(sub_cross_corr)
	# shape (100, 99) # save results for each subject
	FS.append(sub_cross_corr_per_ROI)

np.save("Data_ready", FS)
  
# shape (86, 100, 99)

FS = np.load("Data_ready.npy")
FS = FS.reshape((100,86,99)) 
#########################################
### ACTUAL MODELING + DECONFOUNDING #####
#########################################
'''
same classifier (LogReg)
BUT going to each row (=a source brain region with its connections to all other regions) 
build a feature matrix only with information from that region for all subjects
do same cross validation cycles as in sMRI (but only with information related to THAT source region)
then get the .predict_probaba() across all CV folds and average that, keep that averaged 0...1 continuous number of THAT region
repeat that last for all regions, then paste the average continuous predictions for each region into a nifti, 
we will take a look (once this looks good, we go from there)

I will run the log reg with input.shape = [nb of subject, 99 correlations] 
# which I will do for each ROI (100 in total)
# if I run clf.predict_proba(X_test, Y_test), I will end up with
# a results of shape [89, 2] and not [89, 1]
(2 because a number for homo and another for hetero)

paste the average continuous predictions for each region into a nifti
'''



#######################################################
##### EXTRACTING FUNCTIONAL BRAIN DATA PER SUBJECT ####
#######################################################


# loop to extract ROI 1 results for each patients
pop_accs = []
pop_proba = []
for ROI in range(100):
	print("*******")
	print(ROi, "/100")
	X = FS[ROI]
	sample_accs = []
	sample_proba = []
	# starting the modelisation
	#run the CV 5 fold logistic regression as many times as we got brain regions
	for i_subsample in range(100):
		clf = LogisticRegression()
		sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=i_subsample)
		sss.get_n_splits(X, Y)
		for train_index, test_index in sss.split(X, Y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			clf.fit(X_train, y_train)
			proba = clf.predict_proba(X_test)
			y_pred = clf.predict(X_test)
			acc = (y_pred == y_test).mean()
			sample_accs.append(acc)
			sample_proba.append(np.mean(proba, axis=0))
	pop_proba.append(np.mean(sample_proba, axis=0))
	pop_accs.append(np.mean(sample_accs))






