import glob
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_img
from nilearn import image
import nilearn.datasets as ds
import nibabel as nib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy import stats
from matplotlib import pylab as plt


np.random.seed(0)

######################################
##### EXTRACTING DEMOGRAPHIC DATA ####
######################################


print("Extracting demographic data")
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
# add path to functional imgs in df
df["fMRI_path"] = ['Function/filtered_func_data_warped_{}.nii.gz'.format(i) for i in df.index]


################################
##### Functions needed later ###
################################

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
	TS_ = np.zeros((121, 6)) 
	for i in range(121):
		if i == 120:
			TS_[i] = TS[i] # last columns has no t+1 so same ts
		else:
			TS_[i] = TS[i+1] - TS[i] # transpose because substract column wise
	return TS_



###########################################################
##### Preparing confounds (motion parameter) 1:openfile ###
###########################################################

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



##############################################################################
##### Preparing confounds (motion parameter) 2:compute subtract and square ###
##############################################################################
	
	
print("Computing substracting and squaring of motion parameter")
# add 6  columns for subtract at t+1 and add 12 columns for squared values
MP_paths = glob.glob('MP/first/*.xlsx')
for MP in MP_paths:
	df_mp1 = pd.read_excel(MP)
	TS = df_mp1[df_mp1.columns[1:]].values
	# shape (121, 6)
	# substract neighbor in time 
	cur_FS = subtract(TS)
	# shape (121, 6)
	# add results to df
	for i in range(6):
		df_mp1[i+7] = cur_FS.T[i]
	# df shape (121, 12) 
	data = df_mp1[df_mp1.columns[1:]].values
	# shape (121, 12) 
	data_squared = df_mp1[df_mp1.columns[1:]].values**2
	# shape (121, 12) 
	datum = np.concatenate((data, data_squared), axis=1)
	# shape (121, 24) 
	df_mp1 = pd.DataFrame(columns=np.arange(24), 
					data=datum)
	# df shape (121, 24) 
	df_mp1.to_excel("MP/second/{}.xlsx".format(MP[9:-5]))


###########################
### Preparing masking #####
###########################

print("Masking")
# define masker
atlas = ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
first_nii = nib.load(df.fMRI_path.values[0])
cur_ratlas_nii = resample_img(
    	atlas.maps, target_affine=first_nii.affine, interpolation='nearest')
masker = NiftiLabelsMasker(labels_img=cur_ratlas_nii, standardize=False) # standardization done later in the loop
masker.fit()




###################################
### Preprocessing per subject #####
###################################

print("Starting the ROI time series extraction")
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
	cur_FS = StandardScaler().fit_transform(cur_FS)
	# shape (121, 100)
	# deconfounding
	# upload df with motion parameter values
	confounds = pd.read_excel('MP/second/prefiltered_func_data_mcf_{}.xlsx'.format(i_nii+1))
	confounds = confounds[confounds.columns[1:]].values
	assert confounds.shape == (121, 24)
	# deconfounding
	cur_FS = clean(signals=cur_FS, confounds=confounds, detrend=False)
	# shape (121, 100)
	# compute cross correlation
	sub_cross_corr = np.corrcoef(cur_FS.T)
	# shape (100, 100)
  	# extract results per ROI
	sub_cross_corr_per_ROI = extract_ROIconn(sub_cross_corr)
	# shape (100, 99) # save results for each subject
	FS.append(sub_cross_corr_per_ROI)
FS = np.array(FS)
FS = np.nan_to_num(FS) # 2 subjects with a few nan (3 and 4, both hetero)
FS = FS.reshape((100,86,99)) # for the analysis each ROI at a turn
np.save("Data_ready", FS)
# shape (86, 100, 99)

FS = np.load("Data_ready.npy")

##########################
### ACTUAL MODELING  #####
##########################

# loop to extract ROI 1 results for all patients
roi_pred_accs = []
roi_pred_std = []
roi_pred_proba = []

# need this for the LogReg stalking
probas_staking = []

for ROI in range(100):
	print("*******")
	print(ROI, "/100")
	print("*******")
	X = FS[ROI]
	sample_accs = []
	sample_std = []
	sample_proba = []

	# starting the modelisation
	#run the CV 5 fold logistic regression as many times as we got brain regions
	clf = LogisticRegression()
	kf = KFold(n_splits=5, shuffle=False, random_state=ROI)
	kf.get_n_splits(X)

	# need this for the LogReg stalking
	probs = np.array([])

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# LogReg ROI wise
		clf.fit(X_train, y_train)
		proba = clf.predict_proba(X_test)
		y_pred = clf.predict(X_test)
		acc = (y_pred == y_test).mean()
		sample_accs.append(acc)
		sample_proba.append(np.mean(proba, axis=0))

		# Keep this for the LogReg stalking
		probs = np.concatenate((probs, proba[:, 1] ), axis=None)
	# Keep this for the LogReg stalking
	probas_staking.append(probs)

	roi_pred_proba.append(np.mean(sample_proba, axis=0))
	roi_pred_accs.append(np.mean(sample_accs))
	roi_pred_std.append(np.std(sample_accs))


# arraying the results
roi_pred_proba = np.array(roi_pred_proba)
roi_pred_proba_hetero = roi_pred_proba[:, 1] 
roi_pred_accs = np.array(roi_pred_accs)
roi_pred_std = np.array(roi_pred_std)

# Keep this for the LogReg stalking
probas_staking = np.array(probas_staking)

# running the logreg staking
accs_stalking = []
X = probas_staking.T # Shape (86, 100)
clf = LogisticRegression()
kf = KFold(n_splits=5, shuffle=False, random_state=0)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	acc = (y_pred == y_test).mean()
	accs_stalking.append(acc)
mean_stalking = np.mean(accs_stalking)
std_stalking = np.std(accs_stalking)
print("acc mean = {}, std = {}".format(mean_stalking, std_stalking))





# niftiing the results
pop_proba_hetero = np.array([roi_pred_proba_hetero]*121) # set as fake 4D for nii transform
pop_accs = np.array([roi_pred_accs]*121) # set as fake 4D for nii transform
proba_nii = masker.inverse_transform(pop_proba_hetero).to_filename("proba.nii") # transform as nii and save
accs_nii = masker.inverse_transform(pop_accs).to_filename("accs.nii") # transform as nii and save





