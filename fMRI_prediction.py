"""
fMRI prediction of sexual orientation
2021
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""

import pandas as pd #v1.1.3
import numpy as np #v1.19.2
import glob
import nibabel as nib #v3.2.1
import nilearn.datasets as ds #nilearn v0.7.1
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import StandardScaler #sklearn v0.23.2
from nilearn.signal import clean
from sklearn.linear_model import LogisticRegression
from scipy import stats #v1.5.2
from matplotlib import pylab as plt #matplotlib v3.3.2
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import KFold
from nilearn import image

# reproducibility
np.random.seed(0)

######################################
##### EXTRACTING DEMOGRAPHIC DATA ####
######################################

print("Extracting demographic data")
# read excel doc as df
df = pd.read_excel("C:\\sexualorientproject\\DATA_IZKF_Version.xlsx")
df = df.set_index('No.')
# drop index 87 as we don't have MRI data for this subject
df = df.drop([87])
# Outcome = sexual orientation.
# Group:  1 hetero, 2 homo
Y = df["Group"]
# Group:  1 hetero, 0 homo
Y[Y == 2] = 0 
# 41 homo, 45 hetero in total
Y = Y.values # df to array
# add path to functional imgs in df
df["fMRI_path"] = ['C:\\sexualorientproject\\RS\\filtered_func_data_warped_{}.nii.gz'.format(i) for i in df.index]


################################
##### Functions needed later ###
################################

def extract_ROIconn(matrix):
	''' Extract connectivity of each entry of a matrix with the others, without the 
	connectivity with itself. 
	Return output as an array shape (p, p-1)

	Parameters
	----------
	matrix : must be a squared matrix (p*p)
	
	'''
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


def subtract(TS):
	''' Substract TS(t+1) with TS[t]
	return the result as an array

	Parameters
	----------
	TS : MC parameter for one subject, must be of shape (121, 6)
	6 parameters for 121 time series
	
	'''
	TS_ = np.zeros((121, 6)).astype(TS.dtype)
	for i in range(121):
		if i == 120:
			TS_[i] = TS[i] # last columns has no t+1 so same ts
		else:
			TS_[i] = TS[i+1] - TS[i] # transpose because substract column wise
	return TS_


# Plotting function for the x axis to be centered
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)

###########################################################
##### Preparing confounds (motion parameter) 1:openfile ###
###########################################################

print("Extracting motion parameter data")
# Extract motion parameters from .PAR 
# save it as .xlsx doc
MP_paths = glob.glob('C:\\sexualorientproject\\MC_Parameter\\*.par')
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
	df_mp1.to_excel("C:\\sexualorientproject\\MC_Parameter\\first\\{}.xlsx".format(MP[36:-4]))



##############################################################################
##### Preparing confounds (motion parameter) 2:compute subtract and square ###
##############################################################################
	
	
print("Computing substracting and squaring of motion parameter")
# add 6  columns for subtract at t+1 and add 12 columns for squared values
MP_paths = glob.glob('C:\\sexualorientproject\\MC_Parameter\\first\\*.xlsx')
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
	df_mp1.to_excel("C:\\sexualorientproject\\MC_Parameter\\second\\{}".format(MP[42:]))


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

# extract TS per ROI, preprocess and compute correlation matrix
# loop through 86 subjects

print("Starting the ROI time series extraction")
FS = []
for i_nii, nii_path in enumerate(df.fMRI_path.values):
	# each participant has a 4D nii of shape (91, 109, 91, 121)
	print("Currently doing sub", i_nii + 1, "/86")
	# EXTRACT TIME SERIES PER roi
	cur_FS = masker.transform(nii_path)
	# standardize the time series
	cur_FS = StandardScaler().fit_transform(cur_FS)
	# shape (121, 100)
	# deconfounding
	# upload df with motion parameter values
	confounds = pd.read_excel('C:\\sexualorientproject\\MC_Parameter\\second\\prefiltered_func_data_mcf_{}.xlsx'.format(i_nii+1))
	confounds = confounds[confounds.columns[1:]].values
	assert confounds.shape == (121, 24)
	# deconfounding
	cur_FS = clean(signals=cur_FS, confounds=confounds, detrend=False)
	# shape (121, 100)
	FS.append(cur_FS)

np.save("C:\\sexualorientproject\\fmri_FS_ss", FS)

print("Remove variance explained by confounds in ROI time series extraction")
# extract confound information
confounds = [
	"Biological Sex",
	"Age",
	"EducationalLevel",
	"Handedness"
    	     ]
my_confound = df[confounds].values

# actual signal deconfounding 
FS = np.array(FS).reshape(86, 121*100) # needs to be 2D to be cleaned
FS = clean(signals=FS, confounds=my_confound, detrend=False, standardize=False)
FS = FS.reshape(86, 100, 121)

# compute the correlation matrix subject wise, and extract the correlation per ROI
corrs = []
for ind, cur_FS in enumerate(FS):
	print(ind, ' / ', 86)
	# compute pearson correlation matrix
	sub_cross_corr = np.corrcoef(cur_FS) # shape 100*100
	# extract correlation per ROI (see function at the top of the script)
	sub_cross_corr_per_ROI = extract_ROIconn(sub_cross_corr) # shape 100, 99
	corrs.append(sub_cross_corr_per_ROI) # shape 86 times 100, 99 at the end of the loop

corrs = np.array(corrs)
assert np.isnan(corrs).sum() == 0
corrs = corrs.reshape((100,86,99)) # for the analysis each ROI at a turn
np.save("C:\\sexualorientproject\\corrs", corrs)
# shape (100, 86, 99)



########################################
# LOAD MASKER TO SAVE RESULTS AS NIFTI #
########################################
# usefull if you don't want to run the full extraction again

print("Masking")
# define masker
atlas = ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
first_nii = nib.load(df.fMRI_path.values[0])
cur_ratlas_nii = resample_img(
    	atlas.maps, target_affine=first_nii.affine, interpolation='nearest')
masker = NiftiLabelsMasker(labels_img=cur_ratlas_nii, standardize=False) # standardization done later in the loop
masker.fit()
masker.transform(df.fMRI_path.values[0])

##########################################
### LEVEL 0 of the stacking analysis  ####
##########################################

# this part fits a model per ROI to predict sexual orientation
# the input is 99 correlation values per ROI (n=100) for each subject (n=86) 
# the output is a probability of predicting an hetero per ROI (n=100) per subject (n=86)

# input level 0 shape = 100*86*99
corrs = np.load("C:\\sexualorientproject\\corrs.npy")

# save accuracies to check model fit quality
accs_level_0 = []
accs_std_level_0 = []

# the probabilities as output of level 0, used as input of level 1 
output_level_0 = []

# check if contamination of test set during the whole loop
testSetIdx_level_0 = []

# iterate through each ROI
for ROI in range(100):
	print("*******")
	print(ROI, "/100")
	print("*******")
	X = corrs[ROI] # extract correlation of a particular ROI for all subjects, shape X = (86, 99)
	sample_accs = []
	sample_std = []
	sample_proba = []

	# starting the modelisation
	# run the CV 5 fold logistic regression as many times as we got brain regions (thus 100 times)
	clf = LogisticRegression()
	kf = KFold(n_splits=5, shuffle=False, random_state=0) # shuffle = false to ensure no leaks in the stacking analysis
	kf.get_n_splits(X)

	# Save output for level 1
	probs = np.array([])

	# cross validation
	for train_index, test_index in kf.split(X):
		testSetIdx_level_0.append(test_index) # save to check for leaking
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# LogReg ROI wise
		clf.fit(X_train, y_train)
		proba = clf.predict_proba(X_test)
		y_pred = clf.predict(X_test)
		acc = (y_pred == y_test).mean()
		sample_accs.append(acc)

		# Keep the proba of being hetero according to this fit, for the level 1 LogReg stalking
		probs = np.concatenate((probs, proba[:, 1] ), axis=None)
	# Keep this for the LogReg stalking
	output_level_0.append(probs)
	# for checking model fit quality
	accs_level_0.append(np.mean(sample_accs))
	accs_std_level_0.append(np.std(sample_accs))


# arraying the results
accs_level_0 = np.array(accs_level_0)
accs_std_level_0 = np.array(accs_std_level_0)

# niftiing the results
level0_accs4D = np.array([accs_level_0]*121) # set as fake 4D for nii transform
accs_level0_nii4D = masker.inverse_transform(level0_accs4D)
accs_level0_nii3D = image.index_img(accs_level0_nii4D, 0)
accs_level0_nii3D.to_filename("accs_level0_3D.nii") # transform as nii and save
# save to double check when reran that we obtain same results
np.save("C:\\sexualorientproject\\output_level_0", output_level_0)


# Keep this for level 1 of the stacking analysis
output_level_0 = np.array(output_level_0)

##########################################
### LEVEL 1 of the stacking analysis  ####
##########################################

# keep information for the confusion matrix
all_y_pred = np.array([])
all_y_true = np.array([])

# save accuracies to check model fit quality
accs_level_1 = []

# the coeficients of the final fit 
output_level_1 = []

# take output of level 0 as input of the level 1 stacking analysis
X = output_level_0.T # Shape (86, 100)

# starting the modelisation
# run the CV 5 fold logistic regression
clf = LogisticRegression()
kf = KFold(n_splits=5, shuffle=False, random_state=0) # shuffle = false to ensure no leaks in the stacking analysis
kf.get_n_splits(X)

# check if contamination of test set during the whole loop
testSetIdx_level_1 = []

# cross validation
for train_index, test_index in kf.split(X):
	testSetIdx_level_1.append(test_index) # save to check for leaking
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]
	# LogReg subject wise
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	acc = (y_pred == y_test).mean()
	accs_level_1.append(acc)
	# extract results for the confusion matrix
	all_y_pred = np.concatenate((all_y_pred, y_pred), axis=None)
	all_y_true = np.concatenate((all_y_true, y_test), axis=None)
	output_level_1.append(clf.coef_)

# check for leaking
for i in range(0, 100, 5):
	for j in range(5):
		assert (testSetIdx_level_0[j+i] != testSetIdx_level_1[j]).sum() == 0

# compute mean accuracy and standard deviation
acc_level_1 = np.mean(accs_level_1)
acc_std_level_1 = np.std(accs_level_1)

# print results
print("acc mean = {}, std = {}".format(acc_level_1, acc_std_level_1))

final_coefficients = np.mean(output_level_1, axis=0).reshape(100,)
final_coefficients_std = np.std(output_level_1, axis=0).reshape(100,)

# save it as txt to check when rerun the script #reproducibility
file = open("final_coefficients.txt", "w+")
content = str(final_coefficients)
file.write(content)
file.close()
file = open("final_coefficients_std.txt", "w+")
content = str(final_coefficients_std)
file.write(content)
file.close()



# niftiing the staking results
final_coefficients4D = np.array([final_coefficients]*121) # set as fake 4D for nii transform
final_coefficients_nii4D = masker.inverse_transform(final_coefficients4D)
final_coefficients_nii3D = image.index_img(final_coefficients_nii4D, 0)
final_coefficients_nii3D.to_filename("final_coefficients3D.nii") # transform as nii and save

final_coefficients_std4D = np.array([final_coefficients_std]*121) # set as fake 4D for nii transform
final_coefficients_std_nii4D = masker.inverse_transform(final_coefficients_std4D)
final_coefficients_std_nii3D = image.index_img(final_coefficients_std_nii4D, 0)
final_coefficients_std_nii3D.to_filename("final_coefficients_std3D.nii") # transform as nii and save



################################
#### CONFUSION MATRIX ##########
################################

# compute and plot the confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

f, ax = plt.subplots(figsize=(8, 8))
class_names = ["Homosexual", "Heterosexual"]

# matrix
cm = confusion_matrix(all_y_true, all_y_pred)
cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = (cm_ * 100) # percentage
for indx, i in enumerate(cm):
    for indy, j in enumerate(i):
        j = round(j, 1)
        print(j)
        cm[indx, indy] = j
print(cm) # double check
plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Reds)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize=12)
plt.yticks(tick_marks, class_names, fontsize=12)
rotateTickLabels(ax, -55, 'x')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
             horizontalalignment="center",
             color= "black", fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.ylabel("True label", fontsize=15)
plt.tight_layout()
plt.savefig('confusion_matrix_fmri.png', PNG=300)
plt.show()


# ##########################
# ### DOUBLE CHECKING  #####
# ##########################

# # compare highest weights of stalking logreg with the accuracy that a ROI obtained
# for ind, coef in enumerate(clf.coef_[0]):
# 	if coef > 0.6:
# 		print("*****")
# 		print("ROI name: ", atlas.labels[ind])
# 		print("acc: ", roi_pred_accs[ind], " weight: ", coef)
# 		print("poba obtained and real output:")
# 		print(probas_staking[ind][40:50])
# 		print(Y[40:50])
		
# # running the logreg staking with permutated Y
# accs_stalking = []
# X = probas_staking.T # Shape (86, 100)
# clf = LogisticRegression()
# kf = KFold(n_splits=5, shuffle=False, random_state=0)
# kf.get_n_splits(X)
# for i in range(100):
# 	perm_rs = np.random.RandomState(i)
# 	Y_perm = perm_rs.permutation(Y.copy())
# 	for train_index, test_index in kf.split(X):
# 		X_train, X_test = X[train_index], X[test_index]
# 		y_train, y_test = Y_perm[train_index], Y_perm[test_index]
# 		clf.fit(X_train, y_train)
# 		y_pred = clf.predict(X_test)
# 		print(y_train, y_test, y_pred)
# 		acc = (y_pred == y_test).mean()
# 		accs_stalking.append(acc)
# mean_stalking = np.mean(accs_stalking)
# std_stalking = np.std(accs_stalking)
# print("acc mean = {}, std = {}".format(mean_stalking, std_stalking))


##############################################
####### Non-parameric hypothesis test ########
##############################################

# Non-parameric hypothesis test 
# run the CV 5 fold logistic regression with permutated Y

# take output of level 0 as input of the level 1 stacking analysis
X = output_level_0.T # Shape (86, 100)

n_permutations = 100
perm_rs = np.random.RandomState(0)
permutation_accs = []
permutation_coefs = []
for i_iter in range(n_permutations):
	print(i_iter + 1)
	Y_perm = perm_rs.permutation(Y)
	clf = LogisticRegression()
	kf = KFold(n_splits=5, shuffle=False, random_state=i_iter)
	kf.get_n_splits(X)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y_perm[train_index], Y_perm[test_index]
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		acc = (y_pred == y_test).mean()
		permutation_accs.append(acc)
		permutation_coefs.append(clf.coef_[0, :])

# extract permutation weigth per ROI for hypothesis testing
weight_per_roi = []
for n_perm in range(100):
	weight_roi = []
	for ROI in permutation_coefs:
		weight_roi.append(ROI[n_perm])
	weight_per_roi.append(weight_roi)

# extract 95 and 5% percentile and check if original weight outside these limits to check for significance
pvals = []
for n_roi in range(100):
	ROI_weights = weight_per_roi[n_roi] 
	above = stats.scoreatpercentile(ROI_weights, 95)
	below = stats.scoreatpercentile(ROI_weights, 5)
	if final_coefficients[n_roi] < below or final_coefficients[n_roi] > above:
		pvals.append(1)
	else:
		pvals.append(0)

pvals = np.array(pvals)
print('{} ROIs are significant at p<0.05'.format(np.sum(pvals > 0)))


# get significant ROI name + significant weights
significant_weights = []
dic = {}
for i, roi in enumerate(atlas.labels):
	if pvals[i] == 1:
		significant_weights.append(final_coefficients[i])
		print(roi, final_coefficients[i])
		dic[roi] = final_coefficients[i]
	else:
		significant_weights.append(0)


################################
#### Brain visualisation #######
################################

# extract significant results to nii for visualisation
significant_weights_4D = np.array([significant_weights]*121) # set as fake 4D for nii transform
significant_weights_4D_nii = masker.inverse_transform(significant_weights_4D)
significant_weights_nii3D = image.index_img(significant_weights_4D_nii, 0)
significant_weights_nii3D.to_filename('significantROIs_fmri.nii')
