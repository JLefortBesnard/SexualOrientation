"""
MRI prediction of sexual orientation
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
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats #v1.5.2
from matplotlib import pylab as plt #matplotlib v3.3.2
from sklearn.metrics import confusion_matrix
import itertools

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
# Outcome= sexual orientation.
# Group:  1 hetero, 2 homo
Y = df["Group"]
# Group:  1 hetero, 0 homo
Y[Y == 2] = 0 
# 41 homo, 45 hetero in total
Y = Y.values # df to array
# add path to structural imgs in df
df["MRI_path"] = ['C:\\sexualorientproject\\Anatomy\\highres2standard_{}.nii.gz'.format(i) for i in df.index]



###########################################
##### EXTRACTING STRUCTURAL BRAIN DATA ####
###########################################


print("Masking")
# define masker
atlas = ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
tmp_nii = nib.load(df["MRI_path"][1])
ratlas_nii = resample_img(
  atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
# ratlas_nii.to_filename('debug_ratlas.nii.gz')

# extracting VBM per ROI
FS = []
for i_nii, nii_path in enumerate(df["MRI_path"].values):
	print(nii_path)
	nii = nib.load(nii_path)
	cur_ratlas_nii = resample_img(
		atlas.maps, target_affine=nii.affine, interpolation='nearest')
	nii_fake4D = nib.Nifti1Image(
		nii.get_data()[:, :, :, None], affine=nii.affine)
	masker = NiftiLabelsMasker(labels_img=ratlas_nii)
	masker.fit()
	cur_FS = masker.transform(nii_fake4D)
	FS.append(cur_FS)

# get rid of useless dimension
FS = np.array(FS).squeeze()
assert np.isnan(FS).sum() == 0
assert np.logical_not(np.any(FS==0))
assert len(df["MRI_path"].values) == len(FS)
# standardize the volumes
FS_ss = StandardScaler().fit_transform(FS)
assert np.logical_not(np.any(np.isnan(FS_ss)))
assert np.logical_not(np.any(FS_ss==0))

# grey matter quantity per ROI for each subject used as input for analysis, shape=(86, 100)
X_brain = FS_ss
np.save("C:\\sexualorientproject\\mri_FS_ss", FS)

#######################
### DECONFOUNDING #####
#######################

print("Remove variance explained by confounds in grey matter quantities")
# extract confound information
confounds = [
	"Biological Sex",
	"Age",
	"EducationalLevel",
	"Handedness"
    	     ]
my_confound = df[confounds].values

# actual signal deconfounding 
X_brain = clean(signals=X_brain, confounds=my_confound,
	detrend=False, standardize=False)


##################
### MODELING #####
##################

# starting the modelisation
#run the CV 5 fold logistic regression

# keep information for the confusion matrix
all_y_pred = []
all_y_true = []

# save accuracies and model coefficients
sample_accs = []
sample_coefs = []

for i_subsample in range(100):
	clf = LogisticRegression()
	sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=i_subsample)
	sss.get_n_splits(X_brain, Y)
	for train_index, test_index in sss.split(X_brain, Y):
		X_train, X_test = X_brain[train_index], X_brain[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		acc = (y_pred == y_test).mean()
		sample_accs.append(acc)
		sample_coefs.append(clf.coef_[0, :])
		# extract results for the confusion matrix
		all_y_pred.append(y_pred)
		all_y_true.append(y_test)

# Store the results
acc = np.mean(sample_accs)
acc_std = np.std(sample_accs)

# print and save accuracy as txt
print("acc mean = {}, std = {}".format(acc, acc_std))
text_file = open("final_acc_mri.txt", "w")
n = text_file.write("acc mean = {}, std = {}".format(acc, acc_std))
text_file.close()

final_coefficients = np.mean(sample_coefs, axis=0)
final_coefficients_std = np.std(sample_coefs, axis=0)

# format results for the confusion matrix
all_y_pred_ = np.squeeze(np.array(all_y_pred).reshape(1,4500))
all_y_true_ = np.squeeze(np.array(all_y_true).reshape(1,4500)) 


# save results as csv to check when rerun the script #reproducibility
coefs_to_save = np.array([final_coefficients, final_coefficients_std]).T
df_coef_per_roi_mri = pd.DataFrame(data=coefs_to_save, columns=["Coef", "Coef_std"], index=atlas.labels)
df_coef_per_roi_mri.to_excel("coef_per_roi_mri.xlsx")


# niftiing the MRI results
final_coefficients_nii3D = masker.inverse_transform(final_coefficients.reshape(1, 100))
final_coefficients_nii3D.to_filename("final_coefficients3D_mri.nii") # transform as nii and save

final_coefficients_std_nii3D = masker.inverse_transform(final_coefficients_std.reshape(1, 100))
final_coefficients_std_nii3D.to_filename("final_coefficients_std3D_mri.nii") # transform as nii and save


################################
#### CONFUSION MATRIX ##########
################################

def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
	''' Plotting function for the x axis labels to be centered
	with the plot ticks

	Parameters
	----------
	See stackoverflow 
	https://stackoverflow.com/questions/27349341/how-to-display-the-x-axis-labels-in-seaborn-data-visualisation-library-on-a-vert
	
	'''
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


# compute and plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
class_names = ["Homosexual", "Heterosexual"]

# matrix
cm = confusion_matrix(all_y_true_, all_y_pred_)
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
plt.xticks(tick_marks, class_names, fontsize=20)
plt.yticks(tick_marks, class_names, fontsize=20)
rotateTickLabels(ax, -55, 'x')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
             horizontalalignment="center",
             color= "black", fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.ylabel("True label", fontsize=20)
plt.tight_layout()
plt.savefig('confusion_matrix_mri.png')
plt.show()


##############################################
####### Non-parameric hypothesis test ########
##############################################

# Non-parameric hypothesis test 
# run the CV 5 fold logistic regression with permutated Y

n_permutations = 100
perm_rs = np.random.RandomState(0)
permutation_accs = []
permutation_coefs = []
for i_iter in range(n_permutations):
	print(i_iter + 1)
	Y_perm = perm_rs.permutation(Y)
	clf = LogisticRegression()
	sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=i_iter)
	sss.get_n_splits(X_brain, Y_perm)
	for train_index, test_index in sss.split(X_brain, Y_perm):
		X_train, X_test = X_brain[train_index], X_brain[test_index]
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
significant_weights_3D = masker.inverse_transform(np.array(significant_weights).reshape(1, 100))
significant_weights_3D.to_filename('significantROIs_mri.nii')


# plot heat map of coefficients
from matplotlib import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_excel("sMRI_results/coef_per_roi_mri.xlsx")
plt.figure(figsize=(12, 5))
ax = sns.heatmap(df["Coef"].values.reshape(1, 100), cmap=plt.cm.RdBu_r, cbar_kws={"shrink": 0.25}, 
			center=0, annot=True, annot_kws={'size': 4}, square=True)
tick_marks = np.arange(len(df["Unnamed: 0"].values))
plt.xticks(tick_marks + 0.5, list(df["Unnamed: 0"].values), fontsize=4)
plt.yticks([0], ["Weights"], fontsize=8, rotation=90)
ax.xaxis.set_ticks_position('top')
rotateTickLabels(ax, 45, 'x')
plt.title('Homosexual vs heterosexual subjects: classification contributions', fontsize=13)
plt.tight_layout()
plt.savefig('sMRI_results/coef_mri_heatmap.png', DPI=500)
plt.show()


df = pd.read_excel("sMRI_results/coef_per_roi_mri.xlsx")
df = df.set_index("Unnamed: 0")

sign_rois = ["b'7Networks_LH_Vis_7'", "b'7Networks_LH_SomMot_4'",
			"b'7Networks_LH_Default_pCunPCC_2'", "b'7Networks_RH_Vis_2'",
			"b'7Networks_RH_SomMot_6'", "b'7Networks_RH_SomMot_8'",
			"b'7Networks_RH_SalVentAttn_TempOccPar_1'",
			"b'7Networks_RH_Cont_PFCl_4'"]

df = df.loc[sign_rois]

plt.figure(figsize=(8, 5))
ax = sns.heatmap(df["Coef"].values.reshape(1, -1), linewidths=2, cmap=plt.cm.RdBu_r, cbar_kws={"shrink": 0.25}, 
			center=0, annot=True, annot_kws={'size': 12}, square=True)
tick_marks = np.arange(len(df.index))
plt.xticks(tick_marks + 0.5, sign_rois, fontsize=12)
plt.yticks([], fontsize=12, rotation=90)
ax.xaxis.set_ticks_position('top')
rotateTickLabels(ax, 45, 'x')
plt.title('Classification contributions of significant ROIs', fontsize=13)
plt.tight_layout()
plt.savefig('sMRI_results/coef_mri_heatmap_sign.png', DPI=500)
plt.show()