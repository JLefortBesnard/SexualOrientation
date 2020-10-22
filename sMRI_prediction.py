import glob
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn.datasets as ds
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
from matplotlib import pylab as plt
from nilearn.signal import clean


np.random.seed(0)



######################################
##### EXTRACTING DEMOGRAPHIC DATA ####
######################################

# gather all structural imgs
sMRI_paths = glob.glob('Anatomy/*.nii.gz')

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
df["MRI_path"] = ['Anatomy/highres2standard_{}.nii.gz'.format(i) for i in df.index]



###########################################
##### EXTRACTING STRUCTURAL BRAIN DATA ####
###########################################

tmp_nii = nib.load(df["MRI_path"][1])
atlas = ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1)

ratlas_nii = resample_img(
  atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
ratlas_nii.to_filename('debug_ratlas.nii.gz')

# extracting data MRI
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
FS = np.array(FS).squeeze()
assert len(df["MRI_path"].values) == len(FS)

FS_ss = StandardScaler().fit_transform(FS)
assert np.logical_not(np.any(np.isnan(FS_ss)))


X_brain = FS_ss


#########################################
### ACTUAL MODELING + DECONFOUNDING #####
#########################################

# extraction confound information
confounds = [
	"Biological Sex",
	"Age",
	"EducationalLevel",
	"Handedness"
    	     ]
my_confound = df[confounds].values

# reload copy of original dataset
X_brain_ = X_brain.copy()
Y_ = Y.copy()

# actual signal deconfounding 
X_brain_ = clean(signals=X_brain_, confounds=my_confound,
	detrend=False, standardize=False)

# starting the modelisation
#run the CV 5 fold logistic regression
all_y_pred = []
all_y_true = []
sample_accs = []
sample_coefs = []
for i_subsample in range(100):
	clf = LogisticRegression()
	sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=i_subsample)
	sss.get_n_splits(X_brain_, Y_)
	for train_index, test_index in sss.split(X_brain_, Y_):
		X_train, X_test = X_brain_[train_index], X_brain_[test_index]
		y_train, y_test = Y_[train_index], Y_[test_index]
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		acc = (y_pred == y_test).mean()
		sample_accs.append(acc)
		sample_coefs.append(clf.coef_[0, :])
		all_y_pred.append(y_pred)
		all_y_true.append(y_test)

# Store the results
print(np.mean(sample_accs))
print(np.std(sample_accs))
Weight_results = np.mean(sample_coefs, axis=0)
all_y_pred_ = np.squeeze(np.array(all_y_pred).reshape(1,4500))
all_y_true_ = np.squeeze(np.array(all_y_true).reshape(1,4500)) 


################################
#### CONFUSION MATRIX ##########
################################

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

from sklearn.metrics import confusion_matrix
import itertools

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
print(cm)
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
plt.savefig('confusion_matrix_deconfounded.png', PNG=300)
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
	Y_perm = perm_rs.permutation(Y_)
	clf = LogisticRegression()
	sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=i_iter)
	sss.get_n_splits(X_brain_, Y_perm)
	for train_index, test_index in sss.split(X_brain_, Y_perm):
		X_train, X_test = X_brain_[train_index], X_brain_[test_index]
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
	if Weight_results[n_roi] < below or Weight_results[n_roi] > above:
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
		significant_weights.append(Weight_results[i])
		print(roi, Weight_results[i])
		dic[roi] = Weight_results[i]
	else:
		significant_weights.append(0)


################################
#### Brain visualisation #######
################################

# extract brain results for visualization
Weight_results = np.mean(sample_coefs, axis=0)
MRI_Results = masker.inverse_transform(Weight_results.reshape(1, 100))
MRI_Results.to_filename('out_nii_so_deconfounded.nii')


# extract significant results to nii for visualisation
MRI_Results = masker.inverse_transform(np.array(significant_weights).reshape(1, 100))
MRI_Results.to_filename('significant_nii_so_deconfounded.nii')

