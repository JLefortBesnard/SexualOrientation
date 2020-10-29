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
  	cur_FS = clean(signals=cur_FS, 
		detrend=False, standardize=False) 
	# shape (121, 100)
  	
	# substract neighbor in time 
  	cur_FS = subtract(cur_FS) 
	# shape (120, 100)
  	
	# compute cross correlation
  	sub_cross_corr = np.corrcoef(cur_FS.T) 
	# shape (100, 100)
  	
	# extract results per ROI
	sub_cross_corr_per_ROI = extract_ROIconn(sub_cross_corr) 
	# shape (100, 99)

  	# save results for each subject
  	FS.append(sub_cross_corr_per_ROI) 

FS = np.array(FS).squeeze() 
# shape (86, 100, 99)
