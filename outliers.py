with open(os.path.join("Analysis", 'cadence','outliers.pickle'),'rb') as fileobj:
  outliers = pickle.load(fileobj)
for subj in outliers:
  metadata = metadata[metadata['subjectID']!=subj]
print(metadata[metadata['inout']=='indoors'])

df = one_subject_summary(24)