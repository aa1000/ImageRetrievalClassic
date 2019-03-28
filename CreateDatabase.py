# Imports for image maniplation
import cv2
import numpy as np

# Imports for file maniplation
from glob2 import glob
import os, os.path
from platform import platform

# Imports for math and extracting features
from sklearn.cluster import KMeans
import time
import ImageUtils as IU

# speed-up opencv using multithreads
cv2.setUseOptimized(True)
cv2.setNumThreads(8)


# Calculate the time the generation took
t0 = time.time()
print('Exploring folders...')
# get the path we are currently in
folders_path = os.path.realpath('') + '/'
# get all folders in path
folders = glob(folders_path + '*/')

print('Exploring files...')

img_files = []

# now only get image files in each folder (category)
for folder in folders:
    files_path = folder
    print('Folder: ' + files_path)
    
    img_files.extend(glob(files_path + '*.JPG'))
    img_files.extend(glob(files_path + '*.JPEG'))
    img_files.extend(glob(files_path + '*.BMP'))
    img_files.extend(glob(files_path + '*.PNG'))
    
    # windows is case insensitive so we don't need to add this
    if not platform().startswith('Windows'):
        img_files.extend(glob(files_path + '*.jpg'))
        img_files.extend(glob(files_path + '*.jpeg'))
        img_files.extend(glob(files_path + '*.bmp'))
        img_files.extend(glob(files_path + '*.png'))
        
if len(img_files) > 0:
    print('Search complete,', len(img_files), 'image(s) found.')
else:
    print('No images found, exiting script.')
    exit()


print('Creating pixel features for color indexing...')
# a vector containing the pixels of all images to calculate the color indexing with
pixels_vector = IU.ImgPathToPixelVector(img_files[0])
# read all images and create a vector of them to index
# Ignore the first element since we already calculated that
if len(img_files) > 1:
    for i in range(1, len(img_files)):
        img_file = img_files[i]
        reshaped_image = IU.ImgPathToPixelVector(img_file)
        pixels_vector = np.vstack((pixels_vector, reshaped_image))

        percent = (i+1)/len(img_files) * 100.0
        percent_text = 'Preparing indexed color data ' + str(int(percent)) + '%'
        print(percent_text, end='\r', flush=True)                                                 


print('Calculating cluster centers, this might take a while...')
# It's easier to reconstruct sklearn's model data than opencv's model data
# Having more than 1 job or a full algorithm causes a Memory error on a 16 GB RAM
# If you have a slow PC you might also want to reduce n_iter since 10 takes over 10 hours with 1 job
# However, if you have more memory you might try to increase the number of jobs to make up for it
# An approximation is more than enough, no need to have high accuracy and waste computing time
kmeans = KMeans(n_clusters=IU.n_indexed_colors, n_init =1, tol=0.001, max_iter=100, random_state=0, n_jobs=1, algorithm='full')
kmeans.fit(pixels_vector)
#labels = kmeans.labels_
centers = kmeans.cluster_centers_
print('Cluster centers\' calculation complete.')


# save the centers to be able to reconstruct the model later
print('Saving indexed color classes...')
centers = np.uint8(centers)
np.save(IU.histsogram_centers_file_name, centers)

# clear the list when no longer needed (to save memory?)
pixels_vector = None

# create the feature vector for every image
for i, img_file in enumerate(img_files):
    # read image file
    img = cv2.imread(img_file, 1)
    # convert all images to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    features_vector = IU.CreateImageFeaturesVector(img, centers)
    
    # Save the features vector to a file
    features_vector_file_name = img_file + '.npy'
    np.save(features_vector_file_name, features_vector)
    
    percent = (i+1)/len(img_files) * 100
    percent_text = 'Creating images feature vector ' + str(int(percent)) + '%'
    print(percent_text, end='\r', flush=True)

t1 = time.time()
print('Database creating complete, time elapsed:', t1-t0)

