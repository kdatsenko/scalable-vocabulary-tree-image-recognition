# vocabulary-tree-image-recognition

A recognition pipeline for performing image recognition, retrieval, and localization with vocabulary trees. The task is to recognize DVD covers in real-world images based on a database of images. The system accepts a query image as the input, and should accurately identify the DVD cover in the database that is same cover, but from a regularized viewpoint.

Instructions for running:
---
Before running:  
Folder dvd_covers contains all image testing and training sets.  
Make sure the dataset folder with the Canon, E63, Droid, and Palm test sub-folders is named "dvd_covers" AND is in the same directory as the main program file main_pipeline.  
Lines 3-5 in main_pipeline.m control the dataset directories:  
%Setup: data folders  
db_dir = 'dvd_covers/Reference';  
test_dir='dvd_covers/Canon';  

To run, call the function main_pipeline()  
To set the query image for testing change queries = [3] at line 7 of the main_pipeline routine.  
To see the improvements, change the flag improvement =0; at line 18.  

To retrain the vocabulary tree from scratch, delete the .mat files from the working directory. Do this also before modifying the flag improvement =0; at line 18.  
