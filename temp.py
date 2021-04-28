import os

s3_imgs_dir = '/d_drive_data/S3/L2_WFR'

# adding every image inside the working directory to a list
files = os.listdir(s3_imgs_dir)

# adding the complete path to each image folder listed + xml file
fullpath_xmls = [os.path.join(s3_imgs_dir, image+'/xfdumanifest.xml') for image in files]

print(fullpath_xmls)