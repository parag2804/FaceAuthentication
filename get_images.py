# Load imports
import os 
import cv2

def get_images(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    # read the image using openCV                    
                    img = cv2.imread(
                            os.path.join(image_directory, subfolder, file)
                            )                   
                    width = 100
                    height = 100
                    dim = (width, height)
                    img = cv2.resize(img, dim)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    X.append(img)
                    y.append(subfolder)
    
    print("All images are loaded")     

    return X, y
                    
