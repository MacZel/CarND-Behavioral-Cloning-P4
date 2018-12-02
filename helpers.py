import cv2

def preprocess(image):
    # crop image to remove irrelevant data
    preprocessed_image = image[55:135,:,:]
    # apply gaussian blur to reduce noise (based on Nvidia paper)
    preprocessed_image = cv2.GaussianBlur(preprocessed_image, (3,3), 0)
    # resize (based on Nvidia paper)
    preprocessed_image = cv2.resize(preprocessed_image,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV colorspace (based on Nvidia paper)
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2YUV)
    
    return preprocessed_image