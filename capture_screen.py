from datetime import datetime
import os

import numpy as np
from PIL import ImageGrab
import cv2
import matplotlib.pyplot as plt

while(True):
    frame = np.array(ImageGrab.grab(bbox=(0,40,1200,620)))
    
    #cv2.imshow('window',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    frame_filename = os.path.join('frames', timestamp)
    plt.imsave(f'{frame_filename}.jpg', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break