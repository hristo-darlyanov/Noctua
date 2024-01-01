from transformers import DPTImageProcessor, DPTForDepthEstimation, DPTFeatureExtractor, AutoImageProcessor
import torch
import numpy as np
from PIL import Image
import requests
import time
from transformers import pipeline
import matplotlib.pyplot as plt
import cv2
processor = DPTImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256")
cap = cv2.VideoCapture(0)
pipe = pipeline(task="depth-estimation", model="Intel/dpt-swinv2-tiny-256")
while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    frame_pil = Image.fromarray(frame)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'image preprocessing took {elapsed_time:.4f} seconds to finish.')
    
    start_time = time.time()
    result = pipe(frame_pil)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'prediction took {elapsed_time:.4f} seconds to finish.')

    #cv2.imshow("CV2Frame", frame)
    plt.imshow(result['depth'])
    plt.pause(0.00001)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()