'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_locations = face_recognition.face_locations(gray_image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        detection_results.append([float(left), float(top), float(right - left), float(bottom - top)])
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    count = 0
    face_encoding_results = {}
    face_encodings = np.array([]).reshape(0, 128)
    for img_name in imgs.keys():
        face_encoding = np.array(face_recognition.face_encodings(imgs[img_name]))
        if face_encoding.shape[0] == 0:
            count += 1
            continue
        face_encodings = np.concatenate((face_encodings, face_encoding), axis=0)
        if face_encoding.shape[0] > 1:
            for i in range(face_encoding.shape[0]):
                face_encoding_results[img_name + '_' + str(i)] = np.reshape(face_encoding[i], (1, 128))
        else:
            face_encoding_results[img_name] = np.reshape(face_encoding, (1, 128))
    
    centroids = []
    centroids_names = np.random.choice(np.array(list(face_encoding_results.keys())), K, replace=False)
    for n in centroids_names:
        centroids.append(face_encoding_results[n])

    centroids = np.array(centroids)
    centroids = np.reshape(centroids, (centroids.shape[0], 128))
    cluster_results = get_cluster_results(face_encoding_results, centroids, K)
    while True:
        new_centroids = get_centroids(face_encoding_results, cluster_results)
        new_cluster_results = get_cluster_results(face_encoding_results, new_centroids, K)
        if np.array_equal(cluster_results, new_cluster_results):
            break
        cluster_results = new_cluster_results
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)
def get_distance(face_encoding: np.ndarray, centroids: np.ndarray):
    distances = []
    for centroid in centroids:
        distance = np.linalg.norm(face_encoding - centroid)
        distances.append(distance)
    return distances

def get_centroids(face_encodings: dict, cluster_results):
    centroids = []
    for cluster in cluster_results:
        cluster_images = [face_encodings[i] for i in cluster]
        cluster_images = np.array(cluster_images)
        cluster_images = np.reshape(cluster_images, (cluster_images.shape[0], 128))  
        centroids.append(np.reshape(np.mean(cluster_images, axis=0), (1, 128)))
    centroids = np.array(centroids)
    centroids = np.reshape(centroids, (centroids.shape[0], 128))
    return np.array(centroids)

def get_cluster_results(face_encodings: dict, centroids: np.ndarray, K: int):
    cluster_results = [[] for i in range(K)]
    for i in face_encodings.keys():
        face_encoding = face_encodings[i]
        distances = get_distance(face_encoding, centroids)
        cluster_results[np.argmin(distances)].append(i)
    return cluster_results