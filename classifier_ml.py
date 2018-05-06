"""
Module for training machine learning classifier.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""
from sklearn.naive_bayes import GaussianNB

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import SimpleITK as sitk

import util
import plot
import nodule_finder


def make_data_set(candidate_nodules, training_file):
	"""Make data set."""
	pass



def ml_train():
	"""Build ML classifer using training data."""
	data = make_data_set()
	gnb = GaussianNB()  # Initialize classifier
	model = gnb.fit(train, train_labels)
	preds = gnb.predict(test)
	print(preds)




