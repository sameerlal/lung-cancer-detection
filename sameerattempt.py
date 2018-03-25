import os
import SimpleITK as sitk
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 
from scipy import ndimage as nd


def load_itk(filename):
	return sitk.ReadImage(filename)


def get_image_spacing(itkimg):
	spacing_array = np.array(itkimg.GetSpacing())
	return spacing_array

def get_image_origin(itkimg):
	return np.array(itkimg.GetOrigin())

def get_voxel_spacing(itkimg):
	spacing = np.array(itkimg.GetSpacing())
	return spacing[0], spacing[1]

def resample_image(im, spacing):
	return nd.zoom(im, spacing)

def visualize_slice(np_array):
	imgplot = plt.imshow(np_array[120])
	plt.show()

def get_slice_thickness(itkimg):
	return np.array(itkimg.GetSpacing())[2]

def resample(im, spacing):
	"""
	Resample the input image to 1mm x 1mm x 1mm spacing.

	:param im: numpy array of Image generated from a .mhd file.
	:param spacing: array of spacing in z, y, x, dimensions.

	:return: A resampled image, as a numpy array.
	"""
	return nd.zoom(im, spacing)

def normalize(im):
	"""
	Normalize a numpy array

	:param im: a nunmpy array to be normalized
	"""
	im_min = np.min(im)
	im_max = np.max(im)
	return (im  - im_min)/(im_max - im_min)



def main():
	itk_img = load_itk('Traindata_small/train_1.mhd')
	np_scan = sitk.GetArrayFromImage(itk_img)
	print('Image Spacing:  ', get_image_spacing(itk_img))
	print('Image Origin:  ', get_image_origin(itk_img))
	print('Voxel Spacing:  ', get_voxel_spacing(itk_img))
	print('Slice Thickness:  ', get_slice_thickness(itk_img))
	resampled_img = resample(np_scan, get_image_spacing(itk_img)[::-1])
	normalized_img = normalize(resampled_img)
	
	#print(normalized_img.shape)

	visualize_slice(normalized_img)
	
	# with open('normalized_data.txt', 'wb') as outfile:
	# 	for slice_2d in normalized_img:
	# 		np.savetxt(outfile, slice_2d)

if __name__ == '__main__':
	main()