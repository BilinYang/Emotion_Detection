from config import emotion_config as config
from pyimagesearch.io import HDF5DatasetWriter
import numpy as np

print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__() 
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

for row in f:
	(label, image, usage) = row.strip().split(",")
	label = int(label)

	# if we are ignoring the "disgust" class there will be 6 total
	# class labels instead of 7
	if config.NUM_CLASSES == 6:
		# merge the "anger" and "disgust classes
		if label == 1:
			label = 0

		# if label has a value greater than zero, subtract one from
		# it to make all labels sequential 
		if label > 0:
			label -= 1

	image = np.array(image.split(" "), dtype="uint8")
	image = image.reshape((48, 48))

	if usage == "Training":
		trainImages.append(image)
		trainLabels.append(label)

	elif usage == "PublicTest":
		valImages.append(image)
		valLabels.append(label)

	else:
		testImages.append(image)
		testLabels.append(label)

datasets = [
	(trainImages, trainLabels, config.TRAIN_HDF5),
	(valImages, valLabels, config.VAL_HDF5),
	(testImages, testLabels, config.TEST_HDF5)]

for (images, labels, outputPath) in datasets:
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

	for (image, label) in zip(images, labels):
		writer.add([image], [label])

	writer.close()

f.close()