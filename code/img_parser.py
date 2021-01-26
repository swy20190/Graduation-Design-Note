# This file parse RGB images into joint coordinates
import cv2
import numpy as np


def parse(protoFile, weightsFile, nPoints, image):
    """
    parse RGB image into joints coordinate data
    :param protoFile: prototxt
    :param weightsFile: trained model
    :param nPoints: corresponds to the model
    :param image: RGB file
    :return: a list of (x, y) which is joint coordinate
    """
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    # Read the network into memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # Read image
    frame = cv2.imread(image)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        # find global maxima of the probMap
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # add thr point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # draw
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
    cv2.imwrite('Output-Keypoints.png', frameCopy)
    cv2.imwrite('Output-Skeleton.png', frame)
    cv2.waitKey(0)
    return points
