# This file generates raw joint coordinate data
from img_parser import parse
import cv2

protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
# Read the network into memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
nPoints = 18
line = parse(net, nPoints, "images/test.png")
print(line)
