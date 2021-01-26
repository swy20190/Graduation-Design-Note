# This file generates raw joint coordinate data
from img_parser import parse

protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
nPoints = 18
line = parse(protoFile, weightsFile, nPoints, "images/test.png")
print(line)
