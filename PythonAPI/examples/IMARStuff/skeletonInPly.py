# Given a PFNN pose with coordinates this script can save it in a PLY format

from plyfile import PlyData, PlyElement
import numpy as numpy
import numpy as np
import os
'''
plydata = PlyData.read('test.ply')
print(plydata)

print ("=============")
print(plydata['vertex'])
print ("=============")
print(plydata['vertex']['x'])
print ("=============")
for i in range (8):
    print(plydata['vertex'][i]['x'])
'''

def writeBone(boneIndex, x0, y0, z0, x1, y1, z1, sizeFactor):

    sx = sizeFactor
    sy = sizeFactor

    P0 = numpy.array([x0, y0, z0])
    P1 = numpy.array([x1, y1, z1])

    vertices = [tuple(P0),
                tuple(P1),
                tuple(P1 + np.array([0, sy, 0])),
                tuple(P0 + np.array([0, sy, 0])),
                tuple(P0 + np.array([sx, 0, 0])),
                tuple(P1 + np.array([sx, 0, 0])),
                tuple(P1 + np.array([sx, sy, 0])),
                tuple(P0 + np.array([sx, sy, 0])),
                ]

    '''
    vertices = [(0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 1),
                (1, 1, 0)]

    '''

    faces = [([0, 1, 2, 3], 255, 255, 255),
             ([7, 6, 5, 4], 255, 255, 255),
             ([0, 4, 5, 1], 255, 255, 255),
             ([1, 5, 6, 2], 255, 255, 255),
             ([2, 6, 7, 3], 255, 255, 255),
             ([3, 7, 4, 0], 255, 255, 255)]

    faceIndex = boneIndex * 8
    for x in faces:
        for j in range(4):
            x[0][j] += faceIndex


    return vertices, faces

def tet_ply():

    vertices = []
    faces = []

    vb0, fb0 = writeBone(0, 0,0,0, 0,0,4)
    vb1, fb1 = writeBone(1, 0,0,0, 1,4,2)
    vertices += vb0
    vertices += vb1
    faces += fb0
    faces += fb1

    vertex = numpy.array(vertices,
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    face = numpy.array(faces,
                       dtype=[('vertex_indices', 'i4', (4,)),
                              ('red', 'u1'), ('green', 'u1'),
                              ('blue', 'u1')])

    data = PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['a skeleton pfnn vertices']
            ),
            PlyElement.describe(face, 'face')
        ],
        text="some text",
        comments=['a skeleton pfnn']
    )

    print(data['vertex'][0])
    data.write('test.ply')
    return data

#str = tet_ply("Text")


#import pickle
#data = pickle.load( open( "agent_bremen_0000143_0.p", "rb" ), encoding="latin1" )
#data = data


def writePFNNSkeletonPly(poseData, outTextFile, scaleConstant, sizeFactor):
    # Invert Y and Z axis and convert to meters
    for i in range(31):
        y = poseData[i*3 + 1]
        z = poseData[i*3 + 2]
        poseData[i*3 + 1] = z
        poseData[i*3 + 2] = y

        for j in range(3):
            poseData[i*3 + j] *= scaleConstant

    parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]
    vertices = []
    faces = []

    counter = 0
    for boneId in range(31):
        boneParent = parents[boneId]
        if boneParent == -1:
            continue

        x0, y0, z0 = poseData[boneId * 3 + 0],  poseData[boneId * 3 + 1], poseData[boneId * 3 + 2]
        x1, y1, z1 = poseData[boneParent * 3 + 0],  poseData[boneParent * 3 + 1], poseData[boneParent * 3 + 2]

        vb, fb = writeBone(counter, x0, y0, z0, x1, y1, z1, sizeFactor)
        counter += 1

        vertices += vb
        faces += fb

    vertex = numpy.array(vertices,
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    face = numpy.array(faces,
                       dtype=[('vertex_indices', 'i4', (4,)),
                              ('red', 'u1'), ('green', 'u1'),
                              ('blue', 'u1')])

    data = PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['a skeleton pfnn vertices']
            ),
            PlyElement.describe(face, 'face')
        ],
        text="some text",
        comments=['a skeleton pfnn']
    )
    data.write(outTextFile)

def main():
    sizeFactor = 0.05  # Line size Factor
    posesToSave = [0, 12, 16, 19, 24, 41, 35, 94, 123, 141]
    for pose in posesToSave:
        fileName = 'pose' + str(pose)
        poseData = np.loadtxt(fileName + '.txt', delimiter=',')
        writePFNNSkeletonPly(poseData, fileName + '.ply', 0.01, sizeFactor)  # Original data in centimeters. This means that avg hight is ~ 1.6 m with actual data

if __name__== "__main__":
  main()