import numpy as np
import cv2

a = np.zeros((3,3,3), dtype=np.uint8)  # (c, h, w)

a[:,0,0] = [255, 255, 255]
a[:,1,0] = [0, 255, 255]
a[:,2,0] = [255, 0, 255]
a[:,0,1] = [255, 255, 0]
a[:,1,1] = [0, 0, 255]
a[:,2,1] = [0, 255, 0]
a[:,0,2] = [255, 0, 0]
a[:,1,2] = [0, 0, 0]
a[:,2,2] = [255, 255, 255]

a = a.transpose(1, 2, 0)

b = np.zeros((2,2,3), dtype=np.uint8)
b[0,0] = [255, 255, 255]
b[0,1] = [0, 0, 255]
b[1,0] = [0, 255, 0]
b[1,1] = [255, 0, 0]

cv2.imwrite("test17_0.png", a)

dst2 = cv2.resize(a, (a.shape[1]*2, a.shape[0]*2), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("test17_1.png", dst2)

dst4 = cv2.resize(a, (a.shape[1]*4, a.shape[0]*4), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("test17_2.png", dst4)

cv2.imwrite("test17b_0.png", b)

dst2 = cv2.resize(b, (b.shape[1]*2, b.shape[0]*2), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("test17b_1.png", dst2)

dst4 = cv2.resize(b, (b.shape[1]*4, b.shape[0]*4), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("test17b_2.png", dst4)


u = 1
v = 1
z = 2
d = np.zeros((z*u,z*v,3), dtype=np.uint8)
for x in range(u):
    for y in range(v):
        for i in range(z):
            for j in range(z):
                d[x*z+i,y*z+j] = ((1-(i+0.5)/z) * (1-(j+0.5)/z) * b[x, y] +
                          (1-(i+0.5)/z) * (j+0.5)/z * b[x, y+1] +
                          (1-(j+0.5)/z) * (i+0.5)/z * b[x+1, y] +
                          (i+0.5)/z * (j+0.5)/z * b[x+1,y+1])

cv2.imwrite("test17c_1.png", d)

u = 1
v = 1
z = 4
d = np.zeros((z*u,z*v,3), dtype=np.uint8)
for x in range(u):
    for y in range(v):
        for i in range(z):
            for j in range(z):
                d[x*z+i,y*z+j] = ((1-(i+0.5)/z) * (1-(j+0.5)/z) * b[x, y] +
                          (1-(i+0.5)/z) * (j+0.5)/z * b[x, y+1] +
                          (1-(j+0.5)/z) * (i+0.5)/z * b[x+1, y] +
                          (i+0.5)/z * (j+0.5)/z * b[x+1,y+1])

cv2.imwrite("test17c_2.png", d)


u = 2
v = 2
z = 2
d = np.zeros((z*u,z*v,3), dtype=np.uint8)
for x in range(u):
    for y in range(v):
        for i in range(z):
            for j in range(z):
                d[x*z+i,y*z+j] = ((1-(i+0.5)/z) * (1-(j+0.5)/z) * a[x, y] +
                          (1-(i+0.5)/z) * (j+0.5)/z * a[x, y+1] +
                          (1-(j+0.5)/z) * (i+0.5)/z * a[x+1, y] +
                          (i+0.5)/z * (j+0.5)/z * a[x+1,y+1])

cv2.imwrite("test17d_1.png", d)

u = 2
v = 2
z = 4
d = np.zeros((z*u,z*v,3), dtype=np.uint8)
for x in range(u):
    for y in range(v):
        for i in range(z):
            for j in range(z):
                d[x*z+i,y*z+j] = ((1-(i+0.5)/z) * (1-(j+0.5)/z) * a[x, y] +
                          (1-(i+0.5)/z) * (j+0.5)/z * a[x, y+1] +
                          (1-(j+0.5)/z) * (i+0.5)/z * a[x+1, y] +
                          (i+0.5)/z * (j+0.5)/z * a[x+1,y+1])

cv2.imwrite("test17d_2.png", d)








