import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from copy import deepcopy


class Group:
    def __init__(self, id_, element, image, im_shape):
        self.__id = id_
        self.__elements = []
        self.__connections = {}
        self.__elements.append(element)
        self.__mean = np.float64(image[element[0], element[1], ...])
        self.__valid = 1
        if element[0] > 0:
            self.__connections[self.__id - im_shape[1]] = 1
        if element[0] < im_shape[0] - 1:
            self.__connections[self.__id + im_shape[1]] = 1
        if element[1] > 0:
            self.__connections[self.__id - 1] = 1
        if element[1] < im_shape[1] - 1:
            self.__connections[self.__id + 1] = 1

    def union(self, grp):
        for i in grp.get_elements():
            self.__elements.append(i)

    def get_elements(self):
        return self.__elements

    def get_num_elements(self):
        return len(self.__elements)

    def add_connection(self, index, conn):
        if index not in self.__connections.keys():
            self.__connections[index] = conn
        else:
            self.__connections[index] += conn

    def set_connection(self, index, conn):
        self.__connections[index] = conn

    def rm_connection(self, index):
        self.__connections.pop(index)

    def get_connection(self):
        return self.__connections

    def set_mean(self, mean):
        self.__mean = mean

    def get_mean(self):
        return self.__mean

    def get_id(self):
        return self.__id

    def invalid(self):
        self.__valid = 0

    def is_valid(self):
        return self.__valid


def region_fusion_minimization(signal, lamda):
    '''
    Region Fusion Minimization algorithm.
    :param signal: Signal I of length M
    :param lamda: Sparseness parameter
    :return: Filtered signal S of length M
    '''

    im_shape = signal.shape
    signal_copy = deepcopy(signal)
    # =========== Initialize ================
    # initialize groups
    groups = []
    for y in range(im_shape[0]):
        for x in range(im_shape[1]):
            # Note: element [y, x]
            grp = Group(x + y*im_shape[1], [y, x], signal_copy, im_shape)
            groups.append(grp)

    # initialize beta, iter
    beta = 0
    iter = 0

    while beta < lamda :
        i = 0
        while i < len(groups):
            if not groups[i].is_valid():
                i += 1
                continue
            for j in list(groups[i].get_connection()):
                if not groups[j].is_valid():
                    continue
                wi = groups[i].get_num_elements()
                wj = groups[j].get_num_elements()
                yi = groups[i].get_mean()
                yj = groups[j].get_mean()
                cij = groups[i].get_connection()[groups[j].get_id()]
                if wi*wj*np.sum((yi - yj)**2) <= beta*cij*(wi + wj):
                    # Fusion two groups
                    groups[i].union(groups[j])
                    groups[i].set_mean((wi*yi + wj*yj)/(wi + wj))
                    groups[i].rm_connection(j)
                    groups[j].rm_connection(i)
                    for k in groups[j].get_connection().keys():
                        if k in groups[i].get_connection().keys():
                            groups[i].add_connection(k, groups[j].get_connection()[k])
                            groups[k].add_connection(i, groups[j].get_connection()[k])
                        else:
                            groups[i].set_connection(k, groups[j].get_connection()[k])
                            groups[k].set_connection(i, groups[j].get_connection()[k])

                        groups[k].rm_connection(j)
                    groups[j].invalid()

            i += 1
        iter += 1
        beta = (iter/30)*lamda

        print('beta : {}'.format(beta))

    return reconstruct_output(groups, signal)


def reconstruct_output(groups, signal):
    S = np.zeros(signal.shape)

    print('This is {} pixels in all'.format(signal.size))
    count = 0
    num = 0
    for group in groups:
        if group.is_valid():
            num += 1
            for j in group.get_elements():
                count += 1
                S[j[0], j[1], ...] = group.get_mean()

    print('Count = {}, Num = {}'.format(count, num))
    return S


img_dir2 = 'C:\\Users\\Dell\\Desktop\\tmp\\noise.png'
img = io.imread(img_dir2)

img2 = np.uint8(region_fusion_minimization(img, 20000))
plt.imshow(img2)
plt.axis('off')
io.imsave('C:\\Users\\Dell\\Desktop\\tmp\\esss.jpg',img2)
plt.show()
