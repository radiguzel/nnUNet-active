from re import X
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def d_content(I,J):
    """
    Calculate content distance between I and J.
    :param I: An image 
    :param J: Another image
    return: return 1 minus the mean Euclidean squared distance
    """
    assert I.shape == J.shape
    diff = (I - J) ** 2
    return (1 - diff.sum() / diff.size)

def group_to_image_similarity(S, I, is_content_s=True):
    """
    :param S: Group
    :param I: Image
    :param is_content_s: (Boolean) If true, content distance funciton is used to calculate the similarity between two images.
    :return max_sim: Maximum similarity between image I and an image that is most similar to I in group S.
    """
    max_sim = -1
    for I_s in S:
        if is_content_s:
            similarity = d_content(I_s,I)
        else:
            similarity = cosine_similarity(I_s, I)
        if similarity > max_sim:
             max_sim = similarity
    return max_sim


def group_to_group_similarity(Sa, Su, is_content_s=True):
    """
    :param Sa: First group
    :param Su: Second group to be compared with.
    :param is_content_s: (Boolean) If true, content distance funciton is used to calculate the similarity between two images.
    :return sum_sim: Sum of similarity. This shows how much Sa is similar to Su or represents Su.
    """
    sum_sim = 0
    for su in Su:
        sum_sim += group_to_image_similarity(Sa, su, is_content_s=is_content_s)
    return sum_sim

def get_min_shape(x):
    """
    Out of all the unlabelled samples find the minimum dim on all channels (1,2,3).
    :param x: 5-dimensional array (last layer of the encoder), (# unlabelled samples, # classes, channel-1, channel-2, channel-3)
    :return min_shape: The minimum shape of all the samples
    """
    min_shape = x[0].shape[2:]
    for i in range(len(x)):
        if x[i].shape[2] < min_shape[0]:
            min_shape[0] = x[i].shape[2]
        if x[i].shape[3] < min_shape[1]:
            min_shape[1] = x[i].shape[3]
        if x[i].shape[4] < min_shape[2]:
            min_shape[2] = x[i].shape[4]
    print(min_shape)
    return min_shape

def crop_S(x, min_shape):
    """
    Crop x so that each item's size is equal to min_shape.
    :param x: 5-dimensional array (last layer of the encoder), (# unlabelled samples, # classes, channel-1, channel-2, channel-3)
    :param min_shape: The minimum shape of all the samples
    :return y: Cropped version of x.
    """
    y = [0] * len(x)
    for i in range(len(x)):
        y[i] = x[i][:,:,:min_shape[0],:min_shape[1],:min_shape[2]]
    return y

def mean_S(x):
    """
    Take the mean of the x channels [2:5].
    :param x: 5-dimensional array (last layer of the encoder), (# unlabelled samples, # classes, channel-1, channel-2, channel-3)
    :return z: shape of (# unlabelled samples, # classes)
    """
    z = []
    for i in range(len(x)):
        z.append(np.sum(x[i],(2,3,4))/x[i][0,0].size)
    return z
