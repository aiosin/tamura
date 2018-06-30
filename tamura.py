"""
Module: tamura.py
Author: Wilhelm Buchmueller

Tested with python3.6 only

This module implements the most important textural image features discovered by Tamura et al.
The features should be computed only on greyscale images, but this module allows you to pass
in three dimensional(i.e. volumetric, not primarily RGB) images, because we're extending 
the Tamura features according to Majtner et al (WIP).

This module was developed with formal correctness in mind, 
self-documenting code, as well as testability.
Performance may be worse than existing python/MATLAB solutions, although numpy
features were used wherever possible.

This module is released under the MIT License. If you're using this and it works out 
for you please leave me a star at:
    https://github.com/aiosin/tamura
    
"""

import numpy as np
from scipy.stats import kurtosis,moment
from scipy.signal import convolve2d


def checkarray( arr,logger=None) -> bool:
    """
    checkarray asserts wether the given input array is truly compatible for use with
    the functions in this module

    Args:
        arr - greyscale two dimensional array of type `numpy.ndarray`
    Returns:
        `True` if array compatible
        'False' if not
    """
    if 'float' in str(arr.dtype):
        if logger is not None:
            print('INFO: ndarray of type float was supplied to function ')

    if len(arr.shape) != 2:
        return False

    if 0 < arr.min() or 0 > arr.max():
        return False
    
    if 255 < arr.max() or 255 < arr.min():
        return False

    return True

localvar = 0

def coarseness(arr,n_param=5) -> float:
    '''
    Compute coarseness feature of an two dimensional greyscale image.
    WARNING: Supply an integer array or it will be casted for you
    Args:
        arr - greyscale two dimensional array of type numpy.ndarray with dtype ='int'
        n_param - beighborhood parameter for calculating the max neighborhood averages
    
    Returns:
        res - result of coarseness computation of type float 
    '''
    #parameter k affects "kernel size" used in computing averages etc.
    #Tamura recommends leavint this at k = 5
    k = n_param
    #TODO: check formal correctness

    assert checkarray(arr) is True

    #if possible cast to int, wouldnt want to deal with floats
    src_arr = np.array(arr,dtype='int')

    #initialize padding size
    pad_size = 2**(k-1)
    
    #allocate padded source array and destination array for local avg calculations
    #note that the value doubled here is not the padding size but the tuple
    pad_arr = np.pad(src_arr,(2*(pad_size,) ),'constant')
    l_avg_arr = np.zeros((src_arr.shape)+(k,),dtype='int')
    #step 1: calculate the averages over the neighborhood of size 2^k x 2^k
    rows,cols = src_arr.shape
    for i in range(0,rows):
        for j in range(0,cols):
            tmp = np.zeros((k,),dtype='int')
            for m in range(1,k+1):
                tmp_arr = pad_arr[pad_size+i-(2**(m-1)):pad_size+i+(2**(m-1)-1),
                                pad_size+j-(2**(m-1)):pad_size+j+(2**(m-1)-1)]
                acc_sum = np.sum(tmp_arr)
                # reminder that 2^m * 2^m = 2^(2m)
                # adjust index since our range is 1 k+1
                tmp[m-1] = acc_sum / 2**(2*m)
            l_avg_arr[i][j] = tmp
    
    #l_avg_arr contains now the averages 
    #TODO: figure out why I padded this array
    pad_l_avg_arr = np.pad(l_avg_arr,((pad_size,pad_size,),(pad_size,pad_size),(0,0)),'constant')
    E_h = np.ones((src_arr.shape)+(k,),dtype='float')
    E_v = np.ones((src_arr.shape)+(k,),dtype='float')   


    #step 2:
    for i in range(0,rows):
        for j in range(0,cols):
            #index shift: 
            for m in range(1,k+1):
                tmp_1 = abs(pad_l_avg_arr[i+pad_size,pad_size+j+(2**(m-1)),m-1] -
                            pad_l_avg_arr[i+pad_size,pad_size+j-(2**(m-1)),m-1])
                E_h[i,j,m-1] = tmp_1
                tmp_2 = abs(pad_l_avg_arr[i+(2**(m-1))+pad_size,j+pad_size,m-1] -
                            pad_l_avg_arr[i-(2**(m-1))+pad_size,j+pad_size,m-1])
                E_v[i,j,m-1] = tmp_2
    s_arr = np.zeros((src_arr.shape),dtype='float')
    #step 3:
    for i in range(0,rows):
        for j in range(0,cols):
            s_best = None
            h_max_v = np.max(E_h[i,j])
            v_max_v = np.max(E_v[i,j])
            h_max_i = np.argmax(E_h[i,j])
            v_max_i = np.argmax(E_v[i,j])

            if h_max_v < v_max_v:
                s_best = 2**v_max_i
            else:
                s_best = 2**h_max_i
                
            s_arr[i,j] = s_best
    #step 4:
    avg = np.mean(s_arr)
    return avg


def coarseness_3D(arr)->float:
    '''
    extension of coarseness feature into three dimensions

    Args:
        arr - greyscale image with three dimensions
    Returns:
        res - result of coarseness computation in three dimensions of type float
    '''
    assert len(arr.shape) == 3
    #TODO implement according to Majtner et. al.
    return 0.


def directionality(arr,n=16, t=12)->float:
    '''
    calculate the directionality according to Tamura et al.

    Args:
        arr     2D greyscale image

        n       optional parameter for directionality
                preset to 16
                Tamura et al. recommend setting this value to 
                Change at your own risk.
        t       optional parameter for directionality
                preset to 12
                Tamura et al. recommend setting this value to 
                Change at your own risk.
    Returns:
        res - result of coarseness computation in three dimensions of type float
    '''

    assert checkarray(arr) == True

    arr = np.array(arr, dtype='int')

    h_op = np.array([[-1,0,-1],[-1,0,-1],[-1, 0, 1]])
    v_op = np.array([[ 1,1, 1],[ 0,0, 0],[-1,-1,-1]])
    '''vertical and horizontal operators $\Delta_H, \Delta_V$'''

    pad_d_h = convolve2d(np.pad(arr,(1,1),'constant'),h_op)
    pad_d_v = convolve2d(np.pad(arr,(1,1),'constant'),v_op)
    
    p_rows, p_cols = pad_d_v.shape

    d_h = np.abs(pad_d_h[1:p_rows-1,1:p_cols-1])
    d_v = np.abs(pad_d_v[1:p_rows-1,1:p_cols-1])
    #not using magnitude to threshhold array for now
    #IMPORTANT enable/calculate it for formal correctness
    d_G = (( d_h +d_v )/2).flatten()
    theta = np.arctan((d_v/d_h) + np.pi/2).flatten()

    #IMPORTANT were not using d_G for thresholding since it is clear from the paper 
    #what the purpose of t is and to quote the authors:
    # "HD was not sensitive to the value of t."
    #I'm assuming were good to go by ignoring t thresholding
    #should put "theta > t" in np.logical_and 

    #see "Improving the Functionality of Tamura Directionality on Solar Images" by Ahmadzadeh et al.
    #
    H_D = np.zeros((n*2,))
    for k in range(2*n):
        lower_bound = k * np.pi /(2*n)
        upper_bound = (k+1) * np.pi / (2*n)
        n_theta = np.count_nonzero(np.logical_and(theta > lower_bound, theta < upper_bound))
        H_D[k] = n_theta
        
    #sum all the occurences
    w_H_D = np.sum(H_D)
    #normalizing histogram with said sum
    H_D /= w_H_D

    #peak detection
    peaks = list()
    #rudimentary approach 
    pv1,ip1 = 0,0
    pv2,ip2 = 0,0
    for i in range(len(H_D)):
        if H_D[i] > pv1:
            ip1 = i
            pv1 = H_D[i]
    peaks.append([ip1,pv1])
    #reverse index to find other peak
    for i in range(len(H_D))[::-1]:
        if H_D[i] > pv1:
            ip1 = i
            pv1 = H_D[i]
    peaks.append([ip2,pv2])
    peaks = np.array(peaks)

    #texture unidirectional if only 1 peak
    if ip1 == ip2:
        peaks = peaks[0:1]

    #REMINDER case > 3 peaks not handled
    #IDEA use thresholding to find correct number of peaks
    #same stupid approach for valleys as peaks
    valleys = list()
    #rudimentary approach 
    vv1,iv1 = 0,0
    vv2,iv2 = 0,0
    for i in range(len(H_D)):
        if H_D[i] < vv1:
            iv1 = i
            vv1 = H_D[i]
    peaks.append([iv1,vv1])
    #reverse index to find other peak
    for i in range(len(H_D))[::-1]:
        if H_D[i] < vv2:
            iv1 = i
            vv1 = H_D[i]
    valleys.append([iv2,vv2])
    valleys = np.array(valleys )

    #texture unidirectional if only 1 valley
    if iv1 == iv2:
        valleys = valleys[0:1]
    #test valleys and peaks according to formula
    #TODO
    F_d = 0

    #the approach which we adopted  is to sum the second moments
    #around each peak from valley to valley
    #however no more than two peaks are detected
    #WARNING: 
    if(len(peaks) == 2):
        F_d += moment(H_D[0:iv1] ,  2)
        F_d += moment(H_D[iv1:] ,  2)

    #normalizing factor r:
    pi1_2 = np.pi/2.0
    r=1.0 / (pi1_2**2)

    # F_dir = 1 - r * F_d
    fdir = 1-r*F_d
    return fdir


def directionality_3D(arr):
    pass

def linelikeness(arr):
    pass

def linelikeness_3D(arr):
    '''
    calculate contrast according to Tamura et al.

    Args:
        arr     2D greyscale integer array

        n       (optional) parameter for "weighing" the kurtosis
                0.25 has been experimentally determined to be
                the best performing parameter for this

                Warning: 
                Change the this parameter if you know what you're doing.
                You may end up withuseless results by changing this parameter.

    Returns:
        res     Computed contrast of the image of type flaot
    '''
    assert checkarray(arr) is True
    arr = np.array(arr, dtype='int')
    weight = 0.
    #for now use skimages implementation
    #TODO: remove skimage dependency to cut down on dependencies
    from skimage.feature import greycomatrix
    # gcm = greycomatrix(arr)
    weight = gcm


    res = float(0)
    return res


def contrast(arr,n=0.25)-> float:
    '''
    calculate contrast according to Tamura et al.
    WARNING: Supply a integer array or the given array will be casted
    Args:
        arr     2D greyscale integer array

        n       (optional) parameter for "weighing" the kurtosis
                0.25 has been experimentally determined to be
                the best performing parameter for this

                Warning: 
                Change the this parameter if you know what you're doing.
                You may end up withuseless results by changing this parameter.

    Returns:
        res     Computed contrast of the image of type flaot
    '''
    assert checkarray(arr) is True
    arr = np.array(arr, dtype='int')
    
    kurt = kurtosis(arr)
    fcon = np.std(arr) / (kurt**n)
    return fcon


def contrast_3D(arr,n)->float:
    '''
    calculate contrast in 3D according to Tamura et al.

    Args:
        arr     3D greyscale integer array

        n       (optional) parameter for "weighing" the kurtosis
                0.25 has been experimentally determined to be
                the best performing parameter for this

                Warning: 
                Change the this parameter if you know what you're doing.
                You may end up with useless results by changing this parameter.

    Returns:
        res     Computed contrast of the image of type float
    '''
    assert len(arr.shape) == 3
    arr = np.array(arr,dtype='int')
    kurt = kurtosis(arr)
    fcon = np.std(arr) / (kurt**n)
    return fcon

def roughness(arr, n=0.25)->float:
        '''
    calculate roughness according to Tamura et al.

    Args:
        arr     2D greyscale integer array

        n       (optional) parameter for "weighing" the kurtosis
                0.25 has been experimentally determined to be
                the best performing parameter for this

                Warning: 
                Change the this parameter if you know what you're doing.
                You may end up with useless results by changing this parameter.

    Returns:
        res     Computed roughness of the image of type float
    '''
    assert checkarray(arr) == True
    return coarseness(arr)+ contrast(arr,n)

def roughness3d (arr, n=0.25) ->float:
    return coarseness_3D(arr) + contrast_3D(arr, n)


def regularity()-> float:
    """
    regularity
    NOT IMPLEMENTED
    """
    return 0.
#
if __name__ == '__main__':
    x = 99
    y = 123
    z = x*y 
    print(coarseness(np.arange(z).reshape((x,y))))
    print(coarseness(np.ones((100,100))))