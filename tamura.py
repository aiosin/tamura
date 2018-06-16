import numpy as np

def checkarray(arr):
    #TODO: implement
    pass



def coarseness(arr):
    k = 5
    #check if array is legal ie two or three dimensions, non comple
    #TODO: implement three dimensional tamura features
    #also this assertion line could be improved
    assert checkarray(arr)[0] == True
    #if possible cast to int
    src_arr = np.array(arr,dtype='int')

    #initialize padding size
    pad_size = 2**(k-1)
    
    #allocate padded source array and destination array for local avg calculations
    pad_arr = np.pad(src_arr,(2*(pad_size,) ),'constant')
    l_avg_arr = np.zeros((src_arr.shape)+(5,),dtype='int')

    #step 1: calculate the averages over the neighborhood of size 2^k x 2^k
    rows,cols = src_arr.shape
    for i in range(0,rows):
        for j in range(0,cols):
            tmp = np.zeros((k,),dtype='int')
            for m in range(0,k):
                tmp_arr = pad_arr[pad_size+i-(2**(k-1)):pad_size+i+(2**(k-1)-1),pad_size+j-(2**(k-1)):pad_size+j+(2**(k-1)-1)]
                acc_sum = np.sum(tmp_arr)
                # reminder that 2^k * 2^k = 2^(2k)
                tmp[m] = acc_sum / 2**(2*k)
            l_avg_arr[i][j] = tmp

    #l_avg_arr contains now the averages 


    #step 2:

def tamura_3D(arr):
    pass