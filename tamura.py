import numpy as np


def checkarray(arr):
    #TODO: implement
    pass



def coarseness(arr):
    k = 5
    #check if array is legal ie two or three dimensions, non comple
    #TODO: implement three dimensional tamura features
    #also this assertion line could be improved
    #assert checkarray(arr)[0] == True
    #if possible cast to int
    src_arr = np.array(arr,dtype='int')

    #initialize padding size
    pad_size = 2**(k-1)
    
    #allocate padded source array and destination array for local avg calculations
    #note that the value doubled here is not the padding size but the tuple
    pad_arr = np.pad(src_arr,(2*(pad_size,) ),'constant')
    l_avg_arr = np.zeros((src_arr.shape)+(k,),dtype='int')
    pad_l_avg_arr = np.pad(l_avg_arr,(2*(pad_size,)),'constant')
    #step 1: calculate the averages over the neighborhood of size 2^k x 2^k
    rows,cols = src_arr.shape
    for i in range(0,rows):
        for j in range(0,cols):
            tmp = np.zeros((k,),dtype='int')
            #TODO: check correctness of index shift (1,k+1)
            for m in range(1,k+1):
                tmp_arr = pad_arr[pad_size+i-(2**(m-1)):pad_size+i+(2**(m-1)-1),
                                pad_size+j-(2**(m-1)):pad_size+j+(2**(m-1)-1)]
                acc_sum = np.sum(tmp_arr)
                # reminder that 2^m * 2^m = 2^(2m)
                tmp[m] = acc_sum / 2**(2*m)
            l_avg_arr[i][j] = tmp

    #l_avg_arr contains now the averages 
    E_h = np.zeros((src_arr.shape)+(k,),dtype='int')
    E_v = np.zeros((src_arr.shape)+(k,),dtype='int')

    #step 2:
    for i in range(0,rows):
        for j in range(0,cols):
            #index shift: 
            for m in range(1,k+1):
                E_h[i,j,m] = abs(pad_l_avg_arr[i,j+(2**(m-1)),m] -
                                pad_l_avg_arr[i,j-(2**(m-1)),m])
                E_v[i,j,m] = abs(pad_l_avg_arr[i+(2**(m-1)),j,m] -
                                pad_l_avg_arr[i-(2**(m-1)),j,m])
    s_arr = np.zeros((src_arr.shape),dtype='int')
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

def tamura_3D(arr):
    pass


def main():
    coarseness(np.array(np.arange(20).reshape((4,5))))

if __name__ == '__main__':
    main()
