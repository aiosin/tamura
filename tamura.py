import numpy as np


def checkarray(arr):
    #TODO: implement
    pass



def coarseness(arr):
    #parameter k affects "kernel size" used in computing averages etc.
    k = 5
    #check if array is legal ie two or three dimensions, non comple
    #TODO: implement three dimensional tamura features
    #also this assertion line could be improved
    assert len(arr.shape) == 2

    #if possible cast to int
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
            #TODO: check correctness and nessecity of index shift (1,k+1)
            for m in range(1,k+1):
                tmp_arr = pad_arr[pad_size+i-(2**(m-1)):pad_size+i+(2**(m-1)-1),
                                pad_size+j-(2**(m-1)):pad_size+j+(2**(m-1)-1)]
                acc_sum = np.sum(tmp_arr)
                # reminder that 2^m * 2^m = 2^(2m)
                # adjust index since our range is 1 k+1 and 
                tmp[m-1] = acc_sum / 2**(2*m)
            l_avg_arr[i][j] = tmp
    
    #pad avg arr
    pad_l_avg_arr = np.pad(l_avg_arr,((pad_size,pad_size,),(pad_size,pad_size),(0,0)),'constant')
    #l_avg_arr contains now the averages 
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

def tamura_3D(arr):
    pass


def directionality(arr):
    pass

def linelikeness(arr):
    pass

def contrast(arr):
    pass

def main():
    x = 99
    y = 123
    z = x*y 
    print(coarseness(np.arange(z).reshape((x,y))))
if __name__ == '__main__':
    main()
