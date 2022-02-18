import numpy as np 
import matplotlib.pyplot as plt

# cost function
def c(s, a, d , h, q):
    """
    cost function as defined above
    s: current state
    a: current action
    d: Per-unit ordering cost
    h: holding cost constant
    q: backlogged orders cost constant
    """
    assert a >=0
    return d*a + h * max(s, 0) + q * max(-s, 0)

# Terminal Cost function
def C(s, h = 0, q = 0):
    """
    Termianl cost function as defined above
    s: current state
    h: holding cost constant
    q: backlogged orders cost constant
    """

    return  h * max(s, 0) + q * max(-s, 0)

def state_pdf(s, a, S, W):
    s_probs = np.zeros((31))
    W_max = W[-1]
    idx_s = int(np.where(S == s)[0]) + a
    idx_min = idx_s - W_max
    if idx_min<0:
        zeros_count = -idx_min 
        idx_min = 0
    else:
        zeros_count = 0
    s_probs[idx_min:idx_s+1] = 1/len(W)
    s_probs[0] += 1/len(W)*zeros_count
    return s_probs

# implement algorithm here (you can use different function signature)
def get_value_function_BR(t, S, T, d , h, q):
    W_max = 10
    ### YOUR CODE GOES HERE
    V = []
    V += [np.array([C(s_i, h, q) for s_i in S])]
    for i in range(T-t):
    #chose a to maximize
    #check all actions
    #state_pdf(-12,3,S,W)
        cur_vals = []
        for s_i in S:
            min_val = 1e6
            for a in A:
                val = c(s_i, a, d, h, q) + state_pdf(s_i, a, S, W_max)@V[-1]
                if val < min_val:
                    min_val = val
            cur_vals.append(min_val)
        V.append(np.array(cur_vals))
    return V[-1], V

# Define state range of interest
S =  np.arange(-15,16)
A =  np.arange(0,21)
#V = get_value_function_BR(0, S, 10, 1 , 4, 2)
#a = 10
d = 1
h = 4
q = 2
T = 10
t = 0
V = []
V += [np.array([C(s_i, h, q) for s_i in S])]
for i in range(T-t):
    #chose a to maximize
    #check all actions
#state_pdf(-12,3,S,W)
    cur_vals = []
    for s_i in S:
        min_val = 1e6
        for a in A:
            val = c(s_i, a, d, h, q) + state_pdf(s_i, a, S, W)@V[-1]
            if val < min_val:
                min_val = val
        cur_vals.append(min_val)
    V.append(np.array(cur_vals))
    #      if c(s_i,h,q,a) + E(V[-1]s_i+a-W
print('here')



def state_pdf(s, a, S, W_max):
    s_probs = np.zeros(S.shape)
    idx_s = int(np.where(S == s)[0]) + a
    idx_min = idx_s - W_max
    if idx_min<0:
        zeros_count = -idx_min 
        idx_min = 0
    else:
        zeros_count = 0
    s_probs[idx_min:idx_s+1] = 1/(W_max+1)
    s_probs[0] += 1/(W_max+1)*zeros_count
    return s_probs
    
def get_value_function_BR(t, S, T, d , h, q):
    W_max = 10
    A = np.arange(0,21)
    ### YOUR CODE GOES HERE
    V = []
    V += [np.array([C(s_i, h, q) for s_i in S])]
    for i in range(T-t):
    #chose a to maximize
    #check all actions
    #state_pdf(-12,3,S,W)
        cur_vals = []
        for s_i in S:
            min_val = 1e6
            for a in A:
                if(a+s_i <= S[-1]):
                  val = c(s_i, a, d, h, q) + state_pdf(s_i, a, S, W_max)@V[-1]
                if val < min_val:
                    min_val = val
            cur_vals.append(min_val)
        V.append(np.array(cur_vals))
    return V[-1], V

# Define state range of interest
S =  np.arange(-15,16)
V, V_it = get_value_function_BR(0, S, 10, 1 , 4, 2)