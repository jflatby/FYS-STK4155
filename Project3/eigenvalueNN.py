import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use("ggplot")

tf.set_random_seed(3)
np.random.seed(3)

def f(x, A):
    I = tf.eye(int(A.shape[0]), dtype=tf.float64)
    
    xT = tf.transpose(x)
    value1 = tf.matmul(xT, x)*A
    value2 = (1 - tf.matmul(tf.matmul(xT, A), x))*I
    
    return tf.matmul((value1 + value2), x)

def get_eigval(v, A):
    v = v.reshape(A.shape[0], 1)
    val1 = np.matmul(np.matmul(np.transpose(v), A), v)[0, 0]
    val2 = np.matmul(np.transpose(v), v)[0, 0]
    return val1/val2

def NN(A, v0, t_max, dt, n, precision, learning_rate, hidden_neurons, k, max_iter=100000):
    time_points = int(t_max/dt)
    Nx = n
    t = np.linspace(0, (time_points-1)*dt, time_points)
    x = np.linspace(1, Nx, Nx)
    
    X, T = np.meshgrid(x, t)
    V, T_ = np.meshgrid(v0, t)

    x_ = (X.ravel()).reshape(-1, 1)
    t_ = (T.ravel()).reshape(-1, 1)
    v0_ = (V.ravel()).reshape(-1, 1)

    x_tf = tf.convert_to_tensor(x_, dtype=tf.float64)
    t_tf = tf.convert_to_tensor(t_, dtype=tf.float64)
    v0_tf = tf.convert_to_tensor(v0_, dtype=tf.float64)
    
    input_layer = tf.concat([x_tf, t_tf], 1)
    
    with tf.name_scope('dnn'):
        previous_layer = input_layer
        
        for l in range(len(hidden_neurons)):
            current_layer = tf.layers.dense(previous_layer, hidden_neurons[l], activation=tf.nn.sigmoid)
            previous_layer = current_layer
            
        dnn_output = tf.layers.dense(previous_layer, 1)
        
    with tf.name_scope('loss'):
        trial_func = dnn_output*t_tf + v0_tf*k
        
        trial_dt = tf.gradients(trial_func, t_tf)
        
        trial_rs = tf.reshape(trial_func, (time_points, Nx))
        trial_rs_dt = tf.reshape(trial_dt, (time_points, Nx))
        loss_tmp = 0
        
        for i in range(time_points):
            trial_temp = tf.reshape(trial_rs[i], (n, 1))
            trial_dt_temp = tf.reshape(trial_rs_dt[i], (n, 1))
            rhs = f(trial_temp, A) - trial_temp
            err = tf.square(-trial_dt_temp + rhs)
            loss_tmp += tf.reduce_sum(err)
    

        loss = tf.reduce_sum(loss_tmp/(n*time_points), name='loss')

    # Define how the neural network should be trained
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        traning_op = optimizer.minimize(loss)

    # Define a node that initializes all the nodes within the computational graph
    # for TensorFlow to evaluate
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialize the computational graph
        init.run()
        
        print('Initial loss: %g'%loss.eval())
        
        for i in range(max_iter):
            #print(loss.eval())
            if(i%1000 == 0):
                print(f"step {i} --- loss: {loss.eval()}")
            sess.run(traning_op)
            
            if loss.eval() < precision:
                print(f"Last iteration: {i}")
                break

        print('Final loss: %g'%loss.eval())
        
        eigenvectors = tf.reshape(trial_func, (time_points, n)).eval()
        #eigenvalues = dnn_output.eval()
        
    return eigenvectors, t 

#np_eigenvalues = np.linalg.eig(A_matrix)
#print(eigenvalues)
#print(np_eigenvalues)
#diff_tf = np.abs(g_analytic - g_dnn_tf)
#print('Max absolute difference between the analytical solution and solution from TensorFlow DNN: %g'%np.max(diff_tf))

if __name__ == "__main__":
    
    k = -1 # +1
    
    n = 6
    Q = np.random.rand(n,n)
    A = (Q.T+Q)/2
    A_temp = k*A
    A_tf = tf.convert_to_tensor(A_temp,dtype=tf.float64)
    
    precision = 0.00001
    t_max = 5
    dt = 0.1
    learning_rate = 0.001
    v0 = np.random.rand(n)
    hidden_neurons = [10, 10, 10]
    print(A)
    
    v, t = NN(A_tf, v0, t_max, dt, n, precision, learning_rate, hidden_neurons, k)
    
    eigval = get_eigval(v[-1], A)
    print(v[-1]/np.linalg.norm(v[-1]))
    print(eigval)
    print("--")
    print(np.linalg.eig(A))
    
    print(np.matmul(A, v[-1])/eigval)
    print(v[-1])
    
    plt.plot(t, v)
    plt.title("100 000 iterations(max_t = 5)")
    plt.legend(["v1", "v2", "v3", "v4", "v5", "v6"])
    plt.show()
    
    