import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

tf.set_random_seed(800313234234234)

dx = 0.1
dt = 0.1

Nx = int(1/dx)
Nt = int(1/dt)

x = np.linspace(0, 1, Nx)
t = np.linspace(0, 1, Nt)

X, T = np.meshgrid(x, t)

x_ = X.ravel().reshape(-1, 1)
t_ = T.ravel().reshape(-1, 1)

x_tf = tf.convert_to_tensor(x_)
t_tf = tf.convert_to_tensor(t_)

points = tf.concat([x_tf, t_tf], 1)

num_iterations = 100000

n_h_neurons = [50]
n_h_layers = np.size(num_iterations)

with tf.name_scope('dnn'):
    previous_layer = points
    
    for l in range(n_h_layers):
        current_layer = tf.layers.dense(previous_layer, n_h_neurons[l], activation=tf.nn.sigmoid)
        previous_layer = current_layer
        
    dnn_output = tf.layers.dense(previous_layer, 1)
    
def u(x):
    return tf.sin(np.pi*x)
    
with tf.name_scope('cost'):
    g_trial = (1-t_tf)*u(x_tf) + x_tf*(1-x_tf)*t_tf*dnn_output

    g_trial_dt = tf.gradients(g_trial,t_tf)
    g_trial_d2x = tf.gradients(tf.gradients(g_trial,x_tf),x_tf)

    err = tf.square(g_trial_dt[0] - g_trial_d2x[0])
    #cost = tf.reduce_sum(err, name = 'cost')
    cost = tf.reduce_mean(err, name='cost')

# Define how the neural network should be trained
learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    traning_op = optimizer.minimize(cost)

# Define a variable to reference to the output from the network
g_dnn_tf = None

# Define a node that initializes all the nodes within the computational graph
# for TensorFlow to evaluate
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialize the computational graph
    init.run()
    
    print('Initial cost: %g'%cost.eval())
    
    for i in range(num_iterations):
        sess.run(traning_op)

    print('Final cost: %g'%cost.eval())
    
    g_dnn_tf = g_trial.eval()

g_analytic = np.exp(-np.pi**2*t_)*np.sin(np.pi*x_)

diff_tf = np.abs(g_analytic - g_dnn_tf)
print('Max absolute difference between the analytical solution and solution from TensorFlow DNN: %g'%np.max(diff_tf))

## Make a surface plot of the solutions

G_dnn_tf = g_dnn_tf.reshape((Nt,Nx)).T
G_analytical = g_analytic.reshape((Nt,Nx)).T

diff_mat = np.abs(G_analytical - G_dnn_tf)

T,X = np.meshgrid(t, x)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/ %d layer'%len(n_h_neurons))
s = ax.plot_surface(T,X,G_dnn_tf,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(T,X,G_analytical,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(T,X,diff_mat,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

## Take some 3D slices

indx1 = 0
indx2 = int(Nt/2)
indx3 = Nt-1

t1 = t[indx1]
t2 = t[indx2]
t3 = t[indx3]

# Slice the results from the DNN
res1 = G_dnn_tf[:,indx1]
res2 = G_dnn_tf[:,indx2]
res3 = G_dnn_tf[:,indx3]

# Slice the analytical results
res_analytical1 = G_analytical[:,indx1]
res_analytical2 = G_analytical[:,indx2]
res_analytical3 = G_analytical[:,indx3]

# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t1)
plt.plot(x, res1)
plt.plot(x,res_analytical1)
plt.legend(['dnn, tensorflow','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t2)
plt.plot(x, res2)
plt.plot(x,res_analytical2)
plt.legend(['dnn, tensorflow','analytical'])

plt.figure(figsize=(10,10))
plt.title("Computed solutions at time = %g"%t3)
plt.plot(x, res3)
plt.plot(x,res_analytical3)
plt.legend(['dnn, tensorflow','analytical'])

plt.show()