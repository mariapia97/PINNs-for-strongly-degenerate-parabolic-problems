#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Mariapia De Rosa
"""

#%% Import libraries
import os
import random 
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from fastcore.all import dict2obj
from keras.layers import Input, Dense
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Set tensoflow type
DTYPE = tf.float32

# Set seed
SEED = 69
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def PINN_2D(problem_dict, f,
                initial_condition, boundary_condition,
                analytical_solution=None):

    # Hyperparameters dictionary configuration
    config = dict(
        lr = 3e-3,
        ker_reg = 1e-6,
        epochs = 30000,
        patience = 10000,
        opt = 'Adam',  
        input_dims = problem_dict['input_dims'],
        save = problem_dict['save'],
        t_initial = problem_dict['t_initial'], 
        t_final = problem_dict['t_final'],
        x_initial = problem_dict['x_initial'],
        x_final = problem_dict['x_final'],
        y_initial = problem_dict['y_initial'],
        y_final = problem_dict['y_final'],
        Nt = problem_dict['Nt'],
        Nx = problem_dict['Nx'],
        Ny = problem_dict['Ny'],
        print_epoch = 200,
        plot_epoch = 5000,
        config_name = problem_dict['config_name']
    )

    cf = dict2obj(config)

    # Create figures folder
    os.makedirs(f'Figures/{cf.config_name}/', exist_ok = True)

    # input data
    t = np.linspace(cf.t_initial, cf.t_final, cf.Nt)
    x = np.linspace(cf.x_initial, cf.x_final, cf.Nx)
    y = np.linspace(cf.y_initial, cf.y_final, cf.Ny)

    dataset_train = tf.constant(np.array(
                    list(itertools.product(
                        t, x, y))), dtype=DTYPE)
    
    # Reshape for plotting
    X = dataset_train[...,1].numpy().reshape((cf.Nt, cf.Nx, cf.Ny))
    Y = dataset_train[...,2].numpy().reshape((cf.Nt, cf.Nx, cf.Ny))

    # Create the model
    model = model_pde(cf)

    # Training process
    _, model = train(dataset_train, model, cf, f,
                initial_condition, boundary_condition)

    # Save the model
    if cf.save:
        model.save(f'Models/{cf.config_name}/model')    

    if analytical_solution is not None:
        # Compute the analytical solution
        sol = analytical_solution(dataset_train[...,0], 
                                    dataset_train[...,1], 
                                    dataset_train[...,2])

        S = sol.numpy().reshape((cf.Nt, cf.Nx, cf.Ny))

        # Plot the analytical solution
        i = int(cf.Nt/2)    #select the index you need 
                            #for time t
        sol_i = S[i,:,:]
        plot_slice_2D(cf, i, sol_i, X, Y)

    # Predict the solution
    with tf.device('/cpu:0'):
        preds = model.predict(dataset_train)

        u = preds[...,0]

    U = u.reshape((cf.Nt, cf.Nx, cf.Ny))

    # Plot the predicted solution
    i = int(cf.Nt/2)    #select the index you need 
                        #for time t
    sol_i = U[i,:,:]
    plot_slice_2D(cf, i, sol_i, X, Y, Analytic = False)


def model_pde(cf):
    if cf.ker_reg is not None:
        ker_reg = tf.keras.regularizers.L2(cf.ker_reg)
    else:
        ker_reg = None

    # Input layer 
    inputs = Input(shape=(cf.input_dims))

    # Hidden layers

    x = Dense(20,activation='tanh', name='hidden1', 
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(inputs)
    
    x = Dense(20,activation='tanh', name='hidden2', 
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)  

    x = Dense(20,activation='tanh', name='hidden3',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)

    x = Dense(20,activation='tanh', name='hidden4',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)

    # Output layer 

    outputs = Dense(1, activation='linear', name='output',
                    kernel_regularizer=ker_reg)(x)

    # Create the model

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model    



def loss_2D(inputs, model, cf, f,
            initial_condition, boundary_condition):
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)        
        preds = model(inputs)

    grads = tape.gradient(preds, inputs)

    del tape

    Du = tf.math.sqrt(grads[...,1]**2 + grads[...,2]**2)   
    PositivePart = tf.nn.relu(Du - 1.)  
    if np.all(PositivePart != 0):
        norm = grads[:,:,0]/(tf.constant(Du.numpy().reshape(-1,1))) 
        prod = tf.constant(PositivePart.numpy().reshape(-1,1))*norm
    
        # compute divergence of the vector field using GradientTape
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs[:,:,0])

        div = tape2.gradient(prod, inputs[:,:,0])

        del tape2

        div = div[:,0] + div[:,1]
    else:
        
        div = 0              

    ic = initial_condition(inputs, preds, cf)
    bc = boundary_condition(inputs, preds)    
    residual = f(grads, div, inputs)


    reg = tf.reduce_sum(model.losses)
    loss =  ic + bc + residual + reg

    return {'loss': loss, 
            'f': residual,
            'ic': ic,
            'bc': bc, 
            'reg': reg
            }


# Optimizers
def select_optimizator(cf):
    if cf.opt == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=cf.lr)
    elif cf.opt == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=cf.lr)
    elif cf.opt == 'Adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=cf.lr)
    return opt


def obtain_loss_dict():
        return {'epoch':[],
                'loss':[],
                'f':[],    
                'ic':[],           
                'bc':[],
                'reg':[]
                }

# Train Step
def train_step(inputs, model, cf, opt, f,
                initial_condition, boundary_condition):

    with tf.GradientTape(persistent=True) as loss_tape: 
        if cf.input_dims == 3:       
            loss_dict = loss_2D(inputs, model, cf, f,
                                initial_condition, boundary_condition)
    gradients_of_model = loss_tape.gradient(loss_dict['loss'],  
                                            model.trainable_variables)    
    opt.apply_gradients(zip(gradients_of_model, 
                            model.trainable_variables))
    del loss_tape
    return loss_dict


def train(inputs, model, cf, f,
                initial_condition, boundary_condition):
    import time
    opt = select_optimizator(cf)    
    t0 = time.time()
    best_loss = tf.constant(np.inf, dtype=DTYPE)
    patience = cf.patience
    wait = 0

    history = obtain_loss_dict()
    for epoch in range(1, cf.epochs + 1):     
        start_epoch = time.time()  
        
        loss_dict = train_step(inputs, model, cf, opt, f,
                initial_condition, boundary_condition)
        loss_value = loss_dict['loss']
        history['epoch'].append(epoch)
                
        for key, elem in loss_dict.items():
            history[key].append(elem)
        
        if loss_value<best_loss:
            mega_str = f'Best {epoch}: '             
            
            for key, elem in loss_dict.items():
                mega_str += f'{key} = {elem.numpy():10.6e}  '            
            
            mega_str += f'Time = {time.time()-start_epoch:.2f} sec'
            print('\r'+mega_str, end= '')           

            best_loss = loss_value
            wait = 0
            best_weights= model.get_weights() 
            best_dict = loss_dict.copy()    

        elif wait>=patience:
            print('Stop the train phase')
            break
        else:
            wait +=1
            if epoch % cf.print_epoch == 0:   
                if wait > cf.print_epoch: epoch_string = f'Epoch {epoch} '
                else: epoch_string = f'\nEpoch {epoch} '

                for key, elem in loss_dict.items():
                    if key == 'epoch': continue
                    epoch_string += f'{key} = {elem.numpy():10.6e}  '               

                epoch_string += f'Time = {time.time()-start_epoch:.2f} sec'
                print('\r'+epoch_string)

        if epoch % cf.plot_epoch == 0: 
            plt.figure(figsize=(10,10))
            plt.plot(history['loss'][-cf.plot_epoch:], label='loss')
            plt.show()
            if cf.save:
                os.makedirs(f'Models/{cf.config_name}/', exist_ok=True)
                model.save(f'Models/{cf.config_name}/')

    if wait !=0:
        model.set_weights(best_weights)
    
    if cf.save:
        os.makedirs(f'Models/{cf.config_name}/', exist_ok=True)
        model.save(f'Models/{cf.config_name}/')

    print('\nComputation time: {} seconds'.format(time.time()-t0))
    
    return best_dict, model

def plot_slice_2D(cf, i, Sol_i, X, Y, Analytic = True):    

    X_i = X[i,:,:]
    Y_i = Y[i,:,:]
    A = 'Analytic' if Analytic else 'Predicted'

    fig = plt.figure(figsize=(30, 10))
    # subplot 1 row, 2 columns, with surface and countour
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X_i, Y_i, Sol_i, rstride=1, cstride=1,
                        cmap='RdBu',linewidth=0, antialiased=False)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(elev=25, azim=-90)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Solution ' + str(A))
    plt.xlabel('x')
    plt.ylabel('y')

    ax = fig.add_subplot(1, 2, 2)
    plt.contour(X_i, Y_i, Sol_i)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig(f'Figures/{cf.config_name}/2D_{i}_{A}.png', facecolor = 'w', dpi = 300)
    plt.show()
