#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Mariapia De Rosa
"""
#%% Import libraries
import numpy as np
import tensorflow as tf

from problem_setting import *

#%% Select test number
number_of_test = 1 #choose 1,2,3

if number_of_test == 1:
    # First test problem
    problem_dict = {
                    'input_dims': 3,
                    'save': False,
                    't_initial': 0.,
                    't_final': 10.,
                    'x_initial': -1.,
                    'x_final': 1.,
                    'y_initial': -1.,
                    'y_final': 1.,
                    'Nt': 21,
                    'Nx': 21,
                    'Ny': 21,
                    'config_name': 'test_1',
    }

    def analytical_solution(t, x, y):
        return (1/2)*(x**2 + y**2) + t

    def initial_condition(inputs, preds, cf):
        t = tf.expand_dims(inputs[...,0], axis=1)
        x = tf.expand_dims(inputs[...,1], axis=1)
        y = tf.expand_dims(inputs[...,2], axis=1)

        cond = tf.equal(t, cf.t_initial)
        ic = tf.where(cond, (1/2)*(x**2 + y**2), preds)
        error = tf.reduce_mean((ic - preds)**2)
        return error

    def boundary_condition(inputs, preds):
        t = tf.expand_dims(inputs[...,0], axis=1)
        x = tf.expand_dims(inputs[...,1], axis=1)    
        y = tf.expand_dims(inputs[...,2], axis=1) 

        cond = tf.equal(x**2 + y**2, 1.)
        bc = tf.where(cond, t + 1/2, preds)
        error = tf.reduce_mean((bc - preds)**2)
        return  error

    def f(grads, div, inputs):
        return tf.reduce_mean((grads[...,0] - div - 1.)**2)


    PINN_2D(problem_dict, f,
                    initial_condition, 
                    boundary_condition,
                    analytical_solution)

elif number_of_test == 2:
        
    # Second test problem
    problem_dict = {
                    'input_dims': 3,
                    'save': False,
                    't_initial': 0.,
                    't_final': 1.,
                    'x_initial': -1.,
                    'x_final': 1.,
                    'y_initial': -1.,
                    'y_final': 1.,
                    'Nt': 21,
                    'Nx': 61,
                    'Ny': 61,
                    'config_name': 'test_2',
    }

    alpha = 1/2

    def analytical_solution(t, x, y):
        return t*(x**2 + y**2)**alpha

    def initial_condition(inputs, preds, cf):
        t = tf.expand_dims(inputs[...,0], axis=1)

        cond = tf.equal(t, cf.t_initial)
        ic = tf.where(cond, 0., preds)
        error = tf.reduce_mean((ic - preds)**2)
        return error

    def boundary_condition(inputs, preds):
        t = tf.expand_dims(inputs[...,0], axis=1)
        x = tf.expand_dims(inputs[...,1], axis=1)    
        y = tf.expand_dims(inputs[...,2], axis=1) 

        cond = tf.equal((x**2 + y**2), 1.)
        bc = tf.where(cond, t, preds)
        error = tf.reduce_mean((bc - preds)**2)

        return error

    def f(grads, div, inputs):
        if (np.any(2*(alpha)*inputs[...,0]*(inputs[...,1]**2 + 
                    inputs[...,2]**2)**(alpha-(1/2)) <= 1) & np.any(inputs[...,1]**2 + inputs[...,2]**2 <= 1.)):
            f = tf.reduce_mean((grads[...,0] - div - 
                                (inputs[...,1]**2 + inputs[...,2]**2)**(alpha))**2)
        else:
            f = tf.reduce_mean((grads[...,0] - div - 
                            (inputs[...,1]**2 + inputs[...,2]**2)**(alpha) + 
                            4.*((alpha)**2)*inputs[...,0]*((inputs[...,1]**2 
                            + inputs[...,2]**2)**(alpha-1)) - 
                            1./(tf.math.sqrt(inputs[...,1]**2 + inputs[...,2]**2)))**2)
        return f

    PINN_2D(problem_dict, f,
                    initial_condition, 
                    boundary_condition,
                    analytical_solution)

elif number_of_test == 3:
    # Third test problem
    problem_dict = {
                    'input_dims': 3,
                    'save': False,
                    't_initial': 0.,
                    't_final': 50.,
                    'x_initial': -1.,
                    'x_final': 1.,
                    'y_initial': -1.,
                    'y_final': 1.,
                    'Nt': 105,
                    'Nx': 71,
                    'Ny': 71,
                    'config_name': 'test_3',
    }

    def analytical_solution(t, x, y):
        cond = tf.logical_and(tf.logical_and(tf.less_equal(x, 0.), tf.greater_equal(x, -1.)),
                            tf.logical_and(tf.less_equal(y, 1.), tf.greater_equal(y, -1.))) 
        sol = tf.where(cond, 1. + t, 1. -x + t)
        return sol


    def initial_condition(inputs, preds, cf):
        t = tf.expand_dims(inputs[...,0], axis=1)
        x = tf.expand_dims(inputs[...,1], axis=1)
        y = tf.expand_dims(inputs[...,2], axis=1)

        cond = tf.logical_and(tf.equal(t, cf.t_initial), 
                                tf.logical_and(tf.less_equal(x, 1.), tf.greater(x, 0.)), 
                                tf.logical_and(tf.less_equal(y, 1.), tf.greater_equal(y, -1.)))
        ic = tf.where(cond, 1.- x, preds)
        error = tf.reduce_mean((ic - preds)**2)

        cond2 = tf.logical_and(tf.equal(t, cf.t_initial), 
                            tf.logical_and(tf.less_equal(x, 0.), 
                                            tf.greater_equal(x, -1.)),
                            tf.logical_and(tf.less_equal(y, 1.), 
                                            tf.greater_equal(y, -1.)))
        ic2 = tf.where(cond2, 1. , preds)
        error2 = tf.reduce_mean((ic2 - preds)**2)
        return error + error2

    def boundary_condition(inputs, preds):
        t = tf.expand_dims(inputs[...,0], axis=1)
        x = tf.expand_dims(inputs[...,1], axis=1)    
        y = tf.expand_dims(inputs[...,2], axis=1) 

        cond = tf.logical_and(tf.less_equal(x, 0.), tf.greater_equal(x, -1.), tf.equal(y, -1.))
        bc = tf.where(cond, 1. + t, preds)
        error = tf.reduce_mean((bc - preds)**2)

        cond2 = tf.logical_and(tf.less_equal(x, 0.), tf.greater_equal(x, -1.), tf.equal(y, 1.))
        bc2 = tf.where(cond2, 1. + t, preds)
        error2 = tf.reduce_mean((bc2 - preds)**2)

        cond3 = tf.logical_and(tf.less_equal(y, 1.), tf.greater_equal(y, -1.), tf.equal(x, -1.))
        bc3 = tf.where(cond3, 1. + t, preds)
        error3 = tf.reduce_mean((bc3 - preds)**2)

        cond4 = tf.logical_and(tf.less_equal(y, 1.), tf.greater_equal(y, -1.), tf.equal(x, 1.))
        bc4 = tf.where(cond4, t, preds)
        error4 = tf.reduce_mean((bc4 - preds)**2)

        cond5 = tf.logical_and(tf.less_equal(x, 1.), tf.greater_equal(x, 0.), tf.equal(y, -1.))
        bc5 = tf.where(cond5, 1. + t - x, preds)
        error5 = tf.reduce_mean((bc5 - preds)**2)

        cond6 = tf.logical_and(tf.less_equal(x, 1.), tf.greater_equal(x, 0.), tf.equal(y, 1.))
        bc6 = tf.where(cond6, 1. + t - x, preds)
        error6 = tf.reduce_mean((bc6 - preds)**2)

        return  error + error2 + error3 + error4 + error5 + error6

    def f(grads, div, inputs):
        return tf.reduce_mean((grads[...,0] - div - 1.)**2)

    PINN_2D(problem_dict, f,
                    initial_condition, 
                    boundary_condition,
                    analytical_solution)
    
else:
    print("The test number doesn't exist")