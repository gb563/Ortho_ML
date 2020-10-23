# -*- coding: utf-8 -*-
"""
Last modified 08 OCT 2020

Greg Booth, M.D.
Naval Biotechnology Group
Naval Medical Center Portsmouth
Portsmouth, VA 23323
in collaboration with:
    Jacob Cole, M.D.
    Scott Hughey, M.D.
    Phil Geiger, M.D.
    Ashton Goldman, M.D.
    George Balazs, M.D.
"""

from Ortho_Github_v1 import *

#libraries you will need: pyreadstat, xgboost, bayesian-optimization, numpy, pandas, shap


#load NSQIP data from disk. You must first request the NSQIP data from 
#https://www.facs.org/quality-programs/acs-nsqip/participant-use and then
#modify the file Ortho_Github_v1 with your file path for the NSQIP SPSS files
#just pass the cpt code you want to load, and voila!
inputs_NSQIP,inputs_cal,inputs_test,targets_train,targets_cal,targets_test = process_ortho_data(27130) 

#generate demographic data
#to generate excel file, go to function and uncomment save line and update file path
demographics(inputs_NSQIP,inputs_test)

#now let's have some fun with bayesian optimization
#inputs are in format (x_train,y_train,random_points=20,search_number=100)
#random_points is number of times to randomly search hyperparameter space
#search_number is how many iterations to perform. 
#something like 20, 100 is reasonable
#you can go into this function to change the hyperparameter bounds for your search
#or change the behavior of the bayesian optimization (alpha and kappa)
optimized_params = optimized_data(inputs_NSQIP,targets_train)

#get mean and confidence intervals for model internal validity
#this will run 1000 times to get 95% confidence interval
#this takes a long time because it retrains at each iteration (set up to run 1000 times)
bootstrap_internal(inputs_NSQIP,targets_train,optimized_params)

#get mean and confidence intervals for model predictions on test data
#this will run 1000 times to get 95% confidence interval
bootstrap_test(inputs_NSQIP,targets_train,inputs_test,targets_test,optimized_params)

#generate calibration plots with isotonic and sigmoid calibration
calibrate_model(inputs_NSQIP,targets_train,inputs_cal,targets_cal,inputs_test,targets_test,optimized_params)

#generate clinical metrics like sensitivity, ppv, specificity
#also gives predictions from simulation of 1000 patients undergoing the procedure
clinical_impact(inputs_NSQIP,targets_train,inputs_test,targets_test,optimized_params)

#generate SHAP plots (model summary, force plots for sensitivity, and
#dependency plots to show how bmi and age and their interactions impact predictions)
shap_plots(inputs_NSQIP, inputs_test, targets_train, targets_test,optimized_params)
