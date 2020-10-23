Machine Learning in Orthopedics

Source code for using NSQIP data to generate XGBoost models to predict discharge destination following Total Knee Arthroplasty (TKA) and Total Hip Arthroplasty (THA)

This code was used to generate results in the manuscript "Machine Learning to Predict Discharge Destination after Total Knee Arthroplasty and Total Hip Arthroplasty", submitted to the Journal of Arthoplasty Oct 2020

libraries you will need: pyreadstat, xgboost, bayesian-optimization, numpy, pandas, shap


load NSQIP data from disk. You must first request the NSQIP data from https://www.facs.org/quality-programs/acs-nsqip/participant-use and then modify the file Ortho_Github_v2 with your file path for the NSQIP SPSS files

Functions:

process_ortho_data(cpt_code) # loads all data with specific CPT from NSQIP data and generates random stratified samples for train, calibration, and test data

optimized_data(inputs_NSQIP,targets_train,random_points=20,search_number=100) #returns dictionary with XGBoost parameters tuned with Bayesian Optimization

bootstrap_internal(inputs_NSQIP,targets_train,optimized_params) #generates boostrapped estimates of mean and 95% CI for model performance using train data

bootstrap_test(inputs_NSQIP,targets_train,inputs_test,targets_test,optimized_params) #generates boostrapped estimates of mean and 95% CI for model performance using test data

calibrate_model(inputs_NSQIP,targets_train,inputs_cal,targets_cal,inputs_test,targets_test,optimized_params) #performs sigmoid and isotonic calibration and generates calibration plots

clinical_impact(inputs_NSQIP,targets_train,inputs_test,targets_test,optimized_params) #generates things like sensitivity, specificity, and other clinical metrics using trained models

shap_plots(inputs_NSQIP, inputs_test, targets_train, targets_test,optimized_params) #generates SHAP summary, force, and dependency plots

