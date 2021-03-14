Machine Learning in Orthopedics

Source code for using NSQIP data to generate XGBoost models to predict discharge destination following Total Knee Arthroplasty (TKA) and Total Hip Arthroplasty (THA)

This code was used to generate results in the manuscript "Machine Learning to Predict Discharge Destination after Total Knee Arthroplasty and Total Hip Arthroplasty".

libraries you will need: pyreadstat, xgboost, bayesian-optimization, numpy, pandas, shap


load NSQIP data from disk. You must first request the NSQIP data from https://www.facs.org/quality-programs/acs-nsqip/participant-use and then modify the file Ortho_discharge_github with your file path for the NSQIP SPSS files

Functions:

process_ortho_data(cpt_code) # loads all data with specific CPT from NSQIP data and generates random stratified samples for train, calibration, and test data

demographics(inputs_NSQIP,inputs_test_final) #demographic summary data (i.e. Table 1)

optimized_data(inputs_NSQIP,targets_train,random_points=20,search_number=100) #returns dictionary with XGBoost parameters tuned with Bayesian Optimization

calibration_split(inputs_test,targets_test) #separates data for calibration and validation

create_model(train,targets,test,optimized_params) #fits model with optimized hyperparameters and returns probabilities on test data

bootstrap_internal(inputs_NSQIP,targets_train,optimized_params) #generates boostrapped estimates of mean and 95% CI for AUC and brier for model performance using train data

bootstrap_test(inputs_NSQIP,targets_train,inputs_test_final,targets_test_final,optimized_params,cal_iso_model) #generates bootstrapped estimates for mean and 95% CI for AUC and brier score on test data

calibrate_model(inputs_NSQIP,targets_train,inputs_cal,targets_cal,inputs_test,targets_test,optimized_params) #performs isotonic calibration and generates calibration plots

clinical_impact(x_train,y_train,x_test,targets_test,optimized_params,cal_iso_model) #generates decision curves after finding sensitivity and specificity at various cutoffs

shap_plots(inputs_NSQIP, inputs_test_final, targets_train, targets_test_final,optimized_params,OrthoV1) #generates SHAP summary, force, and dependency plots

