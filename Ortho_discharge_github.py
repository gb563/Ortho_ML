# -*- coding: utf-8 -*-
"""
Created on Wed FEB 17 20:30:57 2021
Last modified 13 MAR 2021

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

from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from numpy import array
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import shap



def process_ortho_data(cpt_code,Train = True):
    
    #define which NSQIP variables you want to extract so you don't load unnecessary data
    cols3 = ['CPT','OTHERCPT1','ELECTSURG','AGE','SEX','FNSTATUS2','WNDINF',
             'EMERGNCY','PRSEPIS','DIABETES','DYSPNEA','ASACLAS','STEROID',
             'ASCITES','VENTILAT','DISCANCR','HYPERMED','HXCHF','SMOKE',
             'HXCOPD','DIALYSIS','RENAFAIL','HEIGHT','WEIGHT','DISCHDEST']
    
    #load all the datasets from disk (need to request NSQIP datasets first)
    if Train:
        df15 = pd.read_spss(r'C:', usecols=cols3)
        df15.rename(columns={"AGE": "Age"},inplace=True)
        df16 = pd.read_spss(r'C:', usecols=cols3)
        df16.rename(columns={"AGE": "Age"},inplace=True)
        df17 = pd.read_spss(r'C:', usecols=cols3)
        df17.rename(columns={"AGE": "Age"},inplace=True)
        df18 = pd.read_spss(r'C:', usecols=cols3)
        df18.rename(columns={"AGE": "Age"},inplace=True)
        #combine each year into one large dataframe
        data2=pd.concat([df15,df16,df17,df18],axis=0)
    else:
        df19 = pd.read_spss(r'C:', usecols=cols3)
        df19.rename(columns={"AGE": "Age"},inplace=True)
        data2 = df19.copy()
    
    data2 = shuffle(data2)
    data2 = data2.reset_index(drop=True)
    #exclusions
    #first load cpt of interest. NSQIP mixes strings, floats, and integers
    data2=data2[(data2['CPT']==cpt_code) | (data2['CPT']==float(cpt_code))|
                (data2['CPT']==str(cpt_code))|(data2['CPT']==str(float(cpt_code)))]
    print('Total cpt {:d} ='.format(cpt_code),data2.shape[0])
    #drop any cases that had secondary CPTs
    data2=data2[data2['OTHERCPT1']=='NULL']
    print('After excluding secondary CPT, Total = {:d}'.format(data2.shape[0]))
    #exclude non-elective (fractures)
    data2=data2[data2['ELECTSURG']=='Yes']
    print('After excluding non-elective cases, Total = {:d}'.format(data2.shape[0]))
    #exclude unknown discharge dest
    data2 = data2.drop(data2[(data2['DISCHDEST'] == 'NULL')| 
                             (data2['DISCHDEST'] == 'Unknown')|
                             (data2['DISCHDEST'] == 'Expired')|
                             (data2['DISCHDEST'] == 'Against Medical Advice (AMA)')].index)
    print('After excluding unknown discharge location, Total = {:d}'.format(data2.shape[0]))
    #drop ASA 5
    data2=data2.drop(data2[data2['ASACLAS']=='5-Moribund'].index)
    print('After excluding ASA 5, Total = {:d}'.format(data2.shape[0]))
    #drop sepsis or septic shock
    data2=data2.drop(data2[(data2['PRSEPIS']=='Sepsis')|
                           (data2['PRSEPIS']=='Septic Shock')|
                           (data2['PRSEPIS']=='Septic')].index)
    #drop wound infection
    data2=data2.drop(data2[(data2['WNDINF']=='Yes')].index)
    print('After excluding sepsis or wound infections, Total = {:d}'.format(data2.shape[0]))

    #we will drop rows with missing data later after processing the various names 
    #used for missing data (e.g. 'NUL','NULL','Unknown',etc)
    
    #define targets - assign 0 to Home or Facility which was home, 1 to everything else
    dest_pos = ['Rehab','Separate Acute Care','Unskilled Facility Not Home',
                'Skilled Care, Not Home','Unskilled Facility Not','Hospice',
                'Multi-level Senior Community']
    dest_neg = ['Home','Facility Which was Home']
    data2['DISCHDEST']=data2['DISCHDEST'].replace(to_replace=dest_neg,value='0')
    data2['DISCHDEST']=data2['DISCHDEST'].replace(to_replace=dest_pos,value='1')
    data2['DISCHDEST']=data2['DISCHDEST'].astype(int)
    targets_data = data2['DISCHDEST']
    targets_data=array(targets_data)
    targets_data=targets_data.reshape(-1,1)
    
    #now process all the inputs and handle missing data
    #process BMI
    BMI=[]
    weights1=data2['WEIGHT'].to_numpy()
    heights1=data2['HEIGHT'].to_numpy()
    for i in range(len(data2)):
        if (weights1[i]!=-99) and (heights1[i]!=-99): 
            #convert height and weight to BMI if both are known
            BMI.append((703*weights1[i])/((heights1[i])**2))
        else: 
            BMI.append(-99)
            
    for i in range(len(BMI)):
        if BMI[i]>=70:
            BMI[i]=70
        if BMI[i] < 15 and BMI[i]>0:
            BMI[i]=15
        if (BMI[i]==-99):
            BMI[i]=np.nan 
    
    BMI=array(BMI).reshape(-1,1)
    
    #process age
    data2['Age'] = data2['Age'] .astype(str).replace('\.0', '', regex=True)
    x00=data2['Age']
    x0=x00.copy()
    for i in range(len(x00)):
        if x00.iloc[i]=='90+':
            x0.iloc[i]='90'
        elif x00.iloc[i]=='-99':
            x0.iloc[i]='nan'
    
    x0=x0.replace({'nan':'10'})
    x0=x0.astype(float)
    x0=x0.replace({10:np.nan})
    x0=x0.to_numpy().reshape(-1,1)
    
    
    x1=data2['SEX']
    x1=x1.replace({'NULL':np.nan,'non-bi':np.nan,'male':0,'female':1})
    x1=x1.to_numpy().reshape(-1,1)
    
    x2 = data2['FNSTATUS2']
    x2=x2.replace({'Independent':0,'Partially Dependent':1,'Partially D':1,
                   'Totally Dep':2,'Totally Dependent':2,'Unknown':np.nan})
    x2=x2.to_numpy().reshape(-1,1)
    
    x4=data2['ASACLAS']
    x4=x4.replace({'NULL':np.nan,'Null':np.nan,'None assigned':np.nan,
                   '1-No Disturb':1,'2-Mild Disturb':2,'3-Severe Disturb':3,
                   '4-Life Threat':4})
    x4=x4.to_numpy().reshape(-1,1)
    
    x5=data2['STEROID']
    x5=x5.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1})
    x5=x5.to_numpy().reshape(-1,1)
    
    x6=data2['ASCITES']
    x6=x6.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1,'Ye':1})
    x6=x6.to_numpy().reshape(-1,1)
    
    x77 = data2['PRSEPIS']
    x77=x77.replace({'NULL':np.nan,'None':0,'SIRS':1})
    x7=x77.to_numpy().reshape(-1,1)
    
    x8=data2['VENTILAT']
    x8=x8.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1})
    x8=x8.to_numpy().reshape(-1,1)
    
    x9=data2['DISCANCR']
    x9=x9.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1})
    x9=x9.to_numpy().reshape(-1,1)
    
    x101 = data2['DIABETES']
    x101=x101.replace({'NULL':np.nan,'NO':0,'ORAL':1,
                      'NON-INSULIN':1,'INSULIN':1})
    x10=x101.to_numpy().reshape(-1,1)
    
    x11=data2['HYPERMED']
    x11=x11.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1})
    x11=x11.to_numpy().reshape(-1,1)
    
    x13=data2['HXCHF']
    x13=x13.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1,'Ye':1})
    x13=x13.to_numpy().reshape(-1,1)
    
    x14= data2['DYSPNEA']
    x14=x14.replace({'NULL':np.nan,'No':0,'MODERATE EXERTION':1,'AT REST':1})
    x14=x14.to_numpy().reshape(-1,1)
    
    x15=data2['SMOKE']
    x15=x15.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1})
    x15=x15.to_numpy().reshape(-1,1)
    
    x16=data2['HXCOPD']
    x16=x16.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1})
    x16=x16.to_numpy().reshape(-1,1)
    
    x17=data2['DIALYSIS']
    x17=x17.replace({'NULL':np.nan,'NUL':np.nan,'No':0,'Yes':1,'Ye':1})
    x17=x17.to_numpy().reshape(-1,1)
    
    x18=data2['RENAFAIL']
    x18=x18.replace({'NULL':np.nan,'NU':np.nan,'No':0,'Yes':1,'Ye':1})
    x18=x18.to_numpy().reshape(-1,1)
    
    x19 = BMI.reshape(-1,1)
    
    #put all inputs together into one array
    inputs_aggregate = np.concatenate([x0,x1,x2,x4,x5,x6,x7,x8,x9,x10,x11,x13,
                                       x14,x15,x16,x17,x18,x19],axis=1)

    #drop nans
    data3 = inputs_aggregate.copy()
    data4 = targets_data.copy()
    data5 = np.concatenate([data3,data4],axis=1)
    data5=data5[~np.isnan(data5).any(axis=1)]
    print('final size of data for CPT {:d} ='.format(cpt_code),data5.shape)
    
    inputs_aggregate = data5[:,-19:-1]
    targets_data = data5[:,-1].reshape(-1,1)

    #make inputs for training (70% of data) inputs_NSQIP and targets_train
    #then split up the holdout data for calibration and testing
    #inputs_NSQIP, inputs_holdout, targets_train, targets_holdout = train_test_split(inputs_aggregate, targets_data, test_size=0.3, random_state=444,stratify=targets_data)
    #inputs_cal, inputs_test, targets_cal, targets_test= train_test_split(inputs_holdout, targets_holdout, test_size=0.67, random_state=444,stratify=targets_holdout)
    #print('train data = ',inputs_NSQIP.shape[0], '\ncal data = ',inputs_cal.shape[0], '\ntest data = ',inputs_test.shape[0])
    
    return inputs_aggregate, targets_data



#now for the fun part
#feel free to adjust the hyperparameters within the function
#this will return optimal hyperparameters after bayesian optimization
#This optimization section is modified from some open source code originally appearing here: https://ayguno.github.io/curious/portfolio/bayesian_optimization.html
def optimized_data(inputs_NSQIP,targets_train,rand_points=40,search_number=40):

    def xgboost_bayesian(max_depth,learning_rate,colsample_bytree, min_child_weight,reg_alpha,gamma):
        
        optimizer = xgb.XGBClassifier(max_depth=int(max_depth),
                                               learning_rate= learning_rate,
                                               n_estimators= 200,
                                               reg_alpha = reg_alpha,
                                               gamma = gamma,
                                               nthread = -1,
                                               colsample_bytree = colsample_bytree,
                                               min_child_weight = min_child_weight,
                                               objective='binary:logistic',
                                               seed = 444,
                                               scale_pos_weight = 1)
        roc_auc_holder=[]
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=444)
        for train_index, test_index in rskf.split(inputs_NSQIP, targets_train):
            x_train, x_test = inputs_NSQIP[train_index],inputs_NSQIP[test_index]
            y_train, y_test = targets_train[train_index], targets_train[test_index]
            
            optimizer.fit(x_train,y_train.ravel(),eval_set =  [(x_test,y_test.ravel())], eval_metric = 'logloss',early_stopping_rounds = 10)
            probs = optimizer.predict_proba(x_test)
            probs = probs[:,1]
            roc1 = roc_auc_score(y_test,probs)
            roc_auc_holder.append(roc1) 

        return sum(roc_auc_holder)/len(roc_auc_holder)
    
    hyperparameters = {
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'reg_alpha': (0, 0.5),
        'gamma': (0, 0.5),
        'min_child_weight': (5,30),
        'colsample_bytree': (0.1, 1)
    }
    
    bayesian_object = BayesianOptimization(f = xgboost_bayesian, 
                                 pbounds =  hyperparameters,
                                 verbose = 2)
    
    bayesian_object.maximize(init_points=rand_points,n_iter=search_number,
                             acq='ucb', kappa= 2, alpha = 1e-7)
    
    #now we have optimal parameters
    OrthoV1 = xgb.XGBClassifier(max_depth=int(bayesian_object.max['params']['max_depth']),
                                           learning_rate= bayesian_object.max['params']['learning_rate'],
                                           n_estimators= 200,
                                           reg_alpha = bayesian_object.max['params']['reg_alpha'],
                                           gamma = bayesian_object.max['params']['gamma'],
                                           nthread = -1,
                                           colsample_bytree = bayesian_object.max['params']['colsample_bytree'],
                                           min_child_weight = bayesian_object.max['params']['min_child_weight'],
                                           objective='binary:logistic',
                                           seed = 444,
                                           scale_pos_weight = 1)

    #now refit all data an determine optimal n_estimators via cross-val
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=444)
    best_it_hold = []
    for train_index, test_index in rskf.split(inputs_NSQIP, targets_train):
        x_train, x_test = inputs_NSQIP[train_index],inputs_NSQIP[test_index]
        y_train, y_test = targets_train[train_index], targets_train[test_index]
        OrthoV1.fit(x_train,y_train.ravel(), eval_set=[(x_test,y_test.ravel())],eval_metric = 'logloss',early_stopping_rounds=10)
        best_it_hold.append(OrthoV1.best_iteration)
        
    best_training_iteration = int(round(sum(best_it_hold)/len(best_it_hold)))

    optimized_params =            {'max_depthV1':int(round(bayesian_object.max['params']['max_depth'])),
                                   'colsample_bytreeV1':bayesian_object.max['params']['colsample_bytree'],
                                   'gammaV1':bayesian_object.max['params']['gamma'],
                                   'learning_rateV1': bayesian_object.max['params']['learning_rate'],
                                   'min_child_weightV1':bayesian_object.max['params']['min_child_weight'],
                                   'reg_alphaV1':bayesian_object.max['params']['reg_alpha'],
                                   'best_training_iteration':best_training_iteration,
                                   'roc':bayesian_object.max['target']}
    return optimized_params



def calibration_split(inputs_test,targets_test):
    inputs_cal, inputs_test_final, targets_cal, targets_test_final = train_test_split(inputs_test, targets_test, test_size=0.5, random_state=444,stratify=targets_test)
    return inputs_cal, inputs_test_final, targets_cal, targets_test_final
    

#now we need to fit the final model (above was just estimating performance)
#and then perform calibration
def calibrate_model(inputs_cal, inputs_test_final, targets_cal, targets_test_final,OrthoV1):

    cal_iso = CalibratedClassifierCV(OrthoV1, method='isotonic', cv='prefit')
    cal_iso.fit(inputs_cal, targets_cal.ravel())
    
    #make some predictions!
    probs = OrthoV1.predict_proba(inputs_test_final)
    probs = probs[:,1]
    probs2 = cal_iso.predict_proba(inputs_test_final)
    probs2=probs2[:,1]
    
    #brier for all three
    b_loss = brier_score_loss(targets_test_final,probs)
    b_loss_iso = brier_score_loss(targets_test_final,probs2)

    #calibration curves for all three
    fop, mpv = calibration_curve(targets_test_final,probs,n_bins = 10,strategy='uniform')
    fop2, mpv2 = calibration_curve(targets_test_final,probs2,n_bins = 10,strategy='uniform')

    #Calibration curves
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.plot(mpv,fop,'b',label='No Cal, Brier = {:.3f}'.format(b_loss))
    ax.plot(mpv2,fop2,'r:',label='Isotonic, Brier = {:.3f}'.format(b_loss_iso))
    ax.plot([0,0.9],[0,0.9],'k--',label='Perfect Calibration')
    ax.legend(loc = 'lower right')
    ax.plot(mpv,fop,'bs')
    ax.plot(mpv2,fop2,'ro')
    ax.set_title('Calibration Curves for Validation Data, Hip')
    ax.set_xlabel('Mean Predicted Value')
    ax.set_ylabel('Fraction of Positives')
    plt.show()
    
    return cal_iso


def create_model(train,targets,test,optimized_params):
    OrthoV1 = xgb.XGBClassifier(max_depth = optimized_params['max_depthV1'],
                                learning_rate= optimized_params['learning_rateV1'],
                                n_estimators= optimized_params['best_training_iteration'],
                                reg_alpha = optimized_params['reg_alphaV1'],
                                gamma = optimized_params['gammaV1'],
                                nthread = -1,
                                colsample_bytree = optimized_params['colsample_bytreeV1'],
                                min_child_weight = optimized_params['min_child_weightV1'],
                                objective='binary:logistic',
                                seed = 444,
                                scale_pos_weight = 1)
    OrthoV1.fit(train,targets.ravel())
    model_probs=OrthoV1.predict_proba(test)
    model_probs=model_probs[:,1]
    return model_probs


def bootstrap_internal(inputs_NSQIP,targets_train,optimized_params):
    #bootstrap code for internal validity (Fitting model each time)

    roc_hold=[]
    brier_hold=[]
    index_holder=range(0,len(inputs_NSQIP))
    j=0
    x_train=[]
    y_train=[]
    for i in range(500):
        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        boot = np.random.choice(index_holder,size=(len(index_holder)),replace=True)
        test_index = [x for x in index_holder if x not in boot]
        for k in range(len(boot)):
            x_train.append(inputs_NSQIP[boot[k]])
            y_train.append(targets_train[boot[k]])
            
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        #define test data (data not in bootstrap)
        for k in range(len(test_index)):
            x_test.append(inputs_NSQIP[test_index[k]])
            y_test.append(targets_train[test_index[k]])
        
        x_test=np.array(x_test)
        y_test=np.array(y_test)
        preds=create_model(x_train,y_train,x_test,optimized_params)
        auc_roc = roc_auc_score(y_test,preds)
        ##Brier score
        b_loss = brier_score_loss(y_test, preds)
        #print('brier score ='+str(b_loss))
        print('be patient, iteration',j,' ROC = ',auc_roc)
        j=j+1
        roc_hold.append(auc_roc)
        brier_hold.append(b_loss)
            
    roc_hold=array(sorted(roc_hold))
    brier_hold=array(sorted(brier_hold))
    
    av_brier = sum(brier_hold)/len(brier_hold)
    av_roc = sum(roc_hold)/len(roc_hold)
    print('ROC AUC = ',av_roc,', 95% CI = ',(roc_hold[11]+roc_hold[12])/2,'to ',(roc_hold[486]+roc_hold[487])/2)
    print('Brier Score = ',av_brier,', 95% CI = ',(brier_hold[11]+brier_hold[12])/2,'to ',(brier_hold[486]+brier_hold[487])/2)
    


def bootstrap_test(inputs_NSQIP,targets_train,inputs_test_final,targets_test_final,optimized_params,cal_iso_model):

    preds= cal_iso_model.predict_proba(inputs_test_final)
    preds=preds[:,1]
    
    roc_hold=[]
    brier_hold=[]
    index_holder=range(0,len(targets_test_final))
    j=0
    y_test=[]
    y_preds=[]
    for i in range(500):
        y_test=[]
        y_preds=[]
        boot = np.random.choice(index_holder,size=(len(index_holder)),replace=True)
        for k in range(len(boot)):
            y_test.append(targets_test_final[boot[k]])
            y_preds.append(preds[boot[k]])
            
        y_test=np.array(y_test)
        y_preds=np.array(y_preds)
    
        auc_roc = roc_auc_score(y_test,y_preds)
        b_loss = brier_score_loss(y_test, y_preds)
        print('be patient, iteration',j)
        j=j+1
        roc_hold.append(auc_roc)
        brier_hold.append(b_loss)
    
    av_brier = sum(brier_hold)/len(brier_hold)
    av_roc = sum(roc_hold)/len(roc_hold)
    roc_hold=sorted(roc_hold)
    brier_hold=sorted(brier_hold)
    print('ROC AUC = ',av_roc,', 95% CI = ',(roc_hold[11]+roc_hold[12])/2,'to ',(roc_hold[486]+roc_hold[487])/2)
    print('Brier Score = ',av_brier,', 95% CI = ',(brier_hold[11]+brier_hold[12])/2,'to ',(brier_hold[486]+brier_hold[487])/2)




def clinical_impact(x_train,y_train,x_test,targets_test,optimized_params,cal_iso_model):
    #clinical impact
    preds=cal_iso_model.predict_proba(x_test)[:,1].reshape(-1,1)
    Thresholds = np.linspace(0.001, 0.6, 100, endpoint=True)
    sens_XGB = []
    spec_XGB = []
    ppv_XGB=[]
    num_tp = []
    num_fn = []
    num_fp = []
    dca = []
    all_treat = []
    no_treat = []
    prevalence = (targets_test==1).sum()/targets_test.shape[0]
    for j in range(len(Thresholds)):
        y_pred_XGB = [1 if i>Thresholds[j] else 0 for i in preds]
        CM_XGB = confusion_matrix(targets_test, y_pred_XGB)
        #sens and ppv
        tp_XGB = CM_XGB[1,1]
        fp_XGB = CM_XGB[0,1]
        fn_XGB = CM_XGB[1,0]
        tn_XGB = CM_XGB[0,0]
        pr_XGB = tp_XGB/[tp_XGB+fp_XGB]
        rec_XGB = tp_XGB/[tp_XGB+fn_XGB]
        spec_XGB_hold = tn_XGB/[tn_XGB+fp_XGB]
        sens_XGB.append(rec_XGB)
        spec_XGB.append(spec_XGB_hold)
        ppv_XGB.append(pr_XGB)
        num_tp.append(tp_XGB)
        num_fn.append(fn_XGB)
        num_fp.append(fp_XGB)
        dca.append((tp_XGB/(preds.shape[0]))-(fp_XGB/(preds.shape[0]))*(Thresholds[j]/(1-Thresholds[j])))
        no_treat.append(0)
        all_treat.append((prevalence)-(1-prevalence)*(Thresholds[j]/(1-Thresholds[j])))
        
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.plot(Thresholds,no_treat,'k',label='No Treatment')
    ax.plot(Thresholds,all_treat,'b-.',label='Treat All')
    ax.plot(Thresholds,dca,'r--',label='Model')
    ax.legend(loc = 'upper right')
    ax.set_title('Decision Curve, Knee Model')
    ax.set_xlabel('Decision Threshold (%)')
    ax.set_ylabel('Net Clinical Benefit')
    plt.xlim([0,0.5])
    plt.ylim([-0.005, .1])
    plt.show()
    

def demographics(inputs_NSQIP,inputs_test_final):

    total_train_inputs = inputs_NSQIP
    place_hold=np.nan
    
    #now get all the means, std, and counts
    age_train_mean = np.nanmean(total_train_inputs[:,0])
    age_train_std = np.nanstd(total_train_inputs[:,0])
    age_test_mean = np.nanmean(inputs_test[:,0])
    age_test_std = np.nanstd(inputs_test[:,0])
    ages=array([age_train_mean,age_train_std,age_test_mean,age_test_std,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    sex_train_male = (total_train_inputs[:,1]==0).sum()
    sex_train_female = (total_train_inputs[:,1]==1).sum()
    sex_test_male = (inputs_test[:,1]==0).sum()
    sex_test_female = (inputs_test[:,1]==1).sum()
    sexes = array([sex_train_male,sex_train_female,sex_test_male,sex_test_female,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    fxn_train_ind = (total_train_inputs[:,2]==0).sum()
    fxn_train_part = (total_train_inputs[:,2]==1).sum()
    fxn_train_tot = (total_train_inputs[:,2]==2).sum()
    fxn_test_ind = (inputs_test[:,2]==0).sum()
    fxn_test_part = (inputs_test[:,2]==1).sum()
    fxn_test_tot = (inputs_test[:,2]==2).sum()
    fxns = array([fxn_train_ind,fxn_train_part,fxn_train_tot,fxn_test_ind,fxn_test_part,fxn_test_tot,place_hold,place_hold]).reshape(-1,1)
    
    asa_train_1 = Counter(total_train_inputs[:,3])[1.0]
    asa_train_2 = Counter(total_train_inputs[:,3])[2.0]
    asa_train_3 = Counter(total_train_inputs[:,3])[3.0]
    asa_train_4 = Counter(total_train_inputs[:,3])[4.0]
    asa_test_1 = Counter(inputs_test[:,3])[1.0]
    asa_test_2 = Counter(inputs_test[:,3])[2.0]
    asa_test_3 = Counter(inputs_test[:,3])[3.0]
    asa_test_4 = Counter(inputs_test[:,3])[4.0]
    asas=array([asa_train_1,asa_train_2,asa_train_3,asa_train_4,asa_test_1,asa_test_2,asa_test_3,asa_test_4]).reshape(-1,1)
    
    steroids_train_yes = Counter(total_train_inputs[:,4])[1.0]
    steroids_train_no = Counter(total_train_inputs[:,4])[0.0]
    steroids_test_yes = Counter(inputs_test[:,4])[1.0]
    steroids_test_no = Counter(inputs_test[:,4])[0.0]
    steroids = array([steroids_train_yes,steroids_train_no,steroids_test_yes,steroids_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    ascites_train_yes = Counter(total_train_inputs[:,5])[1.0]
    ascites_train_no = Counter(total_train_inputs[:,5])[0.0]
    ascites_test_yes = Counter(inputs_test[:,5])[1.0]
    ascites_test_no = Counter(inputs_test[:,5])[0.0]
    ascites=array([ascites_train_yes,ascites_train_no,ascites_test_yes,ascites_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    sirs_train_yes = Counter(total_train_inputs[:,6])[1.0]
    sirs_train_no = Counter(total_train_inputs[:,6])[0.0]
    sirs_test_yes = Counter(inputs_test[:,6])[1.0]
    sirs_test_no = Counter(inputs_test[:,6])[0.0]
    sirs=array([sirs_train_yes,sirs_train_no,sirs_test_yes,sirs_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    vent_train_yes = Counter(total_train_inputs[:,7])[1.0]
    vent_train_no = Counter(total_train_inputs[:,7])[0.0]
    vent_test_yes = Counter(inputs_test[:,7])[1.0]
    vent_test_no = Counter(inputs_test[:,7])[0.0] 
    vents = array([vent_train_yes,vent_train_no,vent_test_yes,vent_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    cancer_train_yes = Counter(total_train_inputs[:,8])[1.0]
    cancer_train_no = Counter(total_train_inputs[:,8])[0.0]
    cancer_test_yes = Counter(inputs_test[:,8])[1.0]
    cancer_test_no = Counter(inputs_test[:,8])[0.0]
    cancers=array([cancer_train_yes,cancer_train_no,cancer_test_yes,cancer_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    diabetes_train_yes = Counter(total_train_inputs[:,9])[1.0]
    diabetes_train_no = Counter(total_train_inputs[:,9])[0.0]
    diabetes_test_yes = Counter(inputs_test[:,9])[1.0]
    diabetes_test_no = Counter(inputs_test[:,9])[0.0]
    diabetes=array([diabetes_train_yes,diabetes_train_no,diabetes_test_yes,diabetes_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    htn_train_yes = Counter(total_train_inputs[:,10])[1.0]
    htn_train_no = Counter(total_train_inputs[:,10])[0.0]
    htn_test_yes = Counter(inputs_test[:,10])[1.0]
    htn_test_no = Counter(inputs_test[:,10])[0.0]
    htn = array([htn_train_yes,htn_train_no,htn_test_yes,htn_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    chf_train_yes = Counter(total_train_inputs[:,11])[1.0]
    chf_train_no = Counter(total_train_inputs[:,11])[0.0]
    chf_test_yes = Counter(inputs_test[:,11])[1.0]
    chf_test_no = Counter(inputs_test[:,11])[0.0]
    chf = array([chf_train_yes,chf_train_no,chf_test_yes,chf_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    dyspnea_train_yes = Counter(total_train_inputs[:,12])[1.0]
    dyspnea_train_no = Counter(total_train_inputs[:,12])[0.0]
    dyspnea_test_yes = Counter(inputs_test[:,12])[1.0]
    dyspnea_test_no = Counter(inputs_test[:,12])[0.0]
    dyspnea = array([dyspnea_train_yes,dyspnea_train_no,dyspnea_test_yes,dyspnea_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    smoker_train_yes = Counter(total_train_inputs[:,13])[1.0]
    smoker_train_no = Counter(total_train_inputs[:,13])[0.0]
    smoker_test_yes = Counter(inputs_test[:,13])[1.0]
    smoker_test_no = Counter(inputs_test[:,13])[0.0]
    smoker = array([smoker_train_yes,smoker_train_no,smoker_test_yes,smoker_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    COPD_train_yes = Counter(total_train_inputs[:,14])[1.0]
    COPD_train_no = Counter(total_train_inputs[:,14])[0.0]
    COPD_test_yes = Counter(inputs_test[:,14])[1.0]
    COPD_test_no = Counter(inputs_test[:,14])[0.0]
    COPD = array([COPD_train_yes,COPD_train_no,COPD_test_yes,COPD_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    dialysis_train_yes = Counter(total_train_inputs[:,15])[1.0]
    dialysis_train_no = Counter(total_train_inputs[:,15])[0.0]
    dialysis_test_yes = Counter(inputs_test[:,15])[1.0]
    dialysis_test_no = Counter(inputs_test[:,15])[0.0]
    dialysis = array([dialysis_train_yes,dialysis_train_no,dialysis_test_yes,dialysis_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    renalfail_train_yes = Counter(total_train_inputs[:,16])[1.0]
    renalfail_train_no = Counter(total_train_inputs[:,16])[0.0]
    renalfail_test_yes = Counter(inputs_test[:,16])[1.0]
    renalfail_test_no = Counter(inputs_test[:,16])[0.0]
    renalfail=array([renalfail_train_yes,renalfail_train_no,renalfail_test_yes,renalfail_test_no,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    bmi_train_mean = np.nanmean(total_train_inputs[:,17])
    bmi_train_std = np.nanstd(total_train_inputs[:,17])
    bmi_test_mean = np.nanmean(inputs_test[:,17])
    bmi_test_std = np.nanstd(inputs_test[:,17])
    bmi=array([bmi_train_mean,bmi_train_std,bmi_test_mean,bmi_test_std,place_hold,place_hold,place_hold,place_hold]).reshape(-1,1)
    
    names_dem = ['Age','Sex','ASA PS','BMI','HTN','Diabetes','COPD','Functional',
                 'Smoker','Dyspnea','Steroids','CHF','Dialysis','Cancer','SIRS',
                 'Renal Failure','Vent','Ascites']
    
    dem_data = np.concatenate([ages,sexes,asas,bmi,htn,diabetes,COPD,fxns,smoker,dyspnea,steroids,chf,dialysis,cancers,sirs,renalfail,vents,ascites],axis=1)
    
    dem_pd=pd.DataFrame(dem_data)
    dem_pd.columns=names_dem
    dem_pd.to_excel(r'C:', index = False)
   
    print(dem_pd)
    

def shap_plots(inputs_NSQIP, inputs_test_final, targets_train, targets_test_final,optimized_params,OrthoV1):
    
    #summary plot
    names2=['Age','Sex','Fxn Status','ASA PS','Steroids','Ascites','SIRS','Ventilator','Cancer','Diabetes','HTN','CHF','Dyspnea','Smoker','COPD','Dialysis','Renal Failure','BMI']
    explainer=shap.TreeExplainer(OrthoV1)
    shap_values=explainer.shap_values(inputs_test_final)
    shap.summary_plot(shap_values,inputs_test_final,feature_names=names2,show=False)
    #shap.summary_plot(shap_values,inputs_test_final,feature_names=names2,plot_type='bar',show=False)
    

    #force plots for sensitivity analysis
    j=0
    k=1
    sens_true_pos_hold =[]
    sens_false_neg_hold = []
    least_conf=[]
    probs=create_model(inputs_NSQIP,targets_train,inputs_test_final,optimized_params)
    preds_hold = np.concatenate([probs.reshape(-1,1),targets_test_final],axis=1)
    preds_hold_sorted=preds_hold.copy()
    preds_hold_sorted=preds_hold_sorted[preds_hold_sorted[:,0].argsort()]
    #first true positives
    while j<5:
        c = preds_hold_sorted[-k,0]
        if preds_hold_sorted[-k,1]==1:
            sens_true_pos_hold.append(np.where(probs==c))
            j=j+1
        k=k+1
    #now false negatives
    j=0
    k=1
    while j<5:
        c = preds_hold_sorted[-k,0]
        if preds_hold_sorted[-k,1]==0:
            sens_false_neg_hold.append(np.where(probs==c))
            j=j+1
        k=k+1
        
    #now find the least confident predictors
    j=0
    k=5;
    least_conf=[]
    while len(least_conf)<5:
        c = preds_hold_sorted[j,0]
        hold1=np.where(probs==c)
        hold1=array(hold1)
        if hold1.shape[1]>1:
            for i in range((hold1.shape[1])):
                least_conf.append(hold1[0,i])
            k=k-hold1.shape[1]
        else:
            least_conf.append(hold1)
        j=j+1
        print("j =",j)
        print('k=',k)
    least_conf=least_conf[0:5]
    least_conf=array(least_conf)
    least_conf=least_conf.astype(int)
        
    #account for any duplicates
    sens_false_neg_hold = np.concatenate(sens_false_neg_hold,axis=1)
    sens_false_neg_hold = sens_false_neg_hold[0,0:5]
    sens_true_pos_hold = np.concatenate(sens_true_pos_hold,axis=1)
    sens_true_pos_hold = sens_true_pos_hold[0,0:5]
    
    sens_true_pos_hold=np.squeeze(sens_true_pos_hold)
    sens_false_neg_hold=np.squeeze(sens_false_neg_hold)
    least_conf=np.squeeze(least_conf)
    ##now we have indices of most confident correct and most confident but incorrect
    sens_true_pos_hold=array(sens_true_pos_hold)
    data_true_pos = inputs_test_final[sens_true_pos_hold]
    
    sens_false_neg_hold=array(sens_false_neg_hold)
    data_false_neg = inputs_test_final[sens_false_neg_hold]
    
    #plot all of the force_plots
    #true positives
    #basic formatting for display purposes only
    inputs_test2=inputs_test_final.copy()
    
    for i in range(len(inputs_test2)):
        inputs_test2[i,-1]=round(inputs_test2[i,-1],1)
    
    for i in range(0,3):
        shap_display=shap.force_plot(explainer.expected_value,shap_values[sens_true_pos_hold[i],:],inputs_test2[sens_true_pos_hold[i],:],matplotlib=True,feature_names=names2,show=False,text_rotation=60)
        print('patient: ',sens_true_pos_hold[i],preds_hold[sens_true_pos_hold[i],0],preds_hold[sens_true_pos_hold[i],1])
        
    #false negatives
    for i in range(0,3):
        shap_display=shap.force_plot(explainer.expected_value,shap_values[sens_false_neg_hold[i],:],inputs_test2[sens_false_neg_hold[i],:],matplotlib=True,feature_names=names2,show=False,text_rotation=60)
        print('patient: ',sens_false_neg_hold[i],preds_hold[sens_false_neg_hold[i],0],preds_hold[sens_false_neg_hold[i],1])
        
    #least confident
    for i in range(0,3):
        shap_display=shap.force_plot(explainer.expected_value,shap_values[least_conf[i],:],inputs_test2[least_conf[i],:],matplotlib=True,feature_names=names2,show=False,text_rotation=60)
        print('patient: ',least_conf[i],preds_hold[least_conf[i],0],preds_hold[least_conf[i],1])
        
    #SHAP dependency plots
    df_inputs_test=pd.DataFrame(inputs_test_final, columns=names2)
    df_knees_test = df_inputs_test.copy()
    #age
    shap.dependence_plot(0,shap_values,df_knees_test,interaction_index=0,show=False)
    #bmi
    shap.dependence_plot(17,shap_values,df_knees_test,interaction_index=17,show=False)
    #age by BMI
    shap.dependence_plot(0,shap_values,df_knees_test,interaction_index=17,show=False)
    #BMI by age
    shap.dependence_plot(17,shap_values,df_knees_test,interaction_index=0,show=False)
    


