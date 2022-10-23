#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
import sys
sys.path.append('./model')
from LCNN_model import Wav2Vec2_LCNN as wav2vec2_lcnn
from CNN_outcome import CNN as cnn_outcome
import sys
sys.path.append('./utils')
from data_reader import train_loader
import time
import torch
from torch.utils.data import Dataset,DataLoader
from logger import get_logger
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
import librosa
import math
import random

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    #gpu setting
    torch.cuda.empty_cache()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_NUM = "0" # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)  
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    #log
    os.makedirs('./log/', exist_ok=True)
    
    
    logger = get_logger('./log/'  + 'model_log')

    

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present','Unknown', 'Absent']  
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    
    
    # Define the model /murmur
#     wav2vec2_lcnn_murmur = wav2vec2_lcnn("murmur")
#     wav2vec2_lcnn_murmur = wav2vec2_lcnn_murmur.cuda(device)
    
    # Define the model /outcome
    cnn_outcome_model = cnn_outcome()
    cnn_outcome_model = cnn_outcome_model.cuda(device)
    
    
    # optimizer, loss
    opt_murmur = torch.optim.Adam(wav2vec2_lcnn_murmur.parameters(), lr=1e-6, betas=(0.9, 0.999))
    scheduler_murmur = torch.optim.lr_scheduler.ExponentialLR(opt_murmur, 0.96)
    criterion_outcome = torch.nn.CrossEntropyLoss()
    criterion_murmur = torch.nn.BCELoss()
    opt_outcome = torch.optim.Adam(cnn_outcome_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler_outcome = torch.optim.lr_scheduler.ExponentialLR(opt_outcome, 0.96)
    
    
    #
    
    #data loader
    trainset = train_loader(data_folder)
#     validset = train_loader(args.valid_data_folder )


    dataloader_train_murmur = DataLoader(dataset=trainset,
                    batch_size=10,
                           shuffle =True,
                           num_workers=0)
    
    dataloader_train_outcome = DataLoader(dataset=trainset,
                    batch_size=64,
                           shuffle =True,
                           num_workers=0)
    dataloader_valid = DataLoader(dataset=validset,
                    batch_size=args.minibatchsize_valid,
                           num_workers=args.dev_num_workers)
    

    # Train the model.
    all_file_train = len(trainset)
    dataloader_train_len = len(dataloader_train_murmur)
    for iter_ in range(25):
        train_pred_label,  train_gt_label = [], []
        start_time = time.time()
        running_loss = 0.0
        idx = 0
        wav2vec2_lcnn_murmur.train()
        for feature, _, _, _, _, _, _, current_murmur, _  in tqdm(dataloader_train): 
            feature = feature.cuda()
            feature = torch.tensor(feature, dtype=torch.float32)
    #         current_age_group = current_age_group.cuda()
    #         sex_feature = sex_feature.cuda()
    #         height_weight = height_weight.cuda()
    #         preg_feature = preg_feature.cuda()
    #         loc_feature = loc_feature.cuda()
            current_murmur = current_murmur.cuda()
    #         current_outcome = current_outcome.cuda()






            output_train = wav2vec2_lcnn_murmur(feature)   # output
            output_train = output_train.squeeze(1)


            output_preds_labels = (torch.ceil(output_train-0.39)).data.cpu().numpy()
            train_pred_label.extend(list(output_preds_labels))
            train_gt_label.extend(list(current_murmur.data.cpu().numpy()))

    #         murmur_weight = [4 if i==1 else 1 for i in list(current_murmur.data.cpu().numpy())]

    #         criterion = torch.nn.BCELoss(weight = torch.tensor(murmur_weight).cuda())
            loss = criterion_murmur(output_train.to(torch.float32), current_murmur.to(torch.float32))




            idx +=1              

            opt_murmur.zero_grad()
            loss.backward()
            opt_murmur.step()

            running_loss += loss.item()

        train_f1 = f1_score(y_true= train_gt_label, y_pred= train_pred_label)


        scheduler_murmur.step()
#         print("lr: ", opt.param_groups[0]['lr'])
        logger.info("Iteration:{0}, train loss = {1:.6f} ,train F1 = {2:.6f} ".format(iter_, 
                     running_loss/dataloader_train_len,train_f1))

    m_n = "murmur"
    # Save the model.
    save_challenge_model(model_folder,wav2vec2_lcnn_murmur ,m_n)
    
    
    
    all_file_train = len(trainset)
    dataloader_train_len = len(dataloader_train_outcome)

    for iter_ in range(30):
        start_time = time.time()
        running_loss = 0.0
        idx = 0
        cnn_outcome_model.train()
        for _, log_mel_feature, current_age_group, sex_feature, height_weight, preg_feature, loc_feature, _, current_outcome in tqdm(dataloader_train):
            log_mel_feature = log_mel_feature.cuda()
            current_age_group = current_age_group.cuda()
            sex_feature = sex_feature.cuda()
            height_weight = height_weight.cuda()
            preg_feature = preg_feature.cuda()
            loc_feature = loc_feature.cuda()
#             current_murmur = current_murmur.cuda()
            current_outcome = current_outcome.cuda()




            output_train = cnn_outcome_model(log_mel_feature,current_age_group,sex_feature,height_weight, preg_feature,loc_feature)
            loss = criterion_outcome(output_train, current_outcome)



            idx +=1              

            opt_outcome.zero_grad()
            loss.backward()
            opt_outcome.step()

            running_loss += loss.item()


        scheduler_outcome.step()
#         print("lr: ", opt.param_groups[0]['lr'])
        logger.info("Iteration:{0}, train loss = {1:.6f}".format(iter_, 
                     running_loss/dataloader_train_len)) #,train F1 = {2:.6f} ,train ACC = {3:.6f} /,train_f1,train_acc
    
    m_n = "outcome"
    # Save the model.
    save_challenge_model(model_folder,cnn_outcome_model, m_n)
    
    

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    murmur_train_model = torch.load(model_folder + '/' + 'murmur.model')
    outcome_train_model = torch.load(model_folder + '/' + 'outcome.model')
    return [murmur_train_model, outcome_train_model]

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    murmur_model , outcome_model = model

    GPU_NUM = "0" # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU

    
    # Define the model
    murmur_classifier = wav2vec2_lcnn("murmur")
    murmur_classifier = murmur_classifier.cuda(device)
    
    outcome_classifier = cnn_outcome()
    outcome_classifier = outcome_classifier.cuda(device)
    
    murmur_classifier.load_state_dict(murmur_model)
    outcome_classifier.load_state_dict(outcome_model)
    
#     wav_files = glob.glob(data_folder + '*.wav')
#     num_wav_files = len(wav_files)
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    
    age_classes = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    
    num_recordings = len(recordings)
    
    audio_length=20
    sr =4000
    recordings_list=[]
    for i in range(num_recordings):
        
        recording = recordings[i]
        recording = recording*1.0
        recording = recording[4000:-4000]


        length = audio_length*sr
        if recording.shape[0] <= length:
            shortage = length - recording.shape[0]
            recording = np.pad(recording, (0, shortage), 'wrap')
        start_frame = np.int64(random.random()*(recording.shape[0]-length))
        recording = recording[start_frame:start_frame + length] 
        recordings_list.append(recording)
        

    features = get_features(data, recordings_list)
    
    predict_murmur_arr= np.zeros((num_recordings,))
    predict_outcome_arr= np.zeros((num_recordings,2))
    
    for i in range(num_recordings):
        
        logmel = torch.tensor(features['mel'][i]).cuda()
        age = torch.tensor(features['age']).cuda()
        sex = torch.tensor(features['sex']).cuda()
        hw = torch.tensor(features['hw']).cuda()
        preg = torch.tensor(features['preg']).cuda()
        loc = torch.tensor(features['loc'][i]).cuda()
        raw_feature = torch.tensor(features['raw'][i], dtype=torch.float32).cuda()
        
        logmel = logmel.unsqueeze(0)
        age = age.unsqueeze(0)
        sex = sex.unsqueeze(0)
        hw = hw.unsqueeze(0)
        preg = preg.unsqueeze(0)
        loc = loc.unsqueeze(0)
        raw_feature = raw_feature.unsqueeze(0)
        
        predict_murmur = murmur_classifier(raw_feature)
        predict_outcome = F.softmax(outcome_classifier(logmel,age,sex,hw,preg,loc), dim=1)
        
        predict_murmur_arr[i]= predict_murmur.data.detach().cpu().numpy()
        predict_outcome_arr[i,:] = predict_outcome.data.detach().cpu().numpy()

    
    
    
   
    # Get classifier probabilities.
    idx1 = predict_murmur_arr.argmax(axis=0)
    murmur_probabilities_max = predict_murmur_arr[idx1] 
    murmur_probabilities = np.array([murmur_probabilities_max,0,1-murmur_probabilities_max])
    idx2 = predict_outcome_arr.argmax(axis=0)[0]
    outcome_probabilities = predict_outcome_arr[idx2,]


    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    if murmur_probabilities_max > 0.39:
        murmur_labels[0] = 1
    else:
         murmur_labels[2] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
# Save your trained model.
def save_challenge_model(model_folder, model, model_name):
    torch.save(model.state_dict(), os.path.join(model_folder, "{}.model".format(model_name)))

# Extract features from the data.
import librosa
import math
import random

def get_features(data, recordings):
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    
    age_classes = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    
    num_recordings = len(recordings)
    
    
    feature_dict={}
    
    feature_dict['raw']=[]
    for i in range(num_recordings):
        raw_data = recordings[i]
        feature_dict['raw'].append(raw_data)
        
    feature_dict['mel']=[]
    for i in range(num_recordings):
        log_mel_feature = librosa.power_to_db(librosa.feature.melspectrogram(y = (recordings[i]/32768).astype(np.float32),
                                                         sr= 4000,
                                                         n_mels=128,
                                                         n_fft=400, 
                                                         hop_length=128, 
                                                         win_length=400))
        feature_dict['mel'].append(log_mel_feature)
        
        
        
        
#     recording_info_lst = data.split('\n')[1:num_recordings+1]
#     feature_dict['wav2vec2'] = []
#     for i in range(num_recordings):
#         pk_feature_fnm = recording_info_lst[i].split(' ')[2].replace('wav','pickle')
#         with open('/home/jh20/Data/nr_data/ECG/Physionet2022/physionet.org/files/circor-heart-sound/1.0.3/validation_wav2vec2/'+ pk_feature_fnm ,'rb') as fr:
#             feature = pickle.load(fr)
#         feature_dict['wav2vec2'].append(feature)
    

    # age
    current_patient_age = get_age(data)
    current_age_group = np.zeros(6, dtype=np.float32)
    if current_patient_age in age_classes:
        j = age_classes.index(current_patient_age)
        current_age_group[j] = 1.0
    else :
        current_age_group[5] = 1.0

    feature_dict['age']=current_age_group



    # sex
    sex = get_sex(data)
    sex_feature = np.zeros(2, dtype=np.float32)
    if compare_strings(sex, 'Female'):
        sex_feature[0] = 1.0
    elif compare_strings(sex, 'Male'):
        sex_feature[1] = 1.0

    feature_dict['sex']=sex_feature

    # height and weight.
    height = get_height(data)
    weight = get_weight(data)

    ## simple impute
    if math.isnan(height) :
        height = 110.846  #mean
    if math.isnan(weight) :
        weight = 23.767   #mean

    height_weight = np.array([height, weight], dtype=np.float32)


    feature_dict['hw']=height_weight

    # Extract pregnancy
    preg_feature = np.zeros(2, dtype=np.float32)
    is_pregnant = get_pregnancy_status(data)
    if is_pregnant == True:
        preg_feature[0] = 1.0
    elif is_pregnant == False:
        preg_feature[1] = 1.0

    feature_dict['preg']=preg_feature

    # Extract location
    feature_dict['loc']=[]
    locations = get_locations(data)
    for j in range(num_recordings):
        
        num_recording_locations = len(recording_locations)
        loc_feature = np.zeros(num_recording_locations, dtype=np.float32)
        if locations[j] in recording_locations:
            idx = recording_locations.index(locations[j])
            loc_feature[idx] = 1.0
            
        feature_dict['loc'].append(loc_feature)




#     # label

#     current_murmur = np.zeros(num_murmur_classes, dtype=np.float32)
#     murmur = get_murmur(data)
#     if murmur in murmur_classes:
#         j = murmur_classes.index(murmur)
#         current_murmur[j] = 1
    
#     feature_dict['murmur']=current_murmur
    
#     current_outcome = np.zeros(num_outcome_classes, dtype=np.float32)
#     outcome = get_outcome(data)
#     if outcome in outcome_classes:
#         j = outcome_classes.index(outcome)
#         current_outcome[j] = 1

#     feature_dict['outcome']=current_outcome
    
    
    
    
#     #label_2
    
#     if get_murmur(data) == self.murmur_classes[0]:
#         murmur_lable.append(1)
#     elif get_murmur(current_patient_data) == self.murmur_classes[1]:
#         murmur_lable.append(0)
#     else :
#         murmur_lable.append(0)
    
#     if get_outcome(current_patient_data) == self.outcome_classes[0]:
#         outcome_lable.append(1)
#     elif get_outcome(current_patient_data) == self.outcome_classes[1]:
#         outcome_lable.append(0)
    
    
    


    return feature_dict