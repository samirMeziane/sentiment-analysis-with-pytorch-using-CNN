import os
import math
import time
import pickle
import argparse
import torch 
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import random
from datetime import datetime
import torch.optim as optim
import gensim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import time

from utils import build_w2c, build_w2i, build_dataset,build_dataset_1,build_batch, associate_parameters, forwards, sort_data_by_length, binary_pred, make_emb_zero, init_V,binary_accuracy
from layers import CNN

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

RANDOM_SEED = 34
np.random.seed(RANDOM_SEED)



RESULTS_DIR = './results/' + datetime.now().strftime('%Y%m%d%H%M')
try:
    os.mkdir('results')
except:
    pass
try:
    os.mkdir(RESULTS_DIR)
except:
    pass




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model,train_x,valid_x, criterion,optimizer):
    leng=max(len(data) for data in train_x)
    for j in range(len(train_x)):
        if(len(train_x[j])<leng):
            train_x[j]+=[0]*(leng-len(train_x[j]))
    optimizer.zero_grad()
        
    predictions = model(torch.tensor(train_x,dtype=torch.long)).squeeze(1)
    #print(predictions)
    #print(torch.tensor(valid_x))
    loss = criterion(predictions, torch.tensor(valid_x,dtype=torch.float))
    acc = binary_accuracy(predictions, torch.tensor(valid_x,dtype=torch.float))
    loss.backward()
    #loss=nn.    
    optimizer.step()
    epoch_loss=0
    epoch_acc=0    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
        
    return epoch_loss , epoch_acc 


def evaluate(model,train_x,valid_x, criterion,optimizer):
    
    leng=max(len(data) for data in train_x)
    for j in range(len(train_x)):
        if(len(train_x[j])<leng):
            train_x[j]+=[0]*(leng-len(train_x[j]))
    optimizer.zero_grad()

        
    predictions = model(torch.tensor(train_x,dtype=torch.long)).squeeze(1)
    loss = criterion(predictions, torch.tensor(valid_x,dtype=torch.float))
    acc = binary_accuracy(predictions, torch.tensor(valid_x,dtype=torch.float))
        
    optimizer.step()
    epoch_loss=0
    epoch_acc=0    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
        
    return epoch_loss , epoch_acc 


def main():
    print("bonjour")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = 8
    WIN_SIZES = [3,4,5]
    BATCH_SIZE = 64
    EMB_DIM = 300
    OUT_DIM = 1
    L2_NORM_LIM = 3.0
    NUM_FIL = 100
    DROPOUT_PROB = 0.5
    V_STRATEGY = 'static'
    ALLOC_MEM = 4096

    if V_STRATEGY in ['rand', 'static', 'non-static']:
        NUM_CHA = 1
    else:
        NUM_CHA = 2

    # FILE paths
    W2V_PATH     = 'GoogleNews-vectors-negative300.bin'
    TRAIN_X_PATH = 'train_x.txt'
    TRAIN_Y_PATH = 'train_y.txt'
    VALID_X_PATH = 'valid_x.txt'
    VALID_Y_PATH = 'valid_y.txt'


    # Load pretrained embeddings
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
    vocab = pretrained_model.wv.vocab.keys()
    w2v = pretrained_model.wv

    # Build dataset =======================================================================================================
    w2c = build_w2c(TRAIN_X_PATH, vocab=vocab)
    w2i, i2w = build_w2i(TRAIN_X_PATH, w2c, unk='unk')
    train_x, train_y = build_dataset(TRAIN_X_PATH, TRAIN_Y_PATH, w2i, unk='unk')
    valid_x, valid_y = build_dataset(VALID_X_PATH, VALID_Y_PATH, w2i, unk='unk')
    train_x, train_y = sort_data_by_length(train_x, train_y)
    valid_x, valid_y = sort_data_by_length(valid_x, valid_y)
    VOCAB_SIZE = len(w2i)
    print('VOCAB_SIZE:', VOCAB_SIZE)
    
    V_init = init_V(w2v, w2i)
    

    with open(os.path.join(RESULTS_DIR, './w2i.dump'), 'wb') as f_w2i, open(os.path.join(RESULTS_DIR, './i2w.dump'), 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

    # Build model =================================================================================
 
    model=CNN(VOCAB_SIZE, EMB_DIM, NUM_FIL, WIN_SIZES, OUT_DIM, 
                 DROPOUT_PROB, len(w2i))


    # Train model ================================================================================
   
    pretrained_embeddings = torch.tensor(V_init)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[len(w2i)-1] = torch.zeros(EMB_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)    
    criterion = criterion.to(device)
    n_batches_train = int(len(train_x)/BATCH_SIZE)
    n_batches_valid = int(len(valid_x)/BATCH_SIZE)
    #print(len(train_x))
    
    best_valid_loss = float('inf')

    for j in range(N_EPOCHS):


        start_time = time.time()
        epoch_loss = 0
        epoch_acc = 0 
        epoch_loss = 0
        epoch_acc = 0
  
  
    
        for i in range(n_batches_train-1):
            start = i*BATCH_SIZE
            end = start+BATCH_SIZE      
            train_loss, train_acc = train(model,train_x[start:end],train_y[start:end], criterion,optimizer)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
            #if valid_loss < best_valid_loss:
             #   best_valid_loss = valid_loss
              #  torch.save(model.state_dict(), 'tut4-model.pt')
        
        for k in range(n_batches_valid-1):
            start = k*BATCH_SIZE
            end = start+BATCH_SIZE      
            valid_loss, valid_acc = evaluate(model,valid_x[start:end],valid_y[start:end], criterion,optimizer)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {j } | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    torch.save(model.state_dict(), 'training.pt')
    return model

    #df_input.columns = ["Id" , "Review" , "Golden"]
    
    # Load model



    #with open (W2I_FILE , 'rb') as f_w2i , open (I2W_FILE , 'rb') as f_i2w:
     #   w2i = pickle.load (f_w2i)
      #  i2w = pickle.load (f_i2w)
    
def predict(df_input,model,batch_size):
    n_batches_test = int(len(df_input)/batch_size)
    for i in range(n_batches_test):
        start = i*batch_size
        end = start+batch_size
        test_x=df_input[start:end]
        leng=max(len(data) for data in test_x)
        for j in range(len(test_x)):
            if(len(test_x[j])<leng):
                test_x[j]+=[0]*(leng-len(test_x[j]))
        pred=model(torch.tensor(test_x,dtype=torch.long)).squeeze(1)
        pred=torch.sigmoid(pred)
        print(pred)

if __name__ == '__main__':
    model=main()
    INPUT_FILE = 'sentiment_dataset_cnn.csv'
    OUTPUT_FILE = 'Apred_sentiment_cnn.csv'
    W2I_FILE = 'results/202001121339/w2i.dump'
    I2W_FILE = 'results/202001121339/i2w.dump'
    ALLOC_MEM = 1024
    df_input = pd.read_csv (INPUT_FILE , sep=";" , encoding="ISO-8859-1")
    with open (W2I_FILE , 'rb') as f_w2i , open (I2W_FILE , 'rb') as f_i2w:
        w2i = pickle.load (f_w2i)
        i2w = pickle.load (f_i2w)
    
    test_X = build_dataset_1(list(df_input["Review"]) , w2i=w2i , unksym='unk')[1:]
    
    predict(test_X,model,64)