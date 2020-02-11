# import os
# import pickle
# import argparse
# import pandas as pd
# import torch as t
# import torch.nn as nn
# from tqdm import tqdm
# from layers import CNN
# from utils import build_dataset_1 , associate_parameters , binary_pred , forwards

# parser = argparse.ArgumentParser (description='Convolutional Neural Networks for Sentence Classification in Pytorch')
# parser.add_argument ('--gpu' , type=int , default=-1 , help='GPU ID to use. For cpu, set -1 [default: -1]')
# parser.add_argument ('--model_file' , type=str , default='C:/Users/Marwan/Documents/texte/results/202001121339/model_e10' ,
#                       help='Model to use for prediction [default: ./model]')
# parser.add_argument ('--input_file' , type=str ,
#                       default='sentiment_dataset_cnn.csv' ,
#                       help='Input file path [default: annotation_S2]')
# parser.add_argument ('--out_file' , type=str , default='pred_sentiment_cnn.csv' ,
#                       help='Output file path [default: ./pred_yannotation_S2.txt]')
# parser.add_argument ('--w2i_file' , type=str , default='C:/Users/Marwan/Documents/texte/results/202001121339/w2i.dump' ,
#                       help='Word2Index file path [default: C:/Users/Marwan/Documents/texte/results/202001121339/w2i.dump]')
# parser.add_argument ('--i2w_file' , type=str , default='C:/Users/Marwan/Documents/texte/results/202001121339/i2w.dump' ,
#                       help='Index2Word file path [default: C:/Users/Marwan/Documents/texte/results/202001121339/i2w.dump]')
# parser.add_argument ('--alloc_mem' , type=int , default=1024 ,
#                       help='Amount of memory to allocate [mb] [default: 1024]')
# #parser.add_argument ('' , type=int , default=1024 ,
#   #                    help='Amount of memory to allocate [mb] [default: 1024]')

# #args = parser.parse_args ()

# def main():

#     os.environ['CUDA_VISIBLE_DEVICES'] = str (-1)

#     MODEL_FILE = 'training.pt'
#     INPUT_FILE = 'sentiment_dataset_cnn.csv'
#     OUTPUT_FILE = 'Apred_sentiment_cnn.csv'
#     W2I_FILE = 'results/202001121339/w2i.dump'
#     I2W_FILE = 'results/202001121339/i2w.dump'
#     ALLOC_MEM = 1024
#     df_input = pd.read_csv (INPUT_FILE , sep=";" , encoding="ISO-8859-1")
#     df_input.drop(["Pred_sentiment"],axis=1)
#     df_input.drop(["Id"],axis=1)
#     df_input.drop(["Golden"],axis=1)
#     #df_input.columns = ["Id" , "Review" , "Golden"]
    
#     # Load model



#     with open (W2I_FILE , 'rb') as f_w2i , open (I2W_FILE , 'rb') as f_i2w:
#         w2i = pickle.load (f_w2i)
#         i2w = pickle.load (f_i2w)
    
#     # #model=CNN()
    
#     # model.load_state_dict(t.load(MODEL_FILE))
#     # test_X = build_dataset_1 (INPUT_FILE , w2i=w2i , unksym='unk')[1:]
#     # #print(test_X) 
#     # for x in test_X:
        
#     #     test_x_tensor=t.tensor(x,dtype=t.long)
#     #     pred=model(test_x_tensor)

#     # Pred
#         #pred_y = []
    
#     # pred_y.append (str (float ( (y.value ()))))
#     # pred_y = pred_y[1:]
#     # P = pd.Series (pred_y).rename ("Pred_sentiment")
#     # print (P , df_input["Review"])
#     # df = pd.DataFrame ()
#     # df["Id"] = df_input["Id"]
#     # df["Review"] = df_input["Review"]
#     # df["Golden"] = df_input["Golden"]
#     # df["Pred_sentiment"] = P
#     # """
#     # df= pd.DataFrame({"Id":df_input["Id"] },
#     #                  {"Review": df_input["Golden"] },
#     #                  {"Pred_sentiment": P })
#     #                  #{"Pred_sentiment": P} )"""
#     # print(df)
#     # df.to_csv (OUTPUT_FILE , sep=";" , index=False)

#     # # with open(OUTPUT_FILE, 'w') as f:
#     # # f.write('\n'.join(pred_y))


# if __name__ == '__main__':
#     main ()