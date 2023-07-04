
import random
import re 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.utils import pad_sequences
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from sklearn.model_selection import KFold, train_test_split
import torch
import emoji
from transformers import AutoModel, AutoTokenizer
import os
import numpy as np
from random import sample
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confustion_matrix(model, X_test, Y_test):
  ConfusionMatrixDisplay.from_estimator(
          model,
          X_test,
          Y_test,
          display_labels=["NoHate","Hate"],
          cmap=plt.cm.Blues,
          normalize="true",
      )

  plt.show()

def heatmap(embeddings):
  array_similarity = squareform(pdist(embeddings, metric='euclidean'))
  sns.heatmap(array_similarity)
  plt.title('visualizing sentence semantic similarity')
  plt.show()


def balance_dataset(x,y_ref) -> tuple:
    number_of_0 = [value for value in y_ref if value == '0']
    number_of_1 = [value for value in y_ref if value != '0']

    elements_to_remove = abs(len(number_of_0) - len(number_of_1))

    indexes_no_hate_elements = []
    for i,elem in enumerate(y_ref):
        if elem == '0':
            indexes_no_hate_elements.append(i)
       
    indexes_element_to_remove = sample(population=indexes_no_hate_elements,k=elements_to_remove)

    x = [x for i,x in enumerate(x) if i not in indexes_element_to_remove]
    y_ref = [y for i,y in enumerate(y_ref) if i not in indexes_element_to_remove]

    return x, y_ref

def replace_substring_with_another_substring(text, substring_to_replace, substring_to_replace_with):
  return text.replace(substring_to_replace, substring_to_replace_with)

def is_token_length_compatible_with_bert(x):
  max_length = 512
  for elem in x:
    if len(elem) > max_length:
      return False
  return True

def token_with_max_length(x):
  max_length_token = 0
  for elem in x:
      if len(elem) > max_length_token:
        max_length_token = len(elem)
  return max_length_token

def create_attention_mask(input_id):
  attention_masks = []
  for sent in input_id:
    att_mask = [int(token_id > 0) for token_id in sent]  # create a list of 0 and 1.
    attention_masks.append(att_mask)  # basically attention_masks is a list of list
  return attention_masks

def handle_emoji(text):
  return emoji.demojize(text,language="it")

def pca(dataset,n_components=2):
  pca = PCA(n_components=n_components)
  pca.fit(dataset)
  return pca.transform(dataset)

def read_dataset(input_file):
  text_to_id_map = {}
  examples = []
  labels = []
  with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
      contents = f.read()
      file_as_list = contents.splitlines()
      random.shuffle(file_as_list)
      for line in file_as_list:
          if line.startswith("id"):
            continue
          split = line.split("\t")
          text = split[1]
          label = split[2]
          text_to_id_map[text] = split[0]
          labels.append(label)
          examples.append(text)
      f.close()
  return examples, labels

def plot_scatter(dataset, y_ref):
  colors = np.array(["black", "green"])
  plt.figure(figsize=(10,10))
  for i in range(len(dataset)):
    if y_ref[i] == 1:
      plt.scatter(dataset[i,0],dataset[i,1],c=colors[int(y_ref[i])], label="Hate speech")
    else:
      plt.scatter(dataset[i,0],dataset[i,1],c=colors[int(y_ref[i])], label="Not hate speech")
  plt.title('2D PCA projection of embedded sentences from BERT')
  plt.show()

  
def pre_process_data(x,tokenizer):

  # lowercase all the sentences
  x = [x.lower() for x in x]
  # delete @user and URL, because they are not useful for the classification task
  x = list(map(lambda x: replace_substring_with_another_substring(x, "@user", ""),x))
  x = list(map(lambda x: replace_substring_with_another_substring(x, "url", ""),x))
  x = list(map(lambda x: handle_emoji(x),x))

  # tokenization according to the BERT tokenizer
  x = list(map(lambda x: tokenizer.encode(x,add_special_tokens=True),x))

  #check if every token is compatible with BERT
  if not is_token_length_compatible_with_bert(x):
    raise Exception("Some token is too long for BERT")
  
  # add padding for shorter sentences
  input_ids = pad_sequences(x, maxlen=token_with_max_length(x) , dtype="long", value=0, truncating="post", padding="post")

  # create attention masks to avoid to attend to padding tokens
  input_masks = create_attention_mask(input_ids)

  # convert ids and masks to pytorch tensors
  input_ids = torch.tensor(input_ids)  
  attention_mask = torch.tensor(input_masks)

  return input_ids, attention_mask


  

def inizialization(model_name="dbmdz/bert-base-italian-uncased"):

  tokenizer = AutoTokenizer.from_pretrained(model_name)

  model = AutoModel.from_pretrained(model_name)

  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  return model, tokenizer


def get_embeddings(model, input_ids, attention_mask,model_name,batch_size=256):

# check if cuda is available
  if torch.cuda.is_available():
      device = torch.device("cuda")
      model.to(device)
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
      print('We will use the GPU:', torch.cuda.get_device_name(0))


  if not os.path.exists(os.path.join(os.getcwd(),f"{model_name.replace('/','_')}_embeddings.csv")):
    # feed forward pass to BERT
    with torch.no_grad():

      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)

      #split input in batches 
      input_ids = torch.split(input_ids, batch_size)
      attention_mask = torch.split(attention_mask, batch_size )

      # get the last hidden state of the first token of the sequence (the [CLS] token)
      last_hidden_states = []
      for i in range(len(input_ids)):
    
          last_hidden_states.append(model(input_ids[i], attention_mask=attention_mask[i])[0][:,0,:].detach().cpu().numpy())
          print(f"batch {i} of {len(input_ids)} done")

      cls_tokens = np.concatenate(last_hidden_states, axis=0) 

    # save the embeddings
    np.savetxt(f"{model_name.replace('/','_') }_embeddings.csv",cls_tokens,delimiter=",")

  else: 
    cls_tokens = np.loadtxt(f"{model_name.replace('/','_') }_embeddings.csv",delimiter=",")
  
  
  return cls_tokens
  

def evaluate(model, X_test, Y_test):
  y_hyp = model.predict(X_test)
  
  a = accuracy_score(Y_test,y_hyp)
  p = precision_score(Y_test, y_hyp, pos_label="1")
  r = recall_score(Y_test, y_hyp, pos_label="1")
  f1 = f1_score(Y_test, y_hyp, pos_label="1")
  print("precision: " + str(p) )
  print("recall: " + str(r) )
  print("accuracy: " + str(a) )
  print("f1: " + str(f1) )


haspedee_dataset_path = 'data/haspeede2_dev_taskAB.tsv'
model_name = "dbmdz/bert-base-italian-uncased"

model, tokenizer = inizialization(model_name=model_name)

x, y_ref = read_dataset(haspedee_dataset_path)

x, y_ref = balance_dataset(x,y_ref)

inputs_id, attention_mask = pre_process_data(x,tokenizer)

embeddings = get_embeddings(model, inputs_id, attention_mask,model_name,batch_size=512)

X_train, X_test, Y_train, Y_test = train_test_split(embeddings,y_ref,test_size=0.30,random_state=454657)

model = LogisticRegressionCV(n_jobs=-1,cv=KFold(n_splits=10,random_state=23445547,shuffle=True),solver='liblinear')

model.fit(X_train,Y_train)

evaluate(model, X_test, Y_test)

plot_confustion_matrix(model, X_test, Y_test)





