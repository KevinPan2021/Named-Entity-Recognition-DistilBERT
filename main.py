from transformers import DistilBertTokenizerFast
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import ast
import pandas as pd
import pickle
import logging

from distilBERT import DistilBERT
from fine_tuning import model_finetuning, feedforward



# Set the logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)


# supports both CUDA and mps
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

    
# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
    
    def __len__(self):
        return len(self.key_to_value)
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key, 0)

    def get_key(self, value):
        return self.value_to_key.get(value)
    
    
def read_data(path):
    df = pd.read_csv(path)
    X = df['Sentence'].tolist()
    Y = df['Tag'].tolist()
    
    # convert to list
    X = [x.split(' ') for x in X]
    Y = [ast.literal_eval(y) for y in Y]
    return X, Y



# set token to -100 for special tokens
# mask the subword representations after the first subword 
def label_token_alignment(tokens, Y, label_all_tokens=True):
    labels = []
    for i, label in enumerate(Y):
        word_ids = tokens.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokens["labels"] = labels
    return tokens



class NER_Dataset(Dataset):
    def __init__(self, data, label_ind_map):
        self.data = data.copy()
        
        # Convert label strings to indices
        self.data['labels'] = [
            torch.tensor([label_ind_map.get_value(label) for label in labels], dtype=torch.long) for labels in self.data['labels']
        ]
        
        
    def __getitem__(self, idx):
        input_ids = self.data['input_ids'][idx]
        attention_mask = self.data['attention_mask'][idx]
        labels = self.data['labels'][idx]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': labels
        }
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    
    
# preform train, valid, test split
# train:81%, valid:9%, test:10%
def train_valid_test_split(data):
    # Convert BatchEncoding object into lists or arrays
    ids = data['input_ids']
    masks = data['attention_mask']
    labels = data['labels']
    
    # Split data into train and test sets
    ids_train, ids_test, masks_train, masks_test, labels_train, labels_test = train_test_split(
        ids, masks, labels, test_size=0.1, random_state=42
    )
    
    # Split train data into train and validation sets
    ids_train, ids_valid, masks_train, masks_valid, labels_train, labels_valid = train_test_split(
        ids_train, masks_train, labels_train, test_size=0.1, random_state=42
    )
    
    # Construct dictionaries for train, validation, and test datasets
    train = {'input_ids':ids_train, 'attention_mask':masks_train, 'labels':labels_train}
    val = {'input_ids':ids_valid, 'attention_mask':masks_valid, 'labels':labels_valid}
    test = {'input_ids':ids_test, 'attention_mask':masks_test, 'labels':labels_test}
    return train, val, test



@torch.no_grad()
def inference(model, data):
    model.eval()
    
    device = next(model.parameters()).device

    input_ids = data.clone().unsqueeze(0).to(device)  # Add batch dimension
    attention_mask = torch.ones_like(input_ids).to(device) # placeholder
    outputs = model(input_ids, attention_mask=attention_mask) 
    
    labels = torch.argmax(outputs.logits, dim=-1).squeeze().cpu()
    return labels.tolist()
    
    

def main():
    # create a ind to label map
    label_ind_map = BidirectionalMap()
    label_ind_map.add_mapping(None, 0)
    labels = [
        'O', 'I-eve', 'I-geo', 'B-eve', 'B-geo', 'I-gpe', 'I-tim', 'I-art', 'I-org', 
        'I-per', 'B-gpe', 'B-tim', 'B-art', 'I-nat', 'B-org', 'B-nat', 'B-per'
    ]
    for i in range(len(labels)):
        label_ind_map.add_mapping(labels[i], i+1) # +1 to account for {unkown, 0}
    # Save the instance to a pickle file
    with open("label_ind_map.pkl", "wb") as f:
        pickle.dump(label_ind_map, f)
    print('nclasses', len(label_ind_map))
        
    # define a model and a tokenizer
    model = DistilBERT(nclasses=len(label_ind_map)).model
    model = model.to(compute_device())
    tokenizer = DistilBertTokenizerFast.from_pretrained('dslim/distilbert-NER')

    
    # load and process text data
    X, Y = read_data('../Datasets/Named Entity Recognition (NER) Corpus/ner.csv')
    print(set([item for sublist in Y for item in sublist]))


    # tokenize
    X = tokenizer(
        X, is_split_into_words=True,
        truncation=True, padding=True
    )

    # label token alignment
    data = label_token_alignment(X, Y)
    
    # train-test split
    train, valid, test = train_valid_test_split(data)

    # dataset
    train_dataset = NER_Dataset(train, label_ind_map)
    val_dataset = NER_Dataset(valid, label_ind_map)
    test_dataset = NER_Dataset(test, label_ind_map)
    del train, valid, test
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # visualize some examples
    for i in range(0, len(train_dataset), len(train_dataset)//5):
        item = train_dataset[i]
        x = item['input_ids']
        x_decoded = [tokenizer.decode(tok) for tok in x if tok != 0]
        y = item['labels'][:len(x_decoded)] # skip the pad inds
        y = [label_ind_map.get_key(int(item)) for item in y]
        print('>', x_decoded)
        print('=', y)
        print()
         
    
    # fine tuning
    model_finetuning(model, train_loader, val_loader)
    
    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}_finetuned.pth'))
    
    # get the test dataset metrics
    print('Test Performance')
    feedforward(model, test_loader)    
    for i in range(0, len(test_dataset), len(test_dataset)//5):
        item = test_dataset[i]
        x = item['input_ids']
        x_decoded = [tokenizer.decode(tok) for tok in x if tok != 0]
        y = item['labels'][:len(x_decoded)] # skip the pad inds
        y = [label_ind_map.get_key(int(item)) for item in y]
        pred = inference(model, x.clone())[:len(x_decoded)] # skip the pad inds
        pred = [label_ind_map.get_key(int(item)) for item in pred]
        print('>', x_decoded)
        print('=', y)
        print('<', pred)
        print()
        
        
        
    

if __name__ == "__main__":
    main()