import torch.nn as nn
from transformers import AutoModelForTokenClassification

from summary import Summary
    
    
    
class DistilBERT():
    def __init__(self, nclasses):
        super(DistilBERT, self).__init__()
        
        # define a model and a tokenizer
        model_name = 'dslim/distilbert-NER'
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # freeze first 5 layer weights        
        for name, param in self.model.named_parameters():
            if 'layer.5' not in name:
                param.requires_grad = False
                
        # Modify the last layer for n-class classification
        self.model.classifier = nn.Linear(self.model.config.hidden_size, nclasses)
        self.model.num_labels = nclasses
        
    def forward(self, x):
        return self.model(x)
    

def main():
    # Setting number of classes
    nclasses = 18
    
    device = 'cuda'
    
    # Creating model and testing output shapes 
    model = DistilBERT(nclasses=nclasses).model
    model.to(device)  # Move the underlying model to the desired device
    
    Summary(model, input_size=(64,))
    

if __name__ == "__main__": 
    main()
    
    