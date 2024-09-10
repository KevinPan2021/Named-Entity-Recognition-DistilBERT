from torch.utils.data import Dataset
import torch

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