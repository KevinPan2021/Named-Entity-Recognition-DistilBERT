import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics import MulticlassF1Score


from visualization import plot_training_curves

    

# feedforward without gradient updates
@torch.no_grad()
def feedforward(model, dataloader):
    model.eval()
    
    running_acc = 0.0
    running_loss = 0.0
    f1_metric = MulticlassF1Score(num_classes=model.num_labels, average='weighted')
    
    device = next(model.parameters()).device
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # mixed precision
            with autocast(dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            loss = outputs.loss
            running_loss += loss.item()
            
            pred = torch.argmax(logits, dim=-1)
            
            # mask the pred and label (ignore padding)
            masked_labels = labels[attention_mask == 1]  # Ignore padding
            masked_pred = pred[attention_mask == 1]
            
            # compute acc           
            running_acc += (masked_pred == masked_labels).float().mean().item()
            
            # compute f1 score
            f1_metric.update(masked_pred.flatten(), masked_labels.flatten())
            f1 = f1_metric.compute().item()

            # Update tqdm description with loss, accuracy, and f1 score
            pbar.set_postfix({
                'Loss': running_loss/(i+1), 
                'Acc': round(100*running_acc/(i+1),1), 
                'F1': round(100*f1,1)
            })
            pbar.update(1)
            
    # averaging over all batches
    running_loss /= len(dataloader)
    running_acc /= len(dataloader)
    running_f1 = f1_metric.compute().item()
    
    return running_loss, running_acc*100, running_f1*100




# back propagation with gradient updates
def backpropagation(model, dataloader, optimizer, scaler):
    model.train()
    
    running_acc = 0.0
    running_loss = 0.0
    f1_metric = MulticlassF1Score(num_classes=model.num_labels, average='weighted')
    
    device = next(model.parameters()).device
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, batch in enumerate(dataloader):
        
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # mixed precision
            with autocast(dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
            logits = outputs.logits
            loss = outputs.loss
            
            # Add the loss to the running loss
            running_loss += loss.item()
            
            pred = torch.argmax(logits, dim=-1)
            
            # mask the pred and label (ignore padding)
            masked_labels = labels[attention_mask == 1]  # Ignore padding
            masked_pred = pred[attention_mask == 1]
            
            # compute acc           
            running_acc += (masked_pred == masked_labels).float().mean().item()
            
            # compute f1 score
            f1_metric.update(masked_pred.flatten(), masked_labels.flatten())
            f1 = f1_metric.compute().item()
            
            # Reset gradients
            optimizer.zero_grad()
    
            # Backpropagate the loss
            scaler.scale(loss).backward()
    
            # Optimization step
            scaler.step(optimizer)
    
            # Updates the scale for next iteration.
            scaler.update()
            
            # Update tqdm description with loss, accuracy, and f1 score
            pbar.set_postfix({
                'Loss': running_loss/(i+1), 
                'Acc': round(100*running_acc/(i+1),1), 
                'F1': round(100*f1,1)
            })
            pbar.update(1)
    
    # averaging over all batches
    running_loss /= len(dataloader)
    running_acc /= len(dataloader)
    running_f1 = f1_metric.compute().item()
    
    return running_loss, running_acc*100, running_f1*100
    

    
# model training loop
def model_finetuning(model, train_loader, valid_loader, device):
    learning_rate = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_epochs = 10
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    
    # get the initial statistics
    print(f'Epoch 0/{n_epochs}')
    train_loss, train_acc, train_f1 = feedforward(model, train_loader)
    valid_loss, valid_acc, valid_f1 = feedforward(model, valid_loader)
    
    # training curves
    train_losses, train_accs, train_f1s = [train_loss], [train_acc], [train_f1]
    valid_losses, valid_accs, valid_f1s = [valid_loss], [valid_acc], [valid_f1]
    
    # saving criteria
    best_valid_loss = valid_loss
    
    # training epoches
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        # feedforward to estimate loss
        train_loss, train_acc, train_f1 = backpropagation(model, train_loader, optimizer, scaler)
        valid_loss, valid_acc, valid_f1 = feedforward(model, valid_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)
        
        # strictly better
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}_finetuned.pth')
            
            
    plot_training_curves(
        train_accs, train_losses, train_f1s,
        valid_accs, valid_losses, valid_f1s
    )
    