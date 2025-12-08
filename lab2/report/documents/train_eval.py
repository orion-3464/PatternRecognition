import torch.optim as optim
import itertools
from sklearn.utils import class_weight

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, overfit_batch=False):
    model.to(device)
    early_stopping = EarlyStopping(patience=20, mode="min", base_model=None)
    sampling_epochs = 1

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )

    train_loss=1.0
    train_acc=0.0
    running_epochs=0

    train_losses=[]
    train_accs=[]
    val_losses=[]
    val_accs=[]

    if overfit_batch is True:
        print("Training LSTM with overfit batch")
        data = list(itertools.islice(train_loader, 4))

        while(train_loss > 0.05 or train_acc < 1.0):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            num_batches = 0
            for num_batch, batch in enumerate(data):
                features, labels, lengths = batch
                features = features.to(device).float()
                labels = labels.to(device)
                lengths = lengths.to(device)

				# zero grads in the optimizer
                optimizer.zero_grad()
				# run forward pass
                outputs = model(features, lengths)
				# calculate loss
                loss = criterion(outputs, labels)
                train_acc = accuracy(outputs, labels)

				# Run backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                running_acc += train_acc
                num_batches += 1
            
            train_loss = running_loss / num_batches
            train_losses.append(train_loss)
            train_accs.append(running_acc / num_batches)

            scheduler.step(train_loss)
            
            print(f"Epoch: {running_epochs}, Loss: {train_loss}, Accuracy: {train_accs[-1]}")
			

            running_epochs = running_epochs+1

            if running_epochs > 100000:
                print("The model failed to overfit")
                return None
	
        print(f"The model indeed overfitted at epoch: {running_epochs}.")
        return train_losses, train_accs, None, None

    else:
        valid_acc = 0.0
        stopped_early = False
        for epoch in range(num_epochs):
            if stopped_early:
                break
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            num_batches = 0
            for num_batch, batch in enumerate(train_loader):
                features, labels, lengths = batch
                features = features.to(device).float()
                labels = labels.to(device)
                lengths = lengths.to(device)

				# zero grads in the optimizer
                optimizer.zero_grad()
				# run forward pass
                outputs = model(features, lengths)
				# calculate loss
                loss = criterion(outputs, labels)
                train_acc = accuracy(outputs, labels)

				# Run backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                running_acc += train_acc
                num_batches += 1
            
            train_loss = running_loss / num_batches
            train_losses.append(train_loss)
            train_accs.append(running_acc / num_batches)
            
            if epoch%sampling_epochs == 0: #possibly change, but keep best overall model...
                valid_loss, y_pred, y_true = evaluate(model, val_loader, criterion, device)
                valid_acc = (y_pred == y_true).sum().item() / len(y_true)
                val_losses.append(valid_loss)
                val_accs.append(valid_acc)

                scheduler.step(valid_loss)
                
                if early_stopping.stop(valid_loss, model):
                    stopped_early = True

            print(f"Epoch: {epoch}, Loss: {train_loss}, Accuracy: {train_accs[-1]}" + f", Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}" if epoch%sampling_epochs == 0 else "")
            if stopped_early:
                print("early stopping...")
                break

        model.load_state_dict(torch.load(checkpoint_filename, weights_only=True))
        return train_losses, train_accs, val_losses, val_accs

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch

            # TODO: YOUR CODE HERE
            features = features.to(device).float()
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Run forward pass
            logits = model(features, lengths)
            # calculate loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predict
            outputs = torch.argmax(logits, dim=1) # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs.cpu()))
            y_true = torch.cat((y_true, labels.cpu()))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true
