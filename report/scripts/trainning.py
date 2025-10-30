def train_model(model, X_train, Y_train, epochs=100, lr=0.001): 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # default learning rate small to avoid overshooting
    losses = []
    
    for epoch in range(epochs):
        model.train() # put model in trainning mode
        optimizer.zero_grad()   # clean accumulated gradients
        
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        
        loss.backward()   # backpropagate loss gradients
        optimizer.step()  # update weights
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses

