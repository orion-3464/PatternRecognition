def evaluate_model(model, X_test, Y_test):
    model.eval()    # put model in eval mode
    with torch.no_grad():   # we do not need to calculate and store gradients in inference
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, Y_test)
        mae = torch.mean(torch.abs(predictions - Y_test))
    return predictions, mse.item(), mae.item()
