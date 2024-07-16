import torch
import numpy as np


class Model_trainer():

    def __init__(self,class_labels: list, device='cpu'):
        self.test_loss = []
        self.train_loss = []
        self.classes = class_labels
        self.prediction: dict
        self.prevepoch_loss = float('inf')
        self.device = device

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= num_batches
        return train_loss

    def test_loop(self, dataloader, model, loss_fn):
        model.eval()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0

        y_pred = []
        y_true = []

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

                y_pred.append(torch.argmax(pred,dim=-1).cpu().numpy())
                y_true.append(y.cpu().numpy())

        test_loss /= num_batches
        return test_loss,{'y_pred':y_pred,'y_true':y_true}


    def fit(self,trainloader, testloader, model, loss_fn, optimizer, scheduler, epochs=30, verbose=False):
        for t in range(epochs):
            train_loss = self.train_loop(trainloader, model, loss_fn, optimizer)
            test_loss,prediction = self.test_loop(testloader,model, loss_fn)
            scheduler.step(test_loss)
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            if(self.prevepoch_loss > test_loss):
                self.prediction = prediction
                if(verbose):
                  print('saving model predictions')
            self.prevepoch_loss = test_loss
            print('Epoch: %d, Train Loss: %.5f, Test Loss: %.5f'%(t+1,train_loss,test_loss))

        self.prediction['y_pred'] = np.concatenate(self.prediction['y_pred'])
        self.prediction['y_true'] = np.concatenate(self.prediction['y_true'])
        return {'Train_loss':self.train_loss,'Test_loss':self.test_loss,'class_label':self.classes},self.prediction
