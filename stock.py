import numpy as np #arrays
import pandas as pd #data set handler
import yfinance as yf #data
import matplotlib.pyplot as plt #visuals

#neural network
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler #scale
from sklearn.metrics import root_mean_squared_error #evaluate model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ticker = 'GOOG'
df = yf.download(ticker, "2020-01-01")#info from 2020

scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df["Close"])

seq_length = 30 #look at last 29 days predict 30th
data = []

for i in range(len(df)- seq_length):
    data.append(df.Close[i:i+seq_length]) 
    #loop through every data point and group it with the next 30 days inside an array

data = np.array(data) #make it easier to handle

train_size = int(0.8 * len(data)) #train on 80% of data

#a tensor is basically a (numpy) array but it can be ran on either a cpu or gpu
x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
#train on first 80% of arrays in data, train on every index except last
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
x_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)#may not include 2nd colon
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

class PredictionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel,self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        #LSTM means long short term memory (type of nn)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)#could add dropout=0.2  
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device) # zeros creates a tensor filled with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out
    
model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss() #meas squared error
optimizer = optim.Adam(model.parameters(), lr=0.01)#lr = learning rate

num_epochs = 200 #number of loops through the whole dataset

for i in range(num_epochs):
    #batch size = train_size
    y_train_pred = model(x_train) #runs data through model

    loss = criterion(y_train_pred, y_train) #calculates the loss/error

    if i % 25 == 0:
        print(i,loss.item())

    optimizer.zero_grad() 
    loss.backward()#back propogate
    optimizer.step() #updates gradients and the re-run the whole dataset through the model again

model.eval()

y_test_pred = model(x_test) # after model has been trained run predictions on the remaining unseen data

y_train_pred = scaler.inverse_transform(y_train_pred.detach().cpu().numpy()) #predictions for latest iteration through training data
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())#actual prices for training data
y_test_pred = scaler.inverse_transform(y_test_pred.detach().cpu().numpy()) #predictions for the unseen test data
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

train_rmse = root_mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_pred[:, 0])

fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4,1)#4rows 1 column

ax1 = fig.add_subplot(gs[:3,0])
# plot(dates, predictions, color, name) iloc used pandas datafram structure
ax1.plot(df.iloc[-len(y_test):].index, y_test, color= "blue", label = "Actual Price")
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color= "green", label = "Predicted Price")
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

ax2 = fig.add_subplot(gs[3,0])
ax2.axhline(test_rmse, color ="blue", linestyle="--", label="RMSE")
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), "r", label="Prediction Error")
ax2.legend()
plt.title("Prediction Error")
plt.xlabel("Date")
plt.ylabel("Error")

plt.tight_layout()
plt.show()