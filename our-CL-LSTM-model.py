import os
from configargparse import ArgParser
import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from tensorboardX import SummaryWriter
import time
import yaml

def generatedatafluxnet(datadir,dataname):
    # Open fluxnetdataset and create data generators
    # TODO: Flexible input data
    z = pd.read_csv(f'{datadir}{dataname}.csv')

    timedata =z.loc[:, ['TIMESTAMP_START']]
    # time variables:year.
    yeartimedata=pd.DataFrame((timedata/10e7),dtype='int')
    yeartimedata.rename(columns={'TIMESTAMP_START':'yeartimedata'}, inplace = True)

    # time variables:month.
    monthtimedata = pd.DataFrame(pd.DataFrame((timedata / 10e5), dtype='int')%1e2, dtype='int')
    monthtimedata.rename(columns={'TIMESTAMP_START': 'monthtimedata'}, inplace=True)

    # time variables:day.
    daytimedata = pd.DataFrame((pd.DataFrame((timedata/10e3),dtype='int')%1e2), dtype='int')
    daytimedata.rename(columns={'TIMESTAMP_START': 'daytimedata'}, inplace=True)

    # time variables:hour.
    hourtimedata = pd.DataFrame((pd.DataFrame((timedata / 10e1), dtype='int') % 1e2), dtype='int')
    hourtimedata.rename(columns={'TIMESTAMP_START': 'hourtimedata'}, inplace=True)


    # surface soil temperature variables.
    Tsdata = pd.DataFrame(z.loc[:, ['TS_F_MDS_1']])
    # precipitation  variables.
    Pdata = pd.DataFrame(z.loc[:, ['P_ERA']])
    # soil moisture  variables.
    SWCdata= pd.DataFrame(z.loc[:, ['SWC_F_MDS_1']])

    # concat all variables.
    ds1 = pd.concat([monthtimedata,daytimedata,hourtimedata,Tsdata,Pdata,SWCdata],axis=1,join='inner')
    # ds1 = pd.concat([monthtimedata, daytimedata, hourtimedata, Tsdata, SWCdata], axis=1, join='inner')
    # soil moisture  variables normal.

    ds=SWCdata
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(ds)
    # convert all variables to array.
    data = ds1.values

    return data,scaler


def LSTMDataGenerator(data, lead_time, batch_size,seq_length,max_lead_time):

    # Currently, we have a big sequence of half hourly cases. We’ll convert it into smaller ones:
    train_xt, train_yt, train_zt= create_sequences(data, seq_length,lead_time,max_lead_time)
    train_xt = torch.from_numpy(train_xt).float().cuda()
    train_yt = torch.from_numpy(train_yt).float().cuda()
    train_zt = torch.from_numpy(train_zt).float().cuda()
    #
    return train_xt,train_yt,train_zt

def create_sequences(data, seq_length,lead_time,max_lead_time):
    xs = []
    ys = []
    zs = []
    for i in range(len(data)-seq_length-max_lead_time-1):
        x = data[i:(i+seq_length)]
        ytemp = data[(i+seq_length+lead_time):(i+seq_length+max_lead_time),data.shape[1]-1]
        y=ytemp[::lead_time]
        y =np.append(y, ytemp[-1])
        z = data[i + seq_length + max_lead_time, data.shape[1]-1]
        xs.append(x)
        ys.append(y)
        zs.append(z)
    # return xs, ys
    return np.array(xs), np.array(ys),np.array(zs)

class build_lstm(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_layers):
        super(build_lstm, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, hidden = self.lstm(x)
        output = output.permute(1, 0, 2)
        return output, hidden

class lstm_linear(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_layers,n_Linearfeatures):
        super(lstm_linear, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers)

        self.linear = torch.nn.Linear(in_features=n_Linearfeatures, out_features=1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, hidden = self.lstm(x)
        output = output.permute(1, 0, 2)
        output = output.view(-1,output.shape[1])
        output = self.linear(output)
        return output


def train_lstm(encoder,decoder,lr,total_epoch,train_loader,data_valid_x,data_valid_y,model_save_fn,max_lead_time,lead_time, hidden_size,num_layers):
    if (os.path.exists(model_save_fn)):
        print('model has existed')
    else:
        LSTM_linear_net = lstm_linear(n_features=1, n_hidden=hidden_size, n_layers=num_layers,n_Linearfeatures=(max_lead_time // lead_time)*2)
        LSTM_linear_net =LSTM_linear_net.cuda()
        sumWriter = SummaryWriter('our_lstmencoder_log')
        params = list(encoder.parameters()) + list(decoder.parameters())+list(LSTM_linear_net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        global_step = 1

        ########## training set##########
        for epoch in range(total_epoch):
            for step, (x, y) in enumerate(train_loader):
                ###LSTM encoder and decoder
                encoderout, hidden = encoder(x)
                decoder_input = encoderout[:, -1:].repeat(1, max_lead_time // lead_time, 1)
                decoderout, decoderout_state = decoder(decoder_input)
                decoderout=decoderout.squeeze(-1)
                ###LSTM encoder and decoder loss function
                MSE_loss = torch.mean(torch.pow((decoderout - y), 2)).cuda()

                ###LSTM_linear_net
                originainput=x[:,-(max_lead_time // lead_time):,-1]
                stateinput=torch.cat((originainput, decoderout), 1)
                stateinput = (stateinput - torch.min(stateinput)) / (torch.max(stateinput)-torch.min(stateinput))
                stateinput = stateinput.unsqueeze(-1)
                #normalize
                out_state = LSTM_linear_net(stateinput)
                out_state=out_state.squeeze(-1)
                ###LSTM_linear_net loss function
                MSE_loss_state = torch.mean(torch.pow((out_state - y[:,-1]), 2)).cuda()
                ####backward
                optimizer.zero_grad()
                MSE_loss.backward(retain_graph=True)
                MSE_loss_state.backward()
                optimizer.step()
                global_step = global_step + 1
            sumWriter.add_scalar("train_loss", MSE_loss.item() / global_step, global_step=global_step)
            # # #########validation#####################
            encoderout, encoderout_state = encoder(data_valid_x)
            decoder_input = encoderout[:, -1:].repeat(1, max_lead_time // lead_time, 1)
            decoderout, decoderout_state = decoder(decoder_input)
            decoderout = decoderout.squeeze(-1)

            valid_loss = torch.mean(torch.pow((decoderout - data_valid_y), 2)).cuda()

            # ########################################################
            print(epoch, 'tain_Loss:', MSE_loss.item(), 'validation_Loss:', valid_loss.item())

        # ########################################################
        # state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        # torch.save(state, model_save_fn)
        # print(f'Saving model weights: {model_save_fn}')
        sumWriter.close()
        return encoder,decoder

def create_predictions(encoder, decoder, data_test_x,scaler, lead_time,seq_length,max_lead_time):
    """Create  predictions"""
    encoderout, encoderout_state = encoder(data_test_x)
    decoder_input = encoderout[:, -1:].repeat(1, max_lead_time // lead_time, 1)
    decoderout, decoderout_state = decoder(decoder_input)
    decoderout = decoderout.squeeze(-1)
    preds=decoderout[:,-1]
    # Unnormalize
    preds = preds.cpu()
    preds=preds.detach().numpy()
    # Unnormalize
    preds = preds.reshape(-1, 1)
    preds=scaler.inverse_transform(preds)
    return preds

def compute_rmse_r2(data_testd_y,pred,modelname):
    # 计算MSE
    rmse = np.sqrt(mean_squared_error(data_testd_y ,pred))
    mae=mean_absolute_error(data_testd_y, pred)
    r2=r2_score(data_testd_y ,pred)
    print(f"均方误差(RMSE)：{rmse}")
    print(f"均方误差(MAE)：{mae}")
    print(f"测试集R^2：{r2}")

    plt.figure(figsize=(12, 8))
    plt.plot(data_testd_y, label='True soil moisture')
    plt.plot(pred, label='Soil moisture prediction ')
    plt.legend(loc='best')
    plt.text(10, 10, 'R2=%.3f' % r2, fontdict={'size': 20, 'color': 'b'}
             ,verticalalignment="bottom",horizontalalignment="left" )
    plt.title(modelname)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(data_testd_y-pred, label='error')
    plt.legend(loc='best')
    plt.show()

def Resultsave(y_test,filename,dir):
    county = 0
    countz = 1
    f=os.path.join(dir, filename)
    xl = xlsxwriter.Workbook(f)
    sheet = xl.add_worksheet()
    for i in range(y_test.shape[0]):
        sheet.write(i, county, y_test[i][0])
        sheet.write(i, countz, y_test[i][1])
    xl.close()
    print("write_excel over")


def main(datadir,dataname,hidden_size,num_layers,lr,total_epoch,batch_size,lead_time,
         seq_length,model_save_fn,iterative,modelname,max_lead_time):

    # TODO: Flexible input data
    # Open fluxnetdataset and create data generators
    data, scaler=generatedatafluxnet(datadir,dataname)

    # TODO: Normalization
    data = scaler.transform(data)

    # TODO: Generate the tensor for lstm model
    [data_x, data_y,data_z] = LSTMDataGenerator(data, lead_time, batch_size, seq_length,max_lead_time)

    # TODO: Flexible valid split
    data_train_x=data_x[:int(0.8 * len(data))]
    data_train_y = data_y[:int(0.8 * len(data))]

    train_data = Data.TensorDataset(data_train_x, data_train_y)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    data_valid_x=data_x[int(0.8 * len(data)):int(0.9 * len(data))]
    data_valid_y=data_y[int(0.8 * len(data)):int(0.9 * len(data))]
    data_test_x=data_x[int(0.9 * len(data)):int(1.0 * len(data))]
    data_testd_z=data_z[int(0.9 * len(data)):int(1.0 * len(data))]
    time_start = time.time()
    # TODO: Flexible input shapes and optimizer
    encoder = build_lstm(n_features=data_x.shape[2], n_hidden=hidden_size,  n_layers=num_layers)
    decoder = build_lstm(n_features=1, n_hidden=hidden_size,  n_layers=num_layers)
    encoder.cuda()
    decoder.cuda()
    # TODO: Trian LSTM based on the training and validation sets
    encoder, decoder=train_lstm(encoder,decoder,lr,total_epoch,train_loader,data_valid_x,data_valid_y,model_save_fn,max_lead_time,lead_time, hidden_size,num_layers)
    # TODO: Create predictions based on the test sets
    pred = create_predictions(encoder, decoder, data_test_x,scaler,lead_time,seq_length,max_lead_time)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    # TODO: Computer score of R2 and RMSE
    # Unnormalize
    data_testd_z=data_testd_z.reshape(-1,1)
    data_testd_z=data_testd_z.cpu()
    data_testd_z=data_testd_z.detach().numpy()
    # Unnormalize
    data_testd_z=scaler.inverse_transform(data_testd_z)
    compute_rmse_r2(data_testd_z,pred,modelname)

    # TODO: Computer score of R2 and RMSE
    dir=r"./results"
    if not os.path.exists(dir):
        os.mkdir(dir)

    resultname=modelname+dataname+ '.csv'
    testresult = np.hstack((data_testd_z, pred))
    Resultsave(testresult,resultname,dir)


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--datadir', type=str, required=True, help='Path to data')
    p.add_argument('--dataname', type=str, required=True, help='Location to data')
    p.add_argument('--hidden_size', type=int, required=True, help='hidden sizes for lstm model')
    p.add_argument('--num_layers', type=int, required=True, help='layer unmber for lstm model')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=30, help='total epochs for training the model')
    p.add_argument('--batch_size', type=int, default=64, help='batch_size')
    p.add_argument('--lead_time', type=int, required=True, help='Forecast lead time')
    p.add_argument('--seq_length', type=int, default=96, help='input timesteps for lstm model')
    p.add_argument('--model_save_fn', type=str, required=True, help='Path to save model')
    p.add_argument('--iterative', type=bool, default=False, help='Is iterative forecast')
    p.add_argument('--modelname', type=str, required=True, help='name for prediction model')
    p.add_argument('--max_lead_time', type=int, required=True, help='max_lead_time for iterative prediction')
    args = p.parse_args()


    main(
        datadir=args.datadir,
        dataname=args.dataname,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size,
        lead_time=args.lead_time,
        seq_length=args.seq_length,
        model_save_fn=args.model_save_fn,
        iterative=args.iterative,
        modelname=args.modelname,
        max_lead_time=args.max_lead_time
    )

