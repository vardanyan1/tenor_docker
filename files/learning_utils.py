import torch
from torch.utils.data import Dataset


class SopranoPhotonDataset(Dataset):
    def __init__(self, params, spectrum):
        self.param_in = params
        self.photon_spectrum = spectrum



    def __len__(self):
        return len(self.photon_spectrum)

    def __getitem__(self, idx):

        return torch.tensor(self.param_in[idx]) , torch.tensor(self.photon_spectrum[idx])



class SimpleModel(torch.nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()


        ## M1: not too bad:
        #self.linear1 = torch.nn.Linear(7, 1024)
        #self.activation = torch.nn.ReLU()
        #self.linear2 = torch.nn.Linear(1024, 1024)
        #self.activation2 = torch.nn.ReLU()
        #self.linear3 = torch.nn.Linear(1024, 1024)
        #self.activation3 = torch.nn.ReLU()
        #self.linear4 = torch.nn.Linear(1024, 150)
        #self.softmax = torch.nn.Tanh()
        self.linear1 = torch.nn.Linear(7, 1024)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1024, 2048)

        self.maxpool1 = torch.nn.MaxPool1d(2)

        self.linear3 = torch.nn.Linear(1024, 2048)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(2048, 2048)

        self.maxpool2 = torch.nn.MaxPool1d(2)
        #self.activation3 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(1024, 150)
        self.softmax = torch.nn.Tanh()

        ## M2:
#        self.linear1 = torch.nn.Linear(7, 1024)
#        self.activation = torch.nn.LeakyReLU()
#        self.linear2 = torch.nn.Linear(1024, 1024)
#        self.activation2 = torch.nn.ReLU()
#        self.linear3 = torch.nn.Linear(1024, 1024)
#        self.activation3 = torch.nn.ReLU()
#        self.linear4 = torch.nn.Linear(1024, 150)
#        self.softmax = torch.nn.Tanh()
    def forward(self, x):
        ## M1: not too bad:
#        x = x.to(torch.float32)
#        x = self.linear1(x)
#        x = self.activation(x)
#        x = self.linear2(x)
#        x = self.activation2(x)
#        x = self.linear3(x)
#        x = self.activation3(x)
#        x = self.linear4(x)
#        x = self.softmax(x)

        resized = False
        if (x.size()[0]) < 8:
            #print("Here:")
            x = x.view(1, 7)
            resized = True

        x = x.to(torch.float32)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.maxpool1(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.maxpool2(x)
        x = self.linear5(x)
        x = self.softmax(x)
        if resized == True:
            x = x[0]
        return x.double()



class SimpleModelSmoothed(torch.nn.Module):

    def __init__(self, batchsize, device, smoothed_version):
        super(SimpleModelSmoothed, self).__init__()

        self.batchsize = batchsize
        self.device = device
        self.smoothed_version = smoothed_version
        ## M1: not too bad:

        ## Layer 1:
        self.linear1 = torch.nn.Linear(7, 1024)
        self.activation1 = torch.nn.ReLU()
#        self.dropout1 = torch.nn.Dropout(0.1)

        self.linear2 = torch.nn.Linear(1024, 2048)
        self.maxpool2 = torch.nn.MaxPool1d(2)

        self.linear3 = torch.nn.Linear(1024, 2048)
        self.activation3 = torch.nn.ReLU()
#        self.dropout3 = torch.nn.Dropout(0.2)

        self.linear4 = torch.nn.Linear(2048, 2048)
        self.maxpool4 = torch.nn.MaxPool1d(2)
        #self.activation3 = torch.nn.ReLU()

        self.linear5 = torch.nn.Linear(1024, 2048)
        self.activation5 = torch.nn.ReLU()
        self.dropout5 = torch.nn.Dropout(0.5)

        self.linear6 = torch.nn.Linear(2048, 2048)
        self.maxpool6 = torch.nn.MaxPool1d(2)

        self.linear7 = torch.nn.Linear(1024, 150)
        self.softmax = torch.nn.Tanh()

#        self.tlayer_gpu = torch.empty([150, 230 ], device = self.device, dtype= torch.float32)
#        self.tlayer_cpu = torch.empty([150, 230 ], device = "cpu", dtype= torch.float32)
#
#        for i in range(150):
#            for j in range(230):
#                self.tlayer_gpu[i,j] = 0.0
#                self.tlayer_cpu[i,j] = 0.0
#
#
#        for i in range(150):
#            self.tlayer_gpu[i,i] = 1
#            self.tlayer_cpu[i,i] = 1
#        for i in range(50):
#            self.tlayer_gpu[3*i  , i+150] = 1
#            self.tlayer_gpu[3*i+1, i+150] = 2
#            self.tlayer_gpu[3*i+2, i+150] = 1
#            self.tlayer_cpu[3*i  , i+150] = 1
#            self.tlayer_cpu[3*i+1, i+150] = 2
#            self.tlayer_cpu[3*i+2, i+150] = 1
#
#        for i in range(30):
#            self.tlayer_gpu[5*i  , i+200] =  3
#            self.tlayer_gpu[5*i+1, i+200] =  1
#            self.tlayer_gpu[5*i+2, i+200] =  4
#            self.tlayer_gpu[5*i+3, i+200] =  1
#            self.tlayer_gpu[5*i+4, i+200] =  3
#
#            self.tlayer_cpu[5*i,   i+200] =  3
#            self.tlayer_cpu[5*i+1, i+200] =  1
#            self.tlayer_cpu[5*i+2, i+200] =  4
#            self.tlayer_cpu[5*i+3, i+200] =  1
#            self.tlayer_cpu[5*i+4, i+200] =  3



#        self.tlayer_gpu = torch.empty([150, 296 ], device = self.device, dtype= torch.float32)
#        self.tlayer_cpu = torch.empty([150, 296 ], device = "cpu", dtype= torch.float32)
#
#        for i in range(150):
#            for j in range(296):
#                self.tlayer_gpu[i,j] = 0.0
#                self.tlayer_cpu[i,j] = 0.0
#
#
#        for i in range(150):
#            self.tlayer_gpu[i,i] = 1
#            self.tlayer_cpu[i,i] = 1
#        for i in range(2,147,1):
#            self.tlayer_gpu[ i - 2  ,  i - 1 + 150] = 1
#            self.tlayer_gpu[ i - 1  ,  i - 1 + 150] = 1
#            self.tlayer_gpu[ i      ,  i - 1 + 150] = 2
#            self.tlayer_gpu[ i + 1  ,  i - 1 + 150] = 1
#            self.tlayer_gpu[ i + 2  ,  i - 1 + 150] = 1
#
#            self.tlayer_cpu[ i - 2  ,  i - 1 + 150] = 1
#            self.tlayer_cpu[ i - 1  ,  i - 1 + 150] = 1
#            self.tlayer_cpu[ i      ,  i - 1 + 150] = 2
#            self.tlayer_cpu[ i + 1  ,  i - 1 + 150] = 1
#            self.tlayer_cpu[ i + 2  ,  i - 1 + 150] = 1


#        ## Derivative order 2:
#        self.tlayer_gpu = torch.empty([150, 298 ], device = self.device, dtype= torch.float32)
#        self.tlayer_cpu = torch.empty([150, 298 ], device = "cpu", dtype= torch.float32)
#
#        for i in range(150):
#            for j in range(298):
#                self.tlayer_gpu[i,j] = 0.0
#                self.tlayer_cpu[i,j] = 0.0
#
#
#        for i in range(150):
#            self.tlayer_gpu[i,i] = 1
#            self.tlayer_cpu[i,i] = 1
#        for i in range(1,148,1):
#            self.tlayer_gpu[ i - 1  ,  i - 1 + 150] = -10
#            self.tlayer_gpu[ i + 1  ,  i - 1 + 150] =  10
#
#            self.tlayer_cpu[ i - 1  ,  i - 1 + 150] = -10
#            self.tlayer_cpu[ i + 1  ,  i - 1 + 150] =  10
#
#        ## Derivative order 6:
#        self.tlayer_gpu = torch.empty([150, 294 ], device = self.device, dtype= torch.float32)
#        self.tlayer_cpu = torch.empty([150, 294 ], device = "cpu", dtype= torch.float32)
#
#        for i in range(150):
#            for j in range(294):
#                self.tlayer_gpu[i,j] = 0.0
#                self.tlayer_cpu[i,j] = 0.0
#
#
#        for i in range(150):
#            self.tlayer_gpu[i,i] = 1
#            self.tlayer_cpu[i,i] = 1
#        for i in range(4,146,1):
#            self.tlayer_gpu[ i - 3  ,  i - 4 + 150] = -20/60.0
#            self.tlayer_gpu[ i - 2  ,  i - 4 + 150] =  20*(3.0/20.0)
#            self.tlayer_gpu[ i - 1  ,  i - 4 + 150] = -20*(3.0/4.0)
#            self.tlayer_gpu[ i + 1  ,  i - 4 + 150] =  20*(3.0/4.0)
#            self.tlayer_gpu[ i + 2  ,  i - 4 + 150] = -20*(3.0/20.0)
#            self.tlayer_gpu[ i + 3  ,  i - 4 + 150] =  20/60.0
#
#
#            self.tlayer_cpu[ i - 3  ,  i - 4 + 150] = -20/60.0
#            self.tlayer_cpu[ i - 2  ,  i - 4 + 150] =  20*(3.0/20.0)
#            self.tlayer_cpu[ i - 1  ,  i - 4 + 150] = -20*(3.0/4.0)
#            self.tlayer_cpu[ i + 1  ,  i - 4 + 150] =  20*(3.0/4.0)
#            self.tlayer_cpu[ i + 2  ,  i - 4 + 150] = -20*(3.0/20.0)
#            self.tlayer_cpu[ i + 3  ,  i - 4 + 150] =  20/60.0
#
#        ## Derivative order 8:
#        self.tlayer_gpu = torch.empty([150, 292 ], device = self.device, dtype= torch.float32)
#        self.tlayer_cpu = torch.empty([150, 292 ], device = "cpu", dtype= torch.float32)
#
#        for i in range(150):
#            for j in range(292):
#                self.tlayer_gpu[i,j] = 0.0
#                self.tlayer_cpu[i,j] = 0.0
#
#
#        for i in range(150):
#            self.tlayer_gpu[i,i] = 1
#            self.tlayer_cpu[i,i] = 1
#        for i in range(4,146,1):
#            self.tlayer_gpu[ i - 4  ,  i - 4 + 150] =  10/280.0
#            self.tlayer_gpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
#            self.tlayer_gpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
#            self.tlayer_gpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
#            self.tlayer_gpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
#            self.tlayer_gpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
#            self.tlayer_gpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
#            self.tlayer_gpu[ i + 4  ,  i - 4 + 150] = -10/280.0
#
#
#            self.tlayer_cpu[ i - 4  ,  i - 4 + 150] =  10/280.0
#            self.tlayer_cpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
#            self.tlayer_cpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
#            self.tlayer_cpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
#            self.tlayer_cpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
#            self.tlayer_cpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
#            self.tlayer_cpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
#            self.tlayer_cpu[ i + 4  ,  i - 4 + 150] = -10/280.0


        ## Derivative order 8 and 2:
        if self.smoothed_version == 4:
            self.tlayer_gpu = torch.empty([150, 440 ], device = self.device, dtype= torch.float32)
            self.tlayer_cpu = torch.empty([150, 440 ], device = "cpu", dtype= torch.float32)

            for i in range(150):
                for j in range(440):
                    self.tlayer_gpu[i,j] = 0.0
                    self.tlayer_cpu[i,j] = 0.0


            for i in range(150):
                self.tlayer_gpu[i,i] = 1
                self.tlayer_cpu[i,i] = 1
            for i in range(4,146,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + 150] =  10/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + 150] = -10/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + 150] =  10/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + 150] = -10/280.0

            for i in range(1,148,1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 292] = -10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 292] =  10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 292] = -10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 292] =  10




        ## Derivative order 8 and 2 and second order derivative of order 4:
        if self.smoothed_version == 5:
            self.tlayer_gpu = torch.empty([150, 586 ], device = self.device, dtype= torch.float32)
            self.tlayer_cpu = torch.empty([150, 586 ], device = "cpu", dtype= torch.float32)

            for i in range(150):
                ## Only derivative
    #            for j in range(440):

                ## With second derivative
                for j in range(586):
                    self.tlayer_gpu[i,j] = 0.0
                    self.tlayer_cpu[i,j] = 0.0


            for i in range(150):
                self.tlayer_gpu[i,i] = 1
                self.tlayer_cpu[i,i] = 1
            for i in range(4,146,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + 150] =  10/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + 150] = -10/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + 150] =  10/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + 150] = -10/280.0

            for i in range(1,148,1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 292] = -10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 292] =  10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 292] = -10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 292] =  10

            for i in range(2,147,1):        # 2nd derivative order 2
                self.tlayer_gpu[ i - 2  ,  i - 2 + 440] = -10.0/12.0
                self.tlayer_gpu[ i - 1  ,  i - 2 + 440] =  10*4.0/3.0
                self.tlayer_gpu[ i      ,  i - 2 + 440] = -10*5.0/2.0
                self.tlayer_gpu[ i + 1  ,  i - 2 + 440] =  10*4.0/3.0
                self.tlayer_gpu[ i + 2  ,  i - 2 + 440] = -10.0/12.0

                self.tlayer_cpu[ i - 2  ,  i - 2 + 440] = -10.0/12.0
                self.tlayer_cpu[ i - 1  ,  i - 2 + 440] =  10*4.0/3.0
                self.tlayer_cpu[ i      ,  i - 2 + 440] = -10*5.0/2.0
                self.tlayer_cpu[ i + 1  ,  i - 2 + 440] =  10*4.0/3.0
                self.tlayer_cpu[ i + 2  ,  i - 2 + 440] = -10.0/12.0

    def forward(self, x):

        #output = torch.empty([self.batchsize, 200 ], device = self.device, dtype= torch.double)

        resized = False
        if (x.size()[0]) < 8:
            #print("Here:")
            x = x.view(1, 7)
            resized = True
            #output = torch.empty([1, 200 ], device = self.device, dtype= torch.double)

        x = x.to(torch.float32)
        ## Working net. Too small
#        x = x.to(torch.float32)
#        x = self.linear1(x)
#        x = self.activation1(x)
#        x = self.linear2(x)
#        x = self.maxpool1(x)
#        x = self.linear3(x)
#        x = self.activation3(x)
#        x = self.linear4(x)
#        x = self.maxpool2(x)
#        x = self.linear5(x)
#        x = self.softmax(x)


        x = x.to(torch.float32)
        x = self.linear1(x)
        x = self.activation1(x)
#        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.maxpool2(x)

        x = self.linear3(x)
        x = self.activation3(x)
#        x = self.dropout3(x)

        x = self.linear4(x)
        x = self.maxpool4(x)

        x = self.linear5(x)
        x = self.activation5(x)
        x = self.dropout5(x)

        x = self.linear6(x)
        x = self.maxpool6(x)

        x = self.linear7(x)
        x = self.softmax(x)


        if x.is_cuda == False:
            x = torch.matmul(x, self.tlayer_cpu)
        if x.is_cuda == True:
            x = torch.matmul(x, self.tlayer_gpu)

#        print("tlayer = ", self.tlayer_gpu)
#        print("size x = ", x.size())

        if resized == True:
            x = x[0]
        return x.double()




class SimpleModelConvSmoothed(torch.nn.Module):

## Version 1: acceptable results.
#    def __init__(self, batchsize, device, smoothed_version):
#        super(SimpleModelConvSmoothed, self).__init__()
#
#        self.batchsize = batchsize
#        self.device = device
#        self.smoothed_version = smoothed_version
#        ## M1: not too bad:
#
#        ## Layer 1:
#        # 1 input layer, 5 output layer, kernel size 2
#        self.conv1 = torch.nn.Conv1d(1, 5, 2)
#        self.relu1 = torch.nn.ReLU()
#        self.conv2 = torch.nn.Conv1d(5, 10, 3)
#        self.relu2 = torch.nn.ReLU()
#        self.conv3 = torch.nn.Conv1d(10, 40, 2)
#        self.relu3 = torch.nn.ReLU()
#        self.conv4 = torch.nn.Conv1d(40, 100, 2)
#        self.relu4 = torch.nn.ReLU()
#        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
#        self.conv5 = torch.nn.Conv1d(50, 200, 2)
#        self.relu5 = torch.nn.ReLU()
#
#        self.conv6 = torch.nn.Conv1d(32, 64, 10)
#        self.relu6 = torch.nn.ReLU()
#
#        self.conv7 = torch.nn.Conv1d(64, 128, 10)
#        self.relu7 = torch.nn.ReLU()
#
#        self.conv8 = torch.nn.Conv1d(128, 128, 10, stride = 3 )
#        self.relu8 = torch.nn.ReLU()
#
#        self.conv9 = torch.nn.Conv1d(32, 64, 10)
#        self.relu9 = torch.nn.ReLU()
#
#        self.conv10 = torch.nn.Conv1d(64, 128, 10)
#        self.relu10 = torch.nn.ReLU()
#
#        self.conv11 = torch.nn.Conv1d(128, 128, 10, stride = 3 )
#        self.relu11 = torch.nn.ReLU()
#
#        self.conv12 = torch.nn.Conv1d(32, 64,20)
#        self.relu12 = torch.nn.ReLU()
#
#        self.conv13 = torch.nn.Conv1d(64, 128, 5)
#        self.relu13 = torch.nn.ReLU()
#
#        self.conv14 = torch.nn.Conv1d(128, 128, 2, stride = 3 )
#        self.relu14 = torch.nn.ReLU()
#
#        self.linear1 = torch.nn.Linear(200, 1024)
#        self.activation1 = torch.nn.ReLU()
##        self.dropout1 = torch.nn.Dropout(0.1)
#
#        self.linear2 = torch.nn.Linear(256, 2048)
#        self.maxpool2 = torch.nn.MaxPool1d(2)
#
#        self.linear3 = torch.nn.Linear(256, 1024)
#        self.activation3 = torch.nn.ReLU()
###        self.dropout3 = torch.nn.Dropout(0.2)
##
##        self.linear4 = torch.nn.Linear(2048, 2048)
##        self.maxpool4 = torch.nn.MaxPool1d(2)
##        #self.activation3 = torch.nn.ReLU()
##
##        self.linear5 = torch.nn.Linear(1024, 2048)
##        self.activation5 = torch.nn.ReLU()
##        self.dropout5 = torch.nn.Dropout(0.5)
##
#        self.linear6 = torch.nn.Linear(384, 1024)
#        self.relu66 = torch.nn.ReLU()
#
#        self.linear7 = torch.nn.Linear(1024, 150)
#        self.softmax = torch.nn.Tanh()
#
#
#
#
#        ## Derivative order 8 and 2
#        if self.smoothed_version == 4:
#            self.tlayer_gpu = torch.empty([150, 440 ], device = self.device, dtype= torch.float32)
#            self.tlayer_cpu = torch.empty([150, 440 ], device = "cpu", dtype= torch.float32)
#
#
#            for i in range(150):
#                ## Only derivative
#    #            for j in range(440):
#
#                ## With second derivative
#                for j in range(586):
#                    self.tlayer_gpu[i,j] = 0.0
#                    self.tlayer_cpu[i,j] = 0.0
#
#
#            for i in range(150):
#                self.tlayer_gpu[i,i] = 1
#                self.tlayer_cpu[i,i] = 1
#            for i in range(4,146,1):        # Start with order 8 here
#                self.tlayer_gpu[ i - 4  ,  i - 4 + 150] =  10/280.0
#                self.tlayer_gpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
#                self.tlayer_gpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
#                self.tlayer_gpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
#                self.tlayer_gpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
#                self.tlayer_gpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
#                self.tlayer_gpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
#                self.tlayer_gpu[ i + 4  ,  i - 4 + 150] = -10/280.0
#
#
#                self.tlayer_cpu[ i - 4  ,  i - 4 + 150] =  10/280.0
#                self.tlayer_cpu[ i - 3  ,  i - 4 + 150] = -10*4/105.0
#                self.tlayer_cpu[ i - 2  ,  i - 4 + 150] =  10*(1.0/5.0)
#                self.tlayer_cpu[ i - 1  ,  i - 4 + 150] = -10*(4.0/5.0)
#                self.tlayer_cpu[ i + 1  ,  i - 4 + 150] =  10*(4.0/5.0)
#                self.tlayer_cpu[ i + 2  ,  i - 4 + 150] = -10*(1.0/5.0)
#                self.tlayer_cpu[ i + 3  ,  i - 4 + 150] =  10*(4/105.0)
#                self.tlayer_cpu[ i + 4  ,  i - 4 + 150] = -10/280.0
#
#            for i in range(1,148,1):        # now order 2
#                self.tlayer_gpu[ i - 1  ,  i - 1 + 292] = -10
#                self.tlayer_gpu[ i + 1  ,  i - 1 + 292] =  10
#
#                self.tlayer_cpu[ i - 1  ,  i - 1 + 292] = -10
#                self.tlayer_cpu[ i + 1  ,  i - 1 + 292] =  10
#
#
#
#        ## Derivative order 8 and 2 and second order derivative of order 4:
#        if self.smoothed_version == 5:
#            self.tlayer_gpu = torch.empty([150, 586 ], device = self.device, dtype= torch.float32)
#            self.tlayer_cpu = torch.empty([150, 586 ], device = "cpu", dtype= torch.float32)
#
#            for i in range(150):
#                ## Only derivative
#    #            for j in range(440):
#
#                ## With second derivative
#                for j in range(586):
#                    self.tlayer_gpu[i,j] = 0.0
#                    self.tlayer_cpu[i,j] = 0.0
#
#
#            for i in range(150):
#                self.tlayer_gpu[i,i] = 1
#                self.tlayer_cpu[i,i] = 1
#            for i in range(4,146,1):        # Start with order 8 here
#                self.tlayer_gpu[ i - 4  ,  i - 4 + 150] =  2.0/280.0
#                self.tlayer_gpu[ i - 3  ,  i - 4 + 150] = -2.0*4/105.0
#                self.tlayer_gpu[ i - 2  ,  i - 4 + 150] =  2.0*(1.0/5.0)
#                self.tlayer_gpu[ i - 1  ,  i - 4 + 150] = -2.0*(4.0/5.0)
#                self.tlayer_gpu[ i + 1  ,  i - 4 + 150] =  2.0*(4.0/5.0)
#                self.tlayer_gpu[ i + 2  ,  i - 4 + 150] = -2.0*(1.0/5.0)
#                self.tlayer_gpu[ i + 3  ,  i - 4 + 150] =  2.0*(4/105.0)
#                self.tlayer_gpu[ i + 4  ,  i - 4 + 150] = -2.0/280.0
#
#
#                self.tlayer_cpu[ i - 4  ,  i - 4 + 150] =  2.0/280.0
#                self.tlayer_cpu[ i - 3  ,  i - 4 + 150] = -2.0*4/105.0
#                self.tlayer_cpu[ i - 2  ,  i - 4 + 150] =  2.0*(1.0/5.0)
#                self.tlayer_cpu[ i - 1  ,  i - 4 + 150] = -2.0*(4.0/5.0)
#                self.tlayer_cpu[ i + 1  ,  i - 4 + 150] =  2.0*(4.0/5.0)
#                self.tlayer_cpu[ i + 2  ,  i - 4 + 150] = -2.0*(1.0/5.0)
#                self.tlayer_cpu[ i + 3  ,  i - 4 + 150] =  2.0*(4/105.0)
#                self.tlayer_cpu[ i + 4  ,  i - 4 + 150] = -2.0/280.0
#
#            for i in range(1,148,1):        # now order 2
#                self.tlayer_gpu[ i - 1  ,  i - 1 + 292] = -10
#                self.tlayer_gpu[ i + 1  ,  i - 1 + 292] =  10
#
#                self.tlayer_cpu[ i - 1  ,  i - 1 + 292] = -10
#                self.tlayer_cpu[ i + 1  ,  i - 1 + 292] =  10
#
#            for i in range(2,147,1):        # 2nd derivative order 2
#                self.tlayer_gpu[ i - 2  ,  i - 2 + 440] = -1.0/12.0
#                self.tlayer_gpu[ i - 1  ,  i - 2 + 440] =  1.0*4.0/3.0
#                self.tlayer_gpu[ i      ,  i - 2 + 440] = -1.0*5.0/2.0
#                self.tlayer_gpu[ i + 1  ,  i - 2 + 440] =  1.0*4.0/3.0
#                self.tlayer_gpu[ i + 2  ,  i - 2 + 440] = -1.0/12.0
#
#                self.tlayer_cpu[ i - 2  ,  i - 2 + 440] = -1.0/12.0
#                self.tlayer_cpu[ i - 1  ,  i - 2 + 440] =  1.0*4.0/3.0
#                self.tlayer_cpu[ i      ,  i - 2 + 440] = -1.0*5.0/2.0
#                self.tlayer_cpu[ i + 1  ,  i - 2 + 440] =  1.0*4.0/3.0
#                self.tlayer_cpu[ i + 2  ,  i - 2 + 440] = -1.0/12.0
#
#
#
#    def forward(self, x):
#
#        #output = torch.empty([self.batchsize, 200 ], device = self.device, dtype= torch.double)
#
#        resized = False
##        x = x.view(self.batchsize, 7, 1)
##        x = x.transpose(1, 2).contiguous()
#        if (x.size()[0]) < 8:
#            #print("Here:")
#            x = x.view(1, 7, 1)
#            x = x.transpose(1, 2).contiguous()
#            resized = True
#        else:
#            x = x.view(self.batchsize, 7, 1)
#            x = x.transpose(1, 2).contiguous()
#            resized = False
#
#        x = x.to(torch.float32)
#
##        print(x.size())
#        #print("x = ", x )
#
#        x = self.conv1(x)
#        x = self.relu1(x)
##        print(x.size())
#        #print("x = ", x )
#        x = self.conv2(x)
#        x = self.relu2(x)
##
#        x = self.conv3(x)
#        x = self.relu3(x)
##
##        print(x.size())
#
#        x = self.conv4(x)
#        x = self.relu4(x)
##
##        print(x.size())
#        x = x.transpose(1, 2).contiguous()
#        x = self.maxpool(x)
##        print(x.size())
#        x = x.transpose(1, 2).contiguous()
#        x = self.conv5(x)
#        x = self.relu5(x)
##
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 200)
#        else:
#            x = x.view(1, 200)
#
#
#        x = self.linear1(x)
#        x = self.activation1(x)
#
#        if resized == False:
#            x = x.view(self.batchsize, 32, 32)
#        else:
#            x = x.view(1, 32, 32)
#
#        x = self.conv6(x)
#        x = self.relu6(x)
#
#        x = self.conv7(x)
#        x = self.relu7(x)
#
#        x = self.conv8(x)
#        x = self.relu8(x)
#
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 256)
#        else:
#            x = x.view(1, 256)
#
#
##        x = self.dropout1(x)
#
#        x = self.linear2(x)
#        x = self.maxpool2(x)
#
#        if resized == False:
#            x = x.view(self.batchsize, 32, 32)
#        else:
#            x = x.view(1, 32, 32)
#
#
#
#        x = self.conv9(x)
#        x = self.relu9(x)
#
#        x = self.conv10(x)
#        x = self.relu10(x)
#
#        x = self.conv11(x)
#        x = self.relu11(x)
#
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 256)
#        else:
#            x = x.view(1, 256)
#
#
#        x = self.linear3(x)
#        x = self.activation3(x)
#
#        if resized == False:
#            x = x.view(self.batchsize, 32, 32)
#        else:
#            x = x.view(1, 32, 32)
#
#
#
#        x = self.conv12(x)
#        x = self.relu12(x)
#
##        print(x.size())
#        x = self.conv13(x)
#        x = self.relu13(x)
#
##        print(x.size())
#        x = self.conv14(x)
#        x = self.relu14(x)
##        print(x.size())
#
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 384)
#        else:
#            x = x.view(1, 384)
#
#
##        x = self.dropout1(x)
#
#
#
##
##        x = self.linear3(x)
##        x = self.activation3(x)
###        x = self.dropout3(x)
##
##        x = self.linear4(x)
##        x = self.maxpool4(x)
###
##        x = self.linear5(x)
##        x = self.activation5(x)
###        x = self.dropout5(x)
##
##        x = self.linear6(x)
##        x = self.maxpool6(x)
#
#        x = self.linear6(x)
#        x = self.relu66(x)
#
#        x = self.linear7(x)
#        x = self.softmax(x)
#
#
#        if x.is_cuda == False:
#            x = torch.matmul(x, self.tlayer_cpu)
#        if x.is_cuda == True:
#            x = torch.matmul(x, self.tlayer_gpu)
#
##        print("tlayer = ", self.tlayer_gpu)
##        print("size x = ", x.size())
#
#        if resized == True:
#            x = x[0]
#        return x.double()
#

## Version 2
    def __init__(self, batchsize, device, smoothed_version, low_energy_index):
        super(SimpleModelConvSmoothed, self).__init__()

        self.batchsize = batchsize
        self.device = device
        self.smoothed_version = smoothed_version
        self.low_energy_index = low_energy_index
        ## M1: not too bad:

        ## Layer 1:
        # 1 input layer, 5 output layer, kernel size 2
        self.conv1 = torch.nn.Conv1d(1, 5, 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(5, 10, 3)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv1d(10, 20, 5)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv1d(20, 30, 25)
        self.relu4 = torch.nn.ReLU()
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.conv5 = torch.nn.Conv1d(30, 40, 50)
        self.relu5 = torch.nn.ReLU()

        self.maxpool = torch.nn.MaxPool1d(kernel_size=8)
#
        self.conv6 = torch.nn.Conv1d(40, 50, 10)
        self.relu6 = torch.nn.ReLU()

#        self.batch3 = torch.nn.BatchNorm1d(20)
#        self.batch4 = torch.nn.BatchNorm1d(30)
#        self.batch5 = torch.nn.BatchNorm1d(40, affine = False)
#        self.batch6 = torch.nn.BatchNorm1d(50, affine = False)

#        self.conv7 = torch.nn.Conv1d(200, 200, 10)
#        self.relu7 = torch.nn.ReLU()

#        self.maxpool = torch.nn.MaxPool1d(kernel_size=16)
#        self.conv8 = torch.nn.Conv1d(128, 128, 10, stride = 3 )
#        self.relu8 = torch.nn.ReLU()
#
#        self.conv9 = torch.nn.Conv1d(32, 64, 10)
#        self.relu9 = torch.nn.ReLU()
#
#        self.conv10 = torch.nn.Conv1d(64, 128, 10)
#        self.relu10 = torch.nn.ReLU()
#
#        self.conv11 = torch.nn.Conv1d(128, 128, 10, stride = 3 )
#        self.relu11 = torch.nn.ReLU()
#
#        self.conv12 = torch.nn.Conv1d(32, 64,20)
#        self.relu12 = torch.nn.ReLU()
#
#        self.conv13 = torch.nn.Conv1d(64, 128, 5)
#        self.relu13 = torch.nn.ReLU()
#
#        self.conv14 = torch.nn.Conv1d(128, 128, 2, stride = 3 )
#        self.relu14 = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(7, 1024)
        self.activation1 = torch.nn.ReLU()

#        self.linear2 = torch.nn.Linear(256, 2048)
#        self.maxpool2 = torch.nn.MaxPool1d(2)

#        self.linear3 = torch.nn.Linear(256, 1024)
#        self.activation3 = torch.nn.ReLU()
##        self.dropout3 = torch.nn.Dropout(0.2)
#
#        self.linear4 = torch.nn.Linear(2048, 2048)
#        self.maxpool4 = torch.nn.MaxPool1d(2)
#        #self.activation3 = torch.nn.ReLU()
#
#        self.linear5 = torch.nn.Linear(1024, 2048)
#        self.activation5 = torch.nn.ReLU()
#        self.dropout5 = torch.nn.Dropout(0.5)
#
#        self.linear6 = torch.nn.Linear(384, 1024)
#        self.relu66 = torch.nn.ReLU()

        self.linear7 = torch.nn.Linear(50*109, 150 - self.low_energy_index)
        self.softmax = torch.nn.Tanh()




        ## Derivative order 8 and 2
        if self.smoothed_version == 4:
            self.tlayer_gpu = torch.empty([150 - self.low_energy_index, 440 - 2*self.low_energy_index ], device = self.device, dtype= torch.float32)
            self.tlayer_cpu = torch.empty([150 - self.low_energy_index, 440 - 2*self.low_energy_index ], device = "cpu", dtype= torch.float32)


            for i in range(150 - self.low_energy_index):
                ## Only derivative
    #            for j in range(440):

                ## With second derivative
                for j in range(440 - 2*self.low_energy_index):
                    self.tlayer_gpu[i,j] = 0.0
                    self.tlayer_cpu[i,j] = 0.0


            for i in range(150-self.low_energy_index):
                self.tlayer_gpu[i,i] = 1
                self.tlayer_cpu[i,i] = 1
            for i in range(4,146 - self.low_energy_index,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + 150- self.low_energy_index] =  10/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + 150- self.low_energy_index] = -10*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + 150- self.low_energy_index] =  10*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + 150- self.low_energy_index] = -10*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + 150- self.low_energy_index] =  10*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + 150- self.low_energy_index] = -10*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + 150- self.low_energy_index] =  10*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + 150- self.low_energy_index] = -10/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + 150- self.low_energy_index] =  10/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + 150- self.low_energy_index] = -10*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + 150- self.low_energy_index] =  10*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + 150- self.low_energy_index] = -10*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + 150- self.low_energy_index] =  10*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + 150- self.low_energy_index] = -10*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + 150- self.low_energy_index] =  10*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + 150- self.low_energy_index] = -10/280.0

            for i in range(1,148- self.low_energy_index,1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 292 - 2*self.low_energy_index] = -10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 292 - 2*self.low_energy_index] =  10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 292 - 2*self.low_energy_index] = -10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 292 - 2*self.low_energy_index] =  10



        ## Derivative order 8 and 2 and second order derivative of order 4:
        if self.smoothed_version == 5:
            self.tlayer_gpu = torch.empty([150 - self.low_energy_index, 586 - 3*self.low_energy_index ], device = self.device, dtype= torch.float32)
            self.tlayer_cpu = torch.empty([150 - self.low_energy_index, 586 - 3*self.low_energy_index], device = "cpu", dtype= torch.float32)

            for i in range(150 - self.low_energy_index):
                ## Only derivative
    #            for j in range(440):

                ## With second derivative
                for j in range(586 - 3*self.low_energy_index):
                    self.tlayer_gpu[i,j] = 0.0
                    self.tlayer_cpu[i,j] = 0.0


            for i in range(150 - self.low_energy_index):
                self.tlayer_gpu[i,i] = 1
                self.tlayer_cpu[i,i] = 1
            for i in range(4,146 - self.low_energy_index,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + 150 - self.low_energy_index] =  2.0/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + 150 - self.low_energy_index] = -2.0*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + 150 - self.low_energy_index] =  2.0*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + 150 - self.low_energy_index] = -2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + 150 - self.low_energy_index] =  2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + 150 - self.low_energy_index] = -2.0*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + 150 - self.low_energy_index] =  2.0*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + 150 - self.low_energy_index] = -2.0/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + 150 - self.low_energy_index] =  2.0/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + 150 - self.low_energy_index] = -2.0*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + 150 - self.low_energy_index] =  2.0*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + 150 - self.low_energy_index] = -2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + 150 - self.low_energy_index] =  2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + 150 - self.low_energy_index] = -2.0*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + 150 - self.low_energy_index] =  2.0*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + 150 - self.low_energy_index] = -2.0/280.0

            for i in range(1,148 - self.low_energy_index,1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 292 - 2*self.low_energy_index] = -10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 292 - 2*self.low_energy_index] =  10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 292 - 2*self.low_energy_index] = -10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 292 - 2*self.low_energy_index] =  10

            for i in range(2,147 - self.low_energy_index,1):        # 2nd derivative order 2
                self.tlayer_gpu[ i - 2  ,  i - 2 + 440 - 3*self.low_energy_index] = -1.0/12.0
                self.tlayer_gpu[ i - 1  ,  i - 2 + 440 - 3*self.low_energy_index] =  1.0*4.0/3.0
                self.tlayer_gpu[ i      ,  i - 2 + 440 - 3*self.low_energy_index] = -1.0*5.0/2.0
                self.tlayer_gpu[ i + 1  ,  i - 2 + 440 - 3*self.low_energy_index] =  1.0*4.0/3.0
                self.tlayer_gpu[ i + 2  ,  i - 2 + 440 - 3*self.low_energy_index] = -1.0/12.0

                self.tlayer_cpu[ i - 2  ,  i - 2 + 440 - 3*self.low_energy_index] = -1.0/12.0
                self.tlayer_cpu[ i - 1  ,  i - 2 + 440 - 3*self.low_energy_index] =  1.0*4.0/3.0
                self.tlayer_cpu[ i      ,  i - 2 + 440 - 3*self.low_energy_index] = -1.0*5.0/2.0
                self.tlayer_cpu[ i + 1  ,  i - 2 + 440 - 3*self.low_energy_index] =  1.0*4.0/3.0
                self.tlayer_cpu[ i + 2  ,  i - 2 + 440 - 3*self.low_energy_index] = -1.0/12.0



    def forward(self, x):

        #output = torch.empty([self.batchsize, 200 ], device = self.device, dtype= torch.double)

        resized = False

        x = x.to(torch.float32)
        if (x.size()[0]) < 8:
            resized = True
        else:
            resized = False

        x = self.linear1(x)
        x = self.activation1(x)


#        print("After linear 1:", x.size(), x.size()[0])
        if resized == True:
            #print("Here:")
            x = x.view(1, 1024, 1)
            x = x.transpose(1, 2).contiguous()
        else:
            x = x.view(self.batchsize, 1024, 1)
            x = x.transpose(1, 2).contiguous()


        x = self.conv1(x)
        x = self.relu1(x)
#        print("After conv 1:",  x.size())
        #print("x = ", x )
        x = self.conv2(x)
        x = self.relu2(x)
#        print("After conv 2:",  x.size())
#
        x = self.conv3(x)
        x = self.relu3(x)
#        print("After conv 3:",  x.size())
#        x = self.batch3(x)

#        print(x.size())

        x = self.conv4(x)
        x = self.relu4(x)
#        x = self.batch4(x)
#        print("After conv 4:",  x.size())


#        print(x.size())
#        x = x.transpose(1, 2).contiguous()
#        x = self.maxpool(x)
##        print(x.size())
#        x = x.transpose(1, 2).contiguous()
        x = self.conv5(x)
        x = self.relu5(x)

#        print("After conv 5:",  x.size())

        x = self.maxpool(x)
#        x = self.batch5(x)
#        print("After maxpool:",  x.size())


#        print(x.size())
        x = self.conv6(x)
        x = self.relu6(x)
#        print("After conv 6:",  x.size())

#        x = self.batch6(x)

#        print(x.size())
#        x = self.maxpool(x)
#        print(x.size())

        if resized == False:
            x = x.view(self.batchsize, 50*109)
        else:
            x = x.view(1, 50*109)

##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 400, 1)
#            x = x.transpose(1, 2).contiguous()
#        else:
#            x = x.view(1, 400, 1)
#            x = x.transpose(1, 2).contiguous()
#
#
##        x = self.linear1(x)
##        x = self.activation1(x)
#
##        if resized == False:
##            x = x.view(self.batchsize, 400, 1)
##        else:
##            x = x.view(1, 400, 1)
#
##        print(x.size())
#        x = self.conv6(x)
#        x = self.relu6(x)
#
##        print(x.size())
#        x = self.conv7(x)
#        x = self.relu7(x)
#
#        x = self.maxpool(x)
#
##        print(x.size())
#
#        if resized == False:
#            x = x.view(self.batchsize, 200*23)
#        else:
#            x = x.view(1, 200*23)
#

#        x = self.conv8(x)
#        x = self.relu8(x)
#
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 256)
#        else:
#            x = x.view(1, 256)
#
#
##        x = self.dropout1(x)
#
#        x = self.linear2(x)
#        x = self.maxpool2(x)
#
#        if resized == False:
#            x = x.view(self.batchsize, 32, 32)
#        else:
#            x = x.view(1, 32, 32)
#
#
#
#        x = self.conv9(x)
#        x = self.relu9(x)
#
#        x = self.conv10(x)
#        x = self.relu10(x)
#
#        x = self.conv11(x)
#        x = self.relu11(x)
#
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 256)
#        else:
#            x = x.view(1, 256)
#
#
#        x = self.linear3(x)
#        x = self.activation3(x)
#
#        if resized == False:
#            x = x.view(self.batchsize, 32, 32)
#        else:
#            x = x.view(1, 32, 32)
#
#
#
#        x = self.conv12(x)
#        x = self.relu12(x)
#
##        print(x.size())
#        x = self.conv13(x)
#        x = self.relu13(x)
#
##        print(x.size())
#        x = self.conv14(x)
#        x = self.relu14(x)
##        print(x.size())
#
##        print(x.size())
##        print("x = ", x )
#        if resized == False:
#            x = x.view(self.batchsize, 384)
#        else:
#            x = x.view(1, 384)
#
#
##        x = self.dropout1(x)
#
#
#
##
##        x = self.linear3(x)
##        x = self.activation3(x)
###        x = self.dropout3(x)
##
##        x = self.linear4(x)
##        x = self.maxpool4(x)
###
##        x = self.linear5(x)
##        x = self.activation5(x)
###        x = self.dropout5(x)
##
##        x = self.linear6(x)
##        x = self.maxpool6(x)
#
#        x = self.linear6(x)
#        x = self.relu66(x)
#
        x = self.linear7(x)
        x = self.softmax(x)


        if x.is_cuda == False:
            x = torch.matmul(x, self.tlayer_cpu)
        if x.is_cuda == True:
            x = torch.matmul(x, self.tlayer_gpu)

#        print("tlayer = ", self.tlayer_gpu)
#        print("size x = ", x.size())

        if resized == True:
            x = x[0]
        return x.double()



    def update_derivative_matrix(self, coeff_der_2, coeff_der_8, coeff_der_2_4):



        if self.smoothed_version == 5:

            for i in range(4,146 - self.low_energy_index,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + 150- self.low_energy_index] =  coeff_der_8*2.0*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + 150- self.low_energy_index] = -coeff_der_8*2.0/280.0

            for i in range(1,148- self.low_energy_index,1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 292- 2*self.low_energy_index] = -coeff_der_2*10.0
                self.tlayer_gpu[ i + 1  ,  i - 1 + 292- 2*self.low_energy_index] =  coeff_der_2*10.0

                self.tlayer_cpu[ i - 1  ,  i - 1 + 292- 2*self.low_energy_index] = -coeff_der_2*10.0
                self.tlayer_cpu[ i + 1  ,  i - 1 + 292- 2*self.low_energy_index] =  coeff_der_2*10.0

            for i in range(2,147- self.low_energy_index,1):        # 2nd derivative order 2
                self.tlayer_gpu[ i - 2  ,  i - 2 + 440- 3*self.low_energy_index] = -coeff_der_2_4*1.0/12.0
                self.tlayer_gpu[ i - 1  ,  i - 2 + 440- 3*self.low_energy_index] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_gpu[ i      ,  i - 2 + 440- 3*self.low_energy_index] = -coeff_der_2_4*1.0*5.0/2.0
                self.tlayer_gpu[ i + 1  ,  i - 2 + 440- 3*self.low_energy_index] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_gpu[ i + 2  ,  i - 2 + 440- 3*self.low_energy_index] = -coeff_der_2_4*1.0/12.0

                self.tlayer_cpu[ i - 2  ,  i - 2 + 440- 3*self.low_energy_index] = -coeff_der_2_4*1.0/12.0
                self.tlayer_cpu[ i - 1  ,  i - 2 + 440- 3*self.low_energy_index] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_cpu[ i      ,  i - 2 + 440- 3*self.low_energy_index] = -coeff_der_2_4*1.0*5.0/2.0
                self.tlayer_cpu[ i + 1  ,  i - 2 + 440- 3*self.low_energy_index] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_cpu[ i + 2  ,  i - 2 + 440- 3*self.low_energy_index] = -coeff_der_2_4*1.0/12.0



class LegendreModel(torch.nn.Module):

    def __init__(self, order, batchsize):
        super(LegendreModel, self).__init__()


        self.batchsize = batchsize
        ## M1: not too bad:
        self.linear1 = torch.nn.Linear(7, 2048)
        self.activation = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(2)
        self.linear2 = torch.nn.Linear(1024, 2048)
        #self.Dropout1 =  torch.nn.Dropout(0.2)
        self.activation2 = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(2)
        self.linear3 = torch.nn.Linear(1024, 1024)
        self.activation3 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear4 = torch.nn.Linear(1024, 1024)
        self.activation4 = torch.nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool1d(8)
        self.linear5 = torch.nn.Linear(128, order)

        self.softmax = torch.nn.Tanh()

    def forward(self, x):

        x = x.to(torch.float32)
        resized = False
        if len(x.size()) < 2:
            x = x.view(1, 7)
            resized = True
#        print(x.size())
        x = self.linear1(x)

        #x = self.activation(x)


        x = self.maxpool2(x)

        x = self.linear2(x)

        #x = self.activation2(x)

        #print(x.size())
        #x = x.reshape(self.batchsize, 32,32)
        x = self.maxpool(x)
        #x = x.reshape(self.batchsize,256)
#        x = self.Dropout1(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.maxpool3(x)
        x = self.linear5(x)
        x = self.softmax(x)
        if resized == True:
            x = x[0]
        return x.double()



class LegendreSplitModel(torch.nn.Module):

    def __init__(self, order, batchsize, device):
        super(LegendreSplitModel, self).__init__()


        self.batchsize = batchsize
        self.order = order
        self.device = device
        ## M1: not too bad:

        sizeM = 128
        # 0 :
        self.linear10 = torch.nn.Linear(7, sizeM)
        #self.activation10 = torch.nn.ReLU()
        self.activation10 = torch.nn.Hardshrink()
        self.linear20 = torch.nn.Linear(sizeM, sizeM)
        self.activation20 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear30 = torch.nn.Linear(sizeM, 1)
        self.softmax0 = torch.nn.Tanh()

        # 1 :
        self.linear11 = torch.nn.Linear(7, sizeM)
#        self.activation11 = torch.nn.ReLU()
        self.activation11 = torch.nn.Hardshrink()
        self.linear21 = torch.nn.Linear(sizeM, sizeM)
        self.activation21 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear31 = torch.nn.Linear(sizeM, 1)
        self.softmax1 = torch.nn.Tanh()

        # 2 :
        self.linear12 = torch.nn.Linear(7, sizeM)
#        self.activation12 = torch.nn.ReLU()
        self.activation12 = torch.nn.Hardshrink()
        self.linear22 = torch.nn.Linear(sizeM, sizeM)
        self.activation22 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear32 = torch.nn.Linear(sizeM, 1)
        self.softmax2 = torch.nn.Tanh()

        # 3 :
        self.linear13 = torch.nn.Linear(7, sizeM)
#        self.activation13 = torch.nn.ReLU()
        self.activation13 = torch.nn.Hardshrink()
        self.linear23 = torch.nn.Linear(sizeM, sizeM)
        self.activation23 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear33 = torch.nn.Linear(sizeM, 1)
        self.softmax3 = torch.nn.Tanh()

        # 4 :
        self.linear14 = torch.nn.Linear(7, sizeM)
#        self.activation14 = torch.nn.ReLU()
        self.activation14 = torch.nn.Hardshrink()
        self.linear24 = torch.nn.Linear(sizeM, sizeM)
        self.activation24 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear34 = torch.nn.Linear(sizeM, 1)
        self.softmax4 = torch.nn.Tanh()

        # 5 :
        self.linear15 = torch.nn.Linear(7, sizeM)
#        self.activation15 = torch.nn.ReLU()
        self.activation15 = torch.nn.Hardshrink()
        self.linear25 = torch.nn.Linear(sizeM, sizeM)
        self.activation25 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear35 = torch.nn.Linear(sizeM, 1)
        self.softmax5 = torch.nn.Tanh()

        # 6 :
        self.linear16 = torch.nn.Linear(7, sizeM)
#        self.activation16 = torch.nn.ReLU()
        self.activation16 = torch.nn.Hardshrink()
        self.linear26 = torch.nn.Linear(sizeM, sizeM)
        self.activation26 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear36 = torch.nn.Linear(sizeM, 1)
        self.softmax6 = torch.nn.Tanh()


        # 7 :
        self.linear17 = torch.nn.Linear(7, sizeM)
#        self.activation17 = torch.nn.ReLU()
        self.activation17 = torch.nn.Hardshrink()
        self.linear27 = torch.nn.Linear(sizeM, sizeM)
        self.activation27 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear37 = torch.nn.Linear(sizeM, 1)
        self.softmax7 = torch.nn.Tanh()


        # 8 :
        self.linear18 = torch.nn.Linear(7, sizeM)
#        self.activation18 = torch.nn.ReLU()
        self.activation18 = torch.nn.Hardshrink()
        self.linear28 = torch.nn.Linear(sizeM, sizeM)
        self.activation28 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear38 = torch.nn.Linear(sizeM, 1)
        self.softmax8 = torch.nn.Tanh()


        # 9 :
        self.linear19 = torch.nn.Linear(7, sizeM)
#        self.activation19 = torch.nn.ReLU()
        self.activation19 = torch.nn.Hardshrink()
        self.linear29 = torch.nn.Linear(sizeM, sizeM)
        self.activation29 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear39 = torch.nn.Linear(sizeM, 1)
        self.softmax9 = torch.nn.Tanh()

        # 10 :
        self.linear110 = torch.nn.Linear(7, sizeM)
#        self.activation110 = torch.nn.ReLU()
        self.activation110 = torch.nn.Hardshrink()
        self.linear210 = torch.nn.Linear(sizeM, sizeM)
        self.activation210 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear310 = torch.nn.Linear(sizeM, 1)
        self.softmax10 = torch.nn.Tanh()

        # 11 :
        self.linear111 = torch.nn.Linear(7, sizeM)
#        self.activation111 = torch.nn.ReLU()
        self.activation111 = torch.nn.Hardshrink()
        self.linear211 = torch.nn.Linear(sizeM, sizeM)
        self.activation211 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear311 = torch.nn.Linear(sizeM, 1)
        self.softmax11 = torch.nn.Tanh()

        # 12 :
        self.linear112 = torch.nn.Linear(7, sizeM)
#        self.activation112 = torch.nn.ReLU()
        self.activation112 = torch.nn.Hardshrink()
        self.linear212 = torch.nn.Linear(sizeM, sizeM)
        self.activation212 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear312 = torch.nn.Linear(sizeM, 1)
        self.softmax12 = torch.nn.Tanh()


        # 13 :
        self.linear113 = torch.nn.Linear(7, sizeM)
#        self.activation113 = torch.nn.ReLU()
        self.activation113 = torch.nn.Hardshrink()
        self.linear213 = torch.nn.Linear(sizeM, sizeM)
        self.activation213 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear313 = torch.nn.Linear(sizeM, 1)
        self.softmax13 = torch.nn.Tanh()

        # 14 :
        self.linear114 = torch.nn.Linear(7, sizeM)
#        self.activation114 = torch.nn.ReLU()
        self.activation114 = torch.nn.Hardshrink()
        self.linear214 = torch.nn.Linear(sizeM, sizeM)
        self.activation214 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear314 = torch.nn.Linear(sizeM, 1)
        self.softmax14 = torch.nn.Tanh()

        # 15 :
        self.linear115 = torch.nn.Linear(7, sizeM)
#        self.activation115 = torch.nn.ReLU()
        self.activation115 = torch.nn.Hardshrink()
        self.linear215 = torch.nn.Linear(sizeM, sizeM)
        self.activation215 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear315 = torch.nn.Linear(sizeM, 1)
        self.softmax15 = torch.nn.Tanh()

        # 16 :
        self.linear116 = torch.nn.Linear(7, sizeM)
#        self.activation116 = torch.nn.ReLU()
        self.activation116 = torch.nn.Hardshrink()
        self.linear216 = torch.nn.Linear(sizeM, sizeM)
        self.activation216 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear316 = torch.nn.Linear(sizeM, 1)
        self.softmax16 = torch.nn.Tanh()

        # 17 :
        self.linear117 = torch.nn.Linear(7, sizeM)
#        self.activation117 = torch.nn.ReLU()
        self.activation117 = torch.nn.Hardshrink()
        self.linear217 = torch.nn.Linear(sizeM, sizeM)
        self.activation217 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear317 = torch.nn.Linear(sizeM, 1)
        self.softmax17 = torch.nn.Tanh()

        # 18 :
        self.linear118 = torch.nn.Linear(7, sizeM)
#        self.activation118 = torch.nn.ReLU()
        self.activation118 = torch.nn.Hardshrink()
        self.linear218 = torch.nn.Linear(sizeM, sizeM)
        self.activation218 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear318 = torch.nn.Linear(sizeM, 1)
        self.softmax18 = torch.nn.Tanh()

        # 19 :
        self.linear119 = torch.nn.Linear(7, sizeM)
#        self.activation119 = torch.nn.ReLU()
        self.activation119 = torch.nn.Hardshrink()
        self.linear219 = torch.nn.Linear(sizeM, sizeM)
        self.activation219 = torch.nn.LeakyReLU(negative_slope = 0.5)
        self.linear319 = torch.nn.Linear(sizeM, 1)
        self.softmax19 = torch.nn.Tanh()



    def forward(self, x):

        x = x.to(torch.float32)
        resized = False
        if (x.size()[0]) < 8:
            #print("Here:")
            x = x.view(1, 7)
            resized = True
        #print("x = ", x, resized, x.size()[0]  )

        output = torch.empty([self.batchsize, self.order ], device = self.device, dtype= torch.double)
        if(resized == True):
            output = torch.empty([1, self.order ], device = self.device, dtype= torch.double)


        # 0
        y= self.linear10(x)
        y = self.activation10(y)
        y= self.linear20(y)
        y = self.activation20(y)
        y= self.linear30(y)
        y = self.softmax0(y)

        if resized == False:
            for i in range(self.batchsize):
                output[i][0] = y[i]
        else:
            output[0][0] = y[0]

        # 1
        y= self.linear11(x)
        y = self.activation11(y)
        y= self.linear21(y)
        y = self.activation21(y)
        y= self.linear31(y)
        y = self.softmax1(y)

        if resized == False:
            for i in range(self.batchsize):
                output[i][1] = y[i]
        else:
            output[0][1] = y[0]


        # 2
        y= self.linear12(x)
        y = self.activation12(y)
        y= self.linear22(y)
        y = self.activation22(y)
        y= self.linear32(y)
        y = self.softmax2(y)

        if resized == False:
            for i in range(self.batchsize):
                output[i][2] = y[i]
        else:
            output[0][2] = y[0]


        # 3
        y= self.linear13(x)
        y = self.activation13(y)
        y= self.linear23(y)
        y = self.activation23(y)
        y= self.linear33(y)
        y = self.softmax3(y)

        if resized == False:
            for i in range(self.batchsize):
                output[i][3] = y[i]
        else:
            output[0][3] = y[0]

        # 4
        y= self.linear14(x)
        y = self.activation14(y)
        y= self.linear24(y)
        y = self.activation24(y)
        y= self.linear34(y)
        y = self.softmax4(y)

        if resized == False:
            for i in range(self.batchsize):
                output[i][4] = y[i]
        else:
            output[0][4] = y[0]

        # 5
        y= self.linear15(x)
        y = self.activation15(y)
        y= self.linear25(y)
        y = self.activation25(y)
        y= self.linear35(y)
        y = self.softmax5(y)

        if resized == False:
            for i in range(self.batchsize):
                output[i][5] = y[i]
        else:
            output[0][5] = y[0]


        # 6
        if self.order > 6:

            y= self.linear16(x)
            y = self.activation16(y)
            y= self.linear26(y)
            y = self.activation26(y)
            y= self.linear36(y)
            y = self.softmax6(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][6] = y[i]
            else:
                output[0][6] = y[0]

        # 7
        if self.order > 7:

            y= self.linear17(x)
            y = self.activation17(y)
            y= self.linear27(y)
            y = self.activation27(y)
            y= self.linear37(y)
            y = self.softmax7(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][7] = y[i]
            else:
                output[0][7] = y[0]

        # 8
        if self.order > 8:

            y= self.linear18(x)
            y = self.activation18(y)
            y= self.linear28(y)
            y = self.activation28(y)
            y= self.linear38(y)
            y = self.softmax8(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][8] = y[i]
            else:
                output[0][8] = y[0]

        # 9
        if self.order > 9:

            y= self.linear19(x)
            y = self.activation19(y)
            y= self.linear29(y)
            y = self.activation29(y)
            y= self.linear39(y)
            y = self.softmax9(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][9] = y[i]
            else:
                output[0][9] = y[0]

        # 10
        if self.order > 10:

            y= self.linear110(x)
            y = self.activation110(y)
            y= self.linear210(y)
            y = self.activation210(y)
            y= self.linear310(y)
            y = self.softmax10(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][10] = y[i]
            else:
                output[0][10] = y[0]

        # 11
        if self.order > 11:

            y= self.linear111(x)
            y = self.activation111(y)
            y= self.linear211(y)
            y = self.activation211(y)
            y= self.linear311(y)
            y = self.softmax11(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][11] = y[i]
            else:
                output[0][11] = y[0]

        # 12
        if self.order > 12:

            y= self.linear112(x)
            y = self.activation112(y)
            y= self.linear212(y)
            y = self.activation212(y)
            y= self.linear312(y)
            y = self.softmax12(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][12] = y[i]
            else:
                output[0][12] = y[0]

        # 13
        if self.order > 13:

            y= self.linear113(x)
            y = self.activation113(y)
            y= self.linear213(y)
            y = self.activation213(y)
            y= self.linear313(y)
            y = self.softmax13(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][13] = y[i]
            else:
                output[0][13] = y[0]

        # 14
        if self.order > 14:

            y= self.linear114(x)
            y = self.activation114(y)
            y= self.linear214(y)
            y = self.activation214(y)
            y= self.linear314(y)
            y = self.softmax14(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][14] = y[i]
            else:
                output[0][14] = y[0]

        # 15
        if self.order > 15:

            y= self.linear115(x)
            y = self.activation115(y)
            y= self.linear215(y)
            y = self.activation215(y)
            y= self.linear315(y)
            y = self.softmax15(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][15] = y[i]
            else:
                output[0][15] = y[0]


        # 16
        if self.order > 16:

            y= self.linear116(x)
            y = self.activation116(y)
            y= self.linear216(y)
            y = self.activation216(y)
            y= self.linear316(y)
            y = self.softmax16(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][16] = y[i]
            else:
                output[0][16] = y[0]

        # 17
        if self.order > 17:

            y= self.linear117(x)
            y = self.activation117(y)
            y= self.linear217(y)
            y = self.activation217(y)
            y= self.linear317(y)
            y = self.softmax17(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][17] = y[i]
            else:
                output[0][17] = y[0]


        # 18
        if self.order > 18:

            y= self.linear118(x)
            y = self.activation118(y)
            y= self.linear218(y)
            y = self.activation218(y)
            y= self.linear318(y)
            y = self.softmax18(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][18] = y[i]
            else:
                output[0][18] = y[0]

        # 19
        if self.order > 19:

            y= self.linear119(x)
            y = self.activation119(y)
            y= self.linear219(y)
            y = self.activation219(y)
            y= self.linear319(y)
            y = self.softmax19(y)

            if resized == False:
                for i in range(self.batchsize):
                    output[i][19] = y[i]
            else:
                output[0][19] = y[0]


        if resized == True:
            #print("Resized is TRUE")
            output = output[0]
        #print("output = ", output)
#        output = output.to(torch.float32)
        return output
