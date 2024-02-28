import torch
from torch.utils.data import Dataset


class SopranoElectronDataset(Dataset):
    def __init__(self, params, spectrum):
        self.param_in = params
        self.electron_spectrum = spectrum

    def __len__(self):
        return len(self.electron_spectrum)

    def __getitem__(self, idx):

        return torch.tensor(self.param_in[idx]) , torch.tensor(self.electron_spectrum[idx])


class ElectronModelConvSmoothed(torch.nn.Module):

    def __init__(self, batchsize, device, smoothed_version ):
        super(ElectronModelConvSmoothed, self).__init__()

        self.batchsize = batchsize
        self.device = device
        self.smoothed_version = smoothed_version
        self.dim = 130
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


        self.linear1 = torch.nn.Linear(7, 1024)
        self.activation1 = torch.nn.ReLU()


        self.linear7 = torch.nn.Linear(50*109, self.dim)
        self.softmax = torch.nn.Tanh()




        ## Derivative order 8 and 2
        if self.smoothed_version == 4:
            self.tlayer_gpu = torch.empty([self.dim , 440 - 2*self.low_energy_index ], device = self.device, dtype= torch.float32)
            self.tlayer_cpu = torch.empty([self.dim , 440 - 2*self.low_energy_index ], device = "cpu", dtype= torch.float32)


            for i in range( self.dim ):
                ## Only derivative
    #            for j in range(440):

                ## With second derivative
                for j in range(440 ):
                    self.tlayer_gpu[i,j] = 0.0
                    self.tlayer_cpu[i,j] = 0.0


            for i in range(self.dim):
                self.tlayer_gpu[i,i] = 1
                self.tlayer_cpu[i,i] = 1
            for i in range(4,self.dim-4 ,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + self.dim ] =  10/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + self.dim ] = -10*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + self.dim ] =  10*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + self.dim ] = -10*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + self.dim ] =  10*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + self.dim ] = -10*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + self.dim ] =  10*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + self.dim ] = -10/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + self.dim ] =  10/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + self.dim ] = -10*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + self.dim ] =  10*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + self.dim ] = -10*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + self.dim ] =  10*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + self.dim ] = -10*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + self.dim ] =  10*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + self.dim ] = -10/280.0

            for i in range(1,self.dim - 2,1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 292 ] = -10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 292 ] =  10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 292 ] = -10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 292 ] =  10



        ## Derivative order 8 and 2 and second order derivative of order 4:
        if self.smoothed_version == 5:
            self.tlayer_gpu = torch.empty([self.dim, 4*self.dim - 14 ], device = self.device, dtype= torch.float32)
            self.tlayer_cpu = torch.empty([self.dim, 4*self.dim - 14 ], device = "cpu", dtype= torch.float32)

            for i in range(self.dim ):
                ## Only derivative
    #            for j in range(440):

                ## With second derivative
                for j in range(4*self.dim - 14 ):
                    self.tlayer_gpu[i,j] = 0.0
                    self.tlayer_cpu[i,j] = 0.0


            for i in range( self.dim ):
                self.tlayer_gpu[i,i] = 1
                self.tlayer_cpu[i,i] = 1

            for i in range(4, self.dim - 4,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + self.dim ] =  2.0/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + self.dim ] = -2.0*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + self.dim ] =  2.0*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + self.dim ] = -2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + self.dim ] =  2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + self.dim ] = -2.0*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + self.dim ] =  2.0*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + self.dim ] = -2.0/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + self.dim ] =  2.0/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + self.dim ] = -2.0*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + self.dim ] =  2.0*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + self.dim ] = -2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + self.dim ] =  2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + self.dim ] = -2.0*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + self.dim ] =  2.0*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + self.dim ] = -2.0/280.0

            for i in range(1, self.dim -1, 1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 2*self.dim - 8 ] = -10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 2*self.dim - 8 ] =  10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 2*self.dim - 8 ] = -10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 2*self.dim - 8 ] =  10

            for i in range(2,self.dim -2,1):        # 2nd derivative order 2
                self.tlayer_gpu[ i - 2  ,  i - 2 + 3*self.dim - 10 ] = -1.0/12.0
                self.tlayer_gpu[ i - 1  ,  i - 2 + 3*self.dim - 10 ] =  1.0*4.0/3.0
                self.tlayer_gpu[ i      ,  i - 2 + 3*self.dim - 10 ] = -1.0*5.0/2.0
                self.tlayer_gpu[ i + 1  ,  i - 2 + 3*self.dim - 10 ] =  1.0*4.0/3.0
                self.tlayer_gpu[ i + 2  ,  i - 2 + 3*self.dim - 10 ] = -1.0/12.0

                self.tlayer_cpu[ i - 2  ,  i - 2 + 3*self.dim - 10 ] = -1.0/12.0
                self.tlayer_cpu[ i - 1  ,  i - 2 + 3*self.dim - 10 ] =  1.0*4.0/3.0
                self.tlayer_cpu[ i      ,  i - 2 + 3*self.dim - 10 ] = -1.0*5.0/2.0
                self.tlayer_cpu[ i + 1  ,  i - 2 + 3*self.dim - 10 ] =  1.0*4.0/3.0
                self.tlayer_cpu[ i + 2  ,  i - 2 + 3*self.dim - 10 ] = -1.0/12.0



    def forward(self, x):


        resized = False

        x = x.to(torch.float32)
        if (x.size()[0]) < 8:
            resized = True
        else:
            resized = False

        x = self.linear1(x)
        x = self.activation1(x)


        if resized == True:
            x = x.view(1, 1024, 1)
            x = x.transpose(1, 2).contiguous()
        else:
            x = x.view(self.batchsize, 1024, 1)
            x = x.transpose(1, 2).contiguous()


        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.maxpool(x)

        x = self.conv6(x)
        x = self.relu6(x)

        if resized == False:
            x = x.view(self.batchsize, 50*109)
        else:
            x = x.view(1, 50*109)

        x = self.linear7(x)
        x = self.softmax(x)


        if x.is_cuda == False:
            x = torch.matmul(x, self.tlayer_cpu)
        if x.is_cuda == True:
            x = torch.matmul(x, self.tlayer_gpu)


        if resized == True:
            x = x[0]
        return x.double()



    def update_derivative_matrix(self, coeff_der_2, coeff_der_8, coeff_der_2_4):



        if self.smoothed_version == 5:



            for i in range(4, self.dim - 4,1):        # Start with order 8 here
                self.tlayer_gpu[ i - 4  ,  i - 4 + self.dim ] =  coeff_der_8*2.0/280.0
                self.tlayer_gpu[ i - 3  ,  i - 4 + self.dim ] = -coeff_der_8*2.0*4/105.0
                self.tlayer_gpu[ i - 2  ,  i - 4 + self.dim ] =  coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_gpu[ i - 1  ,  i - 4 + self.dim ] = -coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 1  ,  i - 4 + self.dim ] =  coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_gpu[ i + 2  ,  i - 4 + self.dim ] = -coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_gpu[ i + 3  ,  i - 4 + self.dim ] =  coeff_der_8*2.0*(4/105.0)
                self.tlayer_gpu[ i + 4  ,  i - 4 + self.dim ] = -coeff_der_8*2.0/280.0


                self.tlayer_cpu[ i - 4  ,  i - 4 + self.dim ] =  coeff_der_8*2.0/280.0
                self.tlayer_cpu[ i - 3  ,  i - 4 + self.dim ] = -coeff_der_8*2.0*4/105.0
                self.tlayer_cpu[ i - 2  ,  i - 4 + self.dim ] =  coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_cpu[ i - 1  ,  i - 4 + self.dim ] = -coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 1  ,  i - 4 + self.dim ] =  coeff_der_8*2.0*(4.0/5.0)
                self.tlayer_cpu[ i + 2  ,  i - 4 + self.dim ] = -coeff_der_8*2.0*(1.0/5.0)
                self.tlayer_cpu[ i + 3  ,  i - 4 + self.dim ] =  coeff_der_8*2.0*(4/105.0)
                self.tlayer_cpu[ i + 4  ,  i - 4 + self.dim ] = -coeff_der_8*2.0/280.0

            for i in range(1, self.dim -1, 1):        # now order 2
                self.tlayer_gpu[ i - 1  ,  i - 1 + 2*self.dim - 8 ] = -coeff_der_2*10
                self.tlayer_gpu[ i + 1  ,  i - 1 + 2*self.dim - 8 ] =  coeff_der_2*10

                self.tlayer_cpu[ i - 1  ,  i - 1 + 2*self.dim - 8 ] = -coeff_der_2*10
                self.tlayer_cpu[ i + 1  ,  i - 1 + 2*self.dim - 8 ] =  coeff_der_2*10

            for i in range(2,self.dim -2,1):        # 2nd derivative order 2
                self.tlayer_gpu[ i - 2  ,  i - 2 + 3*self.dim - 10 ] = -coeff_der_2_4*1.0/12.0
                self.tlayer_gpu[ i - 1  ,  i - 2 + 3*self.dim - 10 ] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_gpu[ i      ,  i - 2 + 3*self.dim - 10 ] = -coeff_der_2_4*1.0*5.0/2.0
                self.tlayer_gpu[ i + 1  ,  i - 2 + 3*self.dim - 10 ] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_gpu[ i + 2  ,  i - 2 + 3*self.dim - 10 ] = -coeff_der_2_4*1.0/12.0

                self.tlayer_cpu[ i - 2  ,  i - 2 + 3*self.dim - 10 ] = -coeff_der_2_4*1.0/12.0
                self.tlayer_cpu[ i - 1  ,  i - 2 + 3*self.dim - 10 ] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_cpu[ i      ,  i - 2 + 3*self.dim - 10 ] = -coeff_der_2_4*1.0*5.0/2.0
                self.tlayer_cpu[ i + 1  ,  i - 2 + 3*self.dim - 10 ] =  coeff_der_2_4*1.0*4.0/3.0
                self.tlayer_cpu[ i + 2  ,  i - 2 + 3*self.dim - 10 ] = -coeff_der_2_4*1.0/12.0




