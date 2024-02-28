import torch
import h5py as h5
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from .learning_utils import (
    SopranoPhotonDataset,
    SimpleModel,
    LegendreModel,
    LegendreSplitModel,
    SimpleModelSmoothed,
    SimpleModelConvSmoothed
)
from .electron_utils import SopranoElectronDataset, ElectronModelConvSmoothed


class SSC_data:

    def __init__(self, fname=None, inference_folder=None):
        if fname:  # for inference it will be None.
            self.f = h5.File(fname, "r")

            self.photon_spectrum = self.f["values"]["output_0"]
            self.electron_spectrum = self.f["values"]["output_1"]

            self.parameter_model = self.f["parameters"]
            self.energy_grid_photon = self.f["energy_grid"]["energy_grid_0"][:]
            self.energy_grid_electron = self.f["energy_grid"]["energy_grid_1"][:]

        else:
            self.energy_grid_photon = np.load(f"{inference_folder}/energy_grid_photon.npy")
            self.energy_grid_electron = np.load(f"{inference_folder}/energy_grid_electron.npy")


class Soprano_SSC(SSC_data):

    def __init__(self, fname, fraction_training=0.8, data_batch_size=32, which_transform="normal", order=35,
                 low_energy_index=35, \
                 single_average=False, smoothed_version=1, update_coeff=False, coeff_reconstruct={}):

        """
        This class provides all necessary elements to load the data from disk, clean them, normalise them, and split them into training set and validation set.

        Arguments:
            1- fname: h5 file containing the model
            2- fraction_training = 0.8: which fraction of the data is used in the training set. 1 - fraction_training will be used for the validation set.
            3- data_batch_size: batchsize of the data to be fed to the learning algorythm
            4- which_transform: changes the method for the transformation of the data, and as such the output layers. Valid choices are:
                (i) "normal": corresponding to the assumption of independence of all outputs
                (ii) "legendre": a Legendre decomposition of order order is performed on the initial spectra with numpy, and the machine learning learns the reconstruction polynoms.
                (iii) "normalenhanced": normal + fit for correlation between outputs
            5- order: order to which to perform the Legendre decomposition.
            6- index at which to cut the low energy. Valid for which_transform == legendre
            7- smoothed version: 1 for derivative order 2, 2 for derivative order 6, 3 for derivative order 8, 4 for derivative order 8 + 2, 5 for derivative order 8 + 2  and second derivative order 4.
            8a- update_coeff: True if to update the coefficient of the derivative. False otherwise
            8b- coeff_reconstruct: dictionnary with coeff_der_2, coeff_der_8, coeff_der_2_4 containing the values of the coefficients to put in front of the derivatives
        """
        super().__init__(fname)

        self.__filter_bad_data()

        self.N = len(self.f_filtered)

        self.transform = which_transform

        self.fraction_training = fraction_training

        self.batchsize = data_batch_size

        self.single_average = single_average

        self.order = order

        self.low_energy_index = low_energy_index

        self.smoothed_version = smoothed_version

        self.coeff_reconstruct = coeff_reconstruct

        if which_transform == "normal":
            self.__initialise_normal_learning(fname, fraction_training, data_batch_size)
            self.tinymodel = SimpleModel()

        elif which_transform == "legendre":
            self.__initialise_legendre_learning(fname, fraction_training, data_batch_size, order)
            #            self.tinymodel = LegendreModel(order, data_batch_size)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tinymodel = LegendreSplitModel(order, data_batch_size, device)

        elif which_transform == "normalenhanced":
            self.__initialise_smoothed_learning(fname, fraction_training, data_batch_size)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #            self.tinymodel = SimpleModelSmoothed(data_batch_size, device, self.smoothed_version )
            self.tinymodel = SimpleModelConvSmoothed(data_batch_size, device, self.smoothed_version,
                                                     self.low_energy_index)

        else:
            raise TypeError("Unknown transform type")

        self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.0001, momentum_decay=0.004,
                                           weight_decay=0.0)
        #        self.optimizer = torch.optim.Adadelta(self.tinymodel.parameters() )
        #        self.loss_fn = torch.nn.HuberLoss( reduction='sum', delta = 0.5 )

        #        self.loss_fn = self.cauchy_loss()
        self.loss_fn = torch.nn.L1Loss(reduction='sum')

    #        self.loss_fn = torch.nn.MSELoss( reduction='sum' )

    def __initialise_normal_learning(self, fname, fraction_training, data_batch_size):

        print("Beginning data transform for the photon spectrum")
        self.full_spectrum_transformed = self.__transform_data(self.f_filtered, self.energy_grid_photon)
        print("End data transform for the photon spectrum")
        print("Beginning detrending for the photon spectrum")
        self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(
            self.full_spectrum_transformed, "parameters")
        print("End data detrending for the photon spectrum")
        print("Beginning detrending parameters")
        self.params_detrended, self.mean_param, self.variance_param = self.__detrend(self.param_filtered, "parameters")
        print("End data detrending")

        full_spectrum = self.full_spectrum_transformed_detrended
        full_param = self.params_detrended
        nb_spectrum = len(full_spectrum)

        ## Set the generator for reproducibility:
        generator1 = torch.Generator().manual_seed(42)

        ## Set the indices of the split:
        A = torch.utils.data.random_split(full_spectrum, [fraction_training, 1.0 - fraction_training],
                                          generator=generator1)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)

        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        self.parameter_train = np.array(full_param)[A_sorted_train]

        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        self.parameter_valid = np.array(full_param)[A_sorted_valid]

        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)

        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=data_batch_size, shuffle=True,
                                      drop_last=True)
        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=data_batch_size, shuffle=True,
                                            drop_last=True)

    def __initialise_smoothed_learning(self, fname, fraction_training, data_batch_size):

        print("Beginning data transform for the photon spectrum")
        self.full_spectrum_transformed = self.__transform_data(self.f_filtered, self.energy_grid_photon)
        print("End data transform for the photon spectrum")

        print("Beginning detrending for the photon spectrum")

        if self.single_average == False:
            self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(
                self.full_spectrum_transformed, "parameters")
        else:
            self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(
                self.full_spectrum_transformed, "model")
        print("End data detrending for the photon spectrum")

        print("Add summs (for now only by 3)")
        self.full_spectrum_transformed_add = self.__add_info_smoothed()
        self.full_spectrum_transformed_add = self.enhanced_output
        print("End Add summs: array self.enhanced_output was created")

        print("Beginning detrending parameters")
        self.params_detrended, self.mean_param, self.variance_param = self.__detrend(self.param_filtered, "parameters")
        print("End data detrending")

        full_spectrum = self.enhanced_output
        full_param = self.params_detrended
        nb_spectrum = len(full_spectrum)

        ## Set the generator for reproducibility:
        generator1 = torch.Generator().manual_seed(42)

        ## Set the indices of the split:
        A = torch.utils.data.random_split(full_spectrum, [fraction_training, 0.5 * (1.0 - fraction_training),
                                                          0.5 * (1.0 - fraction_training)], generator=generator1)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)
        A_sorted_test = np.sort(A[2].indices)

        self.A_sorted_train = np.sort(A[0].indices)
        self.A_sorted_valid = np.sort(A[1].indices)
        self.A_sorted_test = np.sort(A[2].indices)

        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        self.parameter_train = np.array(full_param)[A_sorted_train]

        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        self.parameter_valid = np.array(full_param)[A_sorted_valid]

        self.spectrum_test = np.array(full_spectrum)[A_sorted_test]
        self.parameter_test = np.array(full_param)[A_sorted_test]

        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)
        self.Soprano_SSC_Dataset_test = SopranoPhotonDataset(self.parameter_test, self.spectrum_test)

        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=data_batch_size, shuffle=True,
                                      drop_last=True)
        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=data_batch_size, shuffle=True,
                                            drop_last=True)
        self.test_loader = DataLoader(self.Soprano_SSC_Dataset_test, batch_size=data_batch_size, shuffle=True,
                                      drop_last=True)

        self.save_averages()
        print("Averages were saved")

    def recomputed_smoothed_learning(self, coeff_reconstruct):

        self.coeff_reconstruct = coeff_reconstruct
        print("Add summs (for now only by 3)")
        self.full_spectrum_transformed_add = self.__add_info_smoothed()
        self.full_spectrum_transformed_add = self.enhanced_output
        print("End Add summs: array self.enhanced_output was created")

        full_spectrum = self.enhanced_output
        full_param = self.params_detrended
        nb_spectrum = len(full_spectrum)

        ## Set the generator for reproducibility:
        generator1 = torch.Generator().manual_seed(42)

        #        ## Set the indices of the split:
        #        A = torch.utils.data.random_split(full_spectrum, [self.fraction_training, 1.0 - self.fraction_training], generator=generator1)
        #
        #
        #        ## Sort the indices:
        #        A_sorted_train = np.sort(A[0].indices)
        #        A_sorted_valid = np.sort(A[1].indices)
        #
        #        self.A_sorted_train = np.sort(A[0].indices)
        #        self.A_sorted_valid = np.sort(A[1].indices)
        #
        #        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        #        self.parameter_train = np.array(full_param)[A_sorted_train]
        #
        #        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        #        self.parameter_valid = np.array(full_param)[A_sorted_valid]
        #
        #
        #        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        #        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)
        #
        #
        #        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=self.batchsize, shuffle=True, drop_last = True)
        #        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=self.batchsize, shuffle=True, drop_last = True)

        ## Set the indices of the split:
        A = torch.utils.data.random_split(full_spectrum, [self.fraction_training, 0.5 * (1.0 - self.fraction_training),
                                                          0.5 * (1.0 - self.fraction_training)], generator=generator1)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)
        A_sorted_test = np.sort(A[2].indices)

        self.A_sorted_train = np.sort(A[0].indices)
        self.A_sorted_valid = np.sort(A[1].indices)
        self.A_sorted_test = np.sort(A[2].indices)

        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        self.parameter_train = np.array(full_param)[A_sorted_train]

        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        self.parameter_valid = np.array(full_param)[A_sorted_valid]

        self.spectrum_test = np.array(full_spectrum)[A_sorted_test]
        self.parameter_test = np.array(full_param)[A_sorted_test]

        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)
        self.Soprano_SSC_Dataset_test = SopranoPhotonDataset(self.parameter_test, self.spectrum_test)

        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=self.batchsize, shuffle=True,
                                      drop_last=True)
        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=self.batchsize, shuffle=True,
                                            drop_last=True)
        self.test_loader = DataLoader(self.Soprano_SSC_Dataset_test, batch_size=self.batchsize, shuffle=True,
                                      drop_last=True)

        self.save_averages()

    def __initialise_legendre_learning(self, fname, fraction_training, data_batch_size, order):

        print("Beginning data transform for the photon spectrum")
        self.full_spectrum_transformed = self.__transform_data(self.f_filtered, self.energy_grid_photon)
        print("End data transform for the photon spectrum")
        print("Beginning the computation of the Legendre coefficients for the photon spectrum")
        self.__legendre_compute(order)

        print("Beginning detrending for the photon spectrum")
        self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(
            self.Lcoef, "parameters")
        print("End data detrending for the photon spectrum")
        print("Beginning detrending parameters")
        self.params_detrended, self.mean_param, self.variance_param = self.__detrend(self.param_filtered, "parameters")
        print("End data detrending")

        full_spectrum = self.full_spectrum_transformed_detrended
        full_param = self.params_detrended
        nb_spectrum = len(full_spectrum)

        ## Set the generator for reproducibility:
        generator1 = torch.Generator().manual_seed(42)

        ## Set the indices of the split:
        A = torch.utils.data.random_split(full_spectrum, [fraction_training, 1.0 - fraction_training],
                                          generator=generator1)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)

        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        self.parameter_train = np.array(full_param)[A_sorted_train]

        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        self.parameter_valid = np.array(full_param)[A_sorted_valid]

        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)

        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=data_batch_size, shuffle=True,
                                      drop_last=True)
        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=data_batch_size, shuffle=True,
                                            drop_last=True)

    def cauchy_loss(input, target):
        residual = target - input
        return torch.mean(torch.log1p(residual ** 2))

    def zero_condition(self, x):
        if x > 0:
            return np.log10(x)
        else:
            return -350.0

    def __transform_data(self, data, energy_grid):
        result = []
        for idx in range(len(data)):
            A = [self.zero_condition(
                data[idx][i] * (10 ** energy_grid[i]) * (10 ** energy_grid[i]) / np.sqrt((10 ** energy_grid[i]))) for i
                in range(len(energy_grid))]
            A = A - np.max(A) + 40
            B = -50.0 * np.ones(150)
            B[A > 0] = np.log10(data[idx][A > 0])
            if (np.inf in B or -np.inf in B):
                print(B, idx)
            result.append(B)

        return result

    def __add_info_smoothed(self):

        ## I want to transform self.full_spectrum_transformed_detrended
        ## Derivative order 8
        #        self.enhanced_output = np.zeros((self.N, 292))
        ## Derivative order 8 + 2
        if self.smoothed_version == 4:
            self.enhanced_output = np.zeros((self.N, 440 - 3 * self.low_energy_index))
        ## Derivative order 8 + 2 and 2nd derivative order 4:
        if self.smoothed_version == 5:
            self.enhanced_output = np.zeros((self.N, 586 - 3 * self.low_energy_index))

        for i in range(self.N):

            for j in range(150 - self.low_energy_index):
                self.enhanced_output[i][j] = self.full_spectrum_transformed_detrended[i][j + self.low_energy_index]

            #            for j in range(50):
            #                self.enhanced_output[i][j+150] =  self.full_spectrum_transformed_detrended[i][3*j] + 2.0* self.full_spectrum_transformed_detrended[i][3*j+1] + self.full_spectrum_transformed_detrended[i][3*j+2]
            #
            #            for j in range(30):
            #                self.enhanced_output[i][j+200] =  3.0* self.full_spectrum_transformed_detrended[i][5*j] + self.full_spectrum_transformed_detrended[i][5*j+1] + \
            #                        4.0*self.full_spectrum_transformed_detrended[i][5*j+2] + self.full_spectrum_transformed_detrended[i][5*j+3] + 3.0* self.full_spectrum_transformed_detrended[i][5*j+4]

            #            for j in range(2, 147, 1):
            #                self.enhanced_output[i][j+150 - 1] =  self.full_spectrum_transformed_detrended[i][ j - 2] + self.full_spectrum_transformed_detrended[i][ j - 1] + \
            #                    2.0*self.full_spectrum_transformed_detrended[i][ j ] + self.full_spectrum_transformed_detrended[i][ j + 1 ] + self.full_spectrum_transformed_detrended[i][ j + 2 ]

            ## Derivative order 2
            #            for j in range(1, 148, 1):
            #                self.enhanced_output[i][j+150 - 1] = - 10*self.full_spectrum_transformed_detrended[i][ j - 1] + 10*self.full_spectrum_transformed_detrended[i][ j + 1 ]
            ## Derivative order 6
            #            for j in range(4, 146, 1):
            #                self.enhanced_output[i][j+150 - 4] =  - 20.0/60.0*self.full_spectrum_transformed_detrended[i][ j - 3]       \
            #                                                    + 20*(3.0/20.0)*self.full_spectrum_transformed_detrended[i][ j - 2 ]    \
            #                                                    - 20*(3.0/4.0)*self.full_spectrum_transformed_detrended[i][ j - 1 ]     \
            #                                                    + 20*(3.0/4.0)*self.full_spectrum_transformed_detrended[i][ j + 1 ]     \
            #                                                    - 20*(3.0/20.0)*self.full_spectrum_transformed_detrended[i][ j + 2 ]    \
            #                                                    + 20.0/60.0*self.full_spectrum_transformed_detrended[i][ j + 3]

            ##  Derivative order 8
            #            for j in range(4, 146, 1):
            #                self.enhanced_output[i][j+150 - 4] =   10.0/280.0*self.full_spectrum_transformed_detrended[i][ j - 4]       \
            #                                                    - 10*(4.0/105.0)*self.full_spectrum_transformed_detrended[i][ j - 3 ]   \
            #                                                    + 10*(1.0/5.0)*self.full_spectrum_transformed_detrended[i][ j - 2 ]     \
            #                                                    - 10*(4.0/5.0)*self.full_spectrum_transformed_detrended[i][ j - 1 ]     \
            #                                                    + 10*(4.0/5.0)*self.full_spectrum_transformed_detrended[i][ j + 1 ]     \
            #                                                    - 10*(1.0/5.0)*self.full_spectrum_transformed_detrended[i][ j + 2 ]     \
            #                                                    + 10.0*(4.0/105.0)*self.full_spectrum_transformed_detrended[i][ j + 3]  \
            #                                                    - 10.0/280*self.full_spectrum_transformed_detrended[i][ j + 4]

            ##  Derivative order 8 and 2:
            if self.smoothed_version == 4:
                coeff_der_2 = self.coeff_reconstruct["coeff_der_2"]
                coeff_der_8 = self.coeff_reconstruct["coeff_der_8"]
                coeff_der_2_4 = self.coeff_reconstruct["coeff_der_2_4"]

                for j in range(4, 146 - self.low_energy_index, 1):
                    self.enhanced_output[i][j + 150 - self.low_energy_index - 4] = coeff_der_8 * 10.0 / 280.0 * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 4] \
                                                                                   - coeff_der_8 * 10.0 * (
                                                                                           4.0 / 105.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 3] \
                                                                                   + coeff_der_8 * 10.0 * (1.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 2] \
                                                                                   - coeff_der_8 * 10.0 * (4.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 1] \
                                                                                   + coeff_der_8 * 10.0 * (4.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 1] \
                                                                                   - coeff_der_8 * 10.0 * (1.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 2] \
                                                                                   + coeff_der_8 * 10.0 * (
                                                                                           4.0 / 105.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 3] \
                                                                                   - coeff_der_8 * 10.0 / 280 * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 4]

                for j in range(1, 148 - self.low_energy_index, 1):
                    self.enhanced_output[i][j + 292 - 2 * self.low_energy_index - 1] = - 10.0 * coeff_der_2 * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index - 1] \
                                                                                       + 10.0 * coeff_der_2 * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index + 1]

            ##  Derivative order 8 and 2 and 2nd derivative order 4:
            if self.smoothed_version == 5:
                coeff_der_2 = self.coeff_reconstruct["coeff_der_2"]
                coeff_der_8 = self.coeff_reconstruct["coeff_der_8"]
                coeff_der_2_4 = self.coeff_reconstruct["coeff_der_2_4"]
                for j in range(4, 146 - self.low_energy_index, 1):
                    self.enhanced_output[i][j + 150 - self.low_energy_index - 4] = coeff_der_8 * 2.0 / 280.0 * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 4] \
                                                                                   - coeff_der_8 * 2.0 * (4.0 / 105.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 3] \
                                                                                   + coeff_der_8 * 2.0 * (1.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 2] \
                                                                                   - coeff_der_8 * 2.0 * (4.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index - 1] \
                                                                                   + coeff_der_8 * 2.0 * (4.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 1] \
                                                                                   - coeff_der_8 * 2.0 * (1.0 / 5.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 2] \
                                                                                   + coeff_der_8 * 2.0 * (4.0 / 105.0) * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 3] \
                                                                                   - coeff_der_8 * 2.0 / 280 * \
                                                                                   self.full_spectrum_transformed_detrended[
                                                                                       i][j + self.low_energy_index + 4]

                for j in range(1, 148 - self.low_energy_index, 1):
                    self.enhanced_output[i][j + 292 - 2 * self.low_energy_index - 1] = - coeff_der_2 * 10.0 * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index - 1] \
                                                                                       + coeff_der_2 * 10.0 * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index + 1]

                for j in range(2, 147 - self.low_energy_index, 1):
                    self.enhanced_output[i][j + 440 - 3 * self.low_energy_index - 2] = - coeff_der_2_4 * 1.0 / 12.0 * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index - 2] \
                                                                                       + coeff_der_2_4 * 1.0 * (
                                                                                               4.0 / 3.0) * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index - 1] \
                                                                                       - coeff_der_2_4 * 1.0 * (
                                                                                               5.0 / 2.0) * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][j + self.low_energy_index] \
                                                                                       + coeff_der_2_4 * 1.0 * (
                                                                                               4.0 / 3.0) * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index + 1] \
                                                                                       - coeff_der_2_4 * 1.0 / 12.0 * \
                                                                                       self.full_spectrum_transformed_detrended[
                                                                                           i][
                                                                                           j + self.low_energy_index + 2]

    def __legendre_compute(self, order):

        self.Lcoef = np.zeros((self.N, order))
        self.initial_index = np.zeros(self.N)
        self.last_index = np.zeros(self.N)

        for i in range(self.N):
            # for i in range(10):
            AA = []
            ## Find first non -90 elements:
            ifirst = -1
            ilast = 151
            for j in range(150 - 1):
                ## removed too small spectrum:
                if self.full_spectrum_transformed[i][j] < -89 and self.full_spectrum_transformed[i][j + 1] > -90:
                    ifirst = j + 1

                if self.full_spectrum_transformed[i][j] > -90 and self.full_spectrum_transformed[i][j + 1] < -89:
                    ilast = j

            if ilast > 148:
                ilast = 148

            for j in range(ilast, ifirst, -1):
                ## removed too small spectrum:
                if self.full_spectrum_transformed[i][j - 1] - self.full_spectrum_transformed[i][j] > 3:
                    ilast = j

            AA = np.zeros(150)
            for j in range(ifirst, ilast + 1, 1):
                AA[j] = self.full_spectrum_transformed[i][j]
            #            AA = self.full_spectrum_transformed[i][ifirst: ilast+1]
            self.initial_index[i] = ifirst
            self.last_index[i] = ilast

            for j in range(ifirst, -1, -1):
                AA[j] = AA[j + 1] + self.full_spectrum_transformed[i][ifirst] - self.full_spectrum_transformed[i][
                    ifirst + 1]

            for j in range(ilast, 150, 1):
                AA[j] = AA[j - 1] + 2.0

            #            print(AA, len(AA))

            #            x = np.linspace(-1,1,len(AA))
            x = np.linspace(-1, 1, 150 - self.low_energy_index)
            #            print(len(x), len(AA))
            #            dec = np.polynomial.legendre.legfit(x, np.asarray(AA)+35, deg = order)
            #            dec = np.polynomial.legendre.legfit(x[ifirst: ilast+1], np.asarray(AA)+35, deg = order)
            dec = np.polynomial.legendre.legfit(x, np.asarray(AA)[self.low_energy_index:] + 35, deg=order - 1)
            #            print(dec)
            ## I will realign the frequencies by b^2 gamma_min:

            # Recall all of it is logscale:
            #            b = self.param_filtered[i][0]
            #            gmin2 = 2.0*self.param_filtered[i][3]
            #
            #            nuGrid = self.energy_grid_photon - b - gmin2
            #
            #            ## This is the equivalent of the new grid in set in [-1,1]
            #            nuGrid_transform = self.__frequency_compress(nuGrid)
            #
            #            if np.min(nuGrid_transform[self.low_energy_index:]) < -1.0  or np.max(nuGrid_transform[self.low_energy_index:]) > 1.0 :
            #                raise TypeError("nuGrid does not have the proper bounds", nuGrid_transform)
            #
            #            #print(nuGrid_transform)
            #            dec = np.polynomial.legendre.legfit(nuGrid_transform[self.low_energy_index:], np.asarray(AA)[self.low_energy_index:]+35, deg = order)
            #            print(nuGrid_transform, dec)
            for j in range(order):
                self.Lcoef[i][j] = dec[j]

        ## Now self.Lcoef contains the coefficients of the Legendre polynomials of reconstruction of the spectrum
        ## They will be the data to train onto.

    def __frequency_compress(self, Enu):
        """
        Move the extended frequency grid [-20, 35] to [-1,1] to take the Legendre transform
        """
        xmin = -5
        xmax = 32

        a = 2.0 / (-xmin + xmax)
        b = -(xmax + xmin) / (-xmin + xmax)
        # b = -5.0/11.0
        gridresult = np.zeros(len(Enu))
        for i in range(len(Enu)):
            gridresult[i] = a * Enu[i] + b

        return gridresult

    def __detrend(self, data, data_type):

        if self.single_average == False or data_type == "parameters":
            result = np.zeros(np.shape(data))
            mean = np.zeros(len(data[0]))
            variance = np.zeros(len(data[0]))
            for j in range(len(data[0])):
                A = [data[i][j] for i in range(len(data))]
                # print(j, np.mean(  A  ), np.var(  A  ))
                # varA = np.var(  A  )
                meanA = np.mean(A)
                varA = np.max(np.abs(A - meanA))
                # print(j, meanA,  varA)

                mean[j] = meanA
                variance[j] = varA

                for i in range(len(data)):
                    if (varA > 0.0):
                        result[i][j] = (data[i][j] - meanA) / (1.1 * varA)
                    else:
                        result[i][j] = (data[i][j] - meanA)

            return result, mean, variance

        if self.single_average == True and data_type != "parameters":

            result = np.zeros(np.shape(data))
            meanA = np.mean(np.mean(data))
            maxdata = np.max(np.max(data))
            mindata = np.min(np.min(data))
            varA = maxdata - mindata

            for j in range(len(data[0])):

                for i in range(len(data)):
                    if (varA > 0.0):
                        result[i][j] = (data[i][j] - meanA) / (1.1 * varA)
                    else:
                        result[i][j] = (data[i][j] - meanA)

            return result, meanA, varA

        raise TypeError("I dont know what to do here with the detrend")

    def __filter_bad_data(self):

        self.f_filtered = []
        self.param_filtered = []
        for i in range(len(self.f["parameters"])):
            if np.max(self.f["values"]["output_0"][i]) > 0.0:
                self.f_filtered.append(np.array(self.f["values"]["output_0"][i]))
                self.param_filtered.append(np.array(self.f["parameters"][i]))

    def compute_spectrum(self, parameters):

        param = [parameters["log_B"],
                 parameters["log_electron_luminosity"],
                 parameters["log_gamma_cut"],
                 parameters["log_gamma_min"],
                 parameters["log_radius"],
                 parameters["lorentz_factor"],
                 parameters["spectral_index"]
                 ]

        # Apply detrend to the physical parameter:
        for j in range(7):

            meanA = self.mean_param[j]
            varA = self.variance_param[j]

            if (varA > 0.0):
                param[j] = (param[j] - meanA) / (1.1 * varA)
            else:
                param[j] = (param[j] - meanA)

        # Compute the model to get fully detrended reconstruction;
        result = torch.detach(self.tinymodel(torch.tensor(param))).numpy()

        for j in range(150 - self.low_energy_index):
            if self.single_average == False:
                meanA = self.mean_spectrum_transformed[j]
                varA = self.variance_spectrum_transformed[j]
            else:
                meanA = self.mean_spectrum_transformed
                varA = self.variance_spectrum_transformed

            if (varA > 0.0):
                result[j] = (result[j] * (1.1 * varA) + meanA)
            else:
                result[j] = (result[j] + meanA)

        return result

    def compute_spectrum_Legendre(self, parameters):

        param = [parameters["log_B"],
                 parameters["log_electron_luminosity"],
                 parameters["log_gamma_cut"],
                 parameters["log_gamma_min"],
                 parameters["log_radius"],
                 parameters["lorentz_factor"],
                 parameters["spectral_index"]
                 ]

        # Apply detrend to the physical parameter:
        for j in range(7):

            meanA = self.mean_param[j]
            varA = self.variance_param[j]

            if (varA > 0.0):
                param[j] = (param[j] - meanA) / (1.1 * varA)
            else:
                param[j] = (param[j] - meanA)

        # Compute the model to get fully detrended reconstruction;
        Lresult = torch.detach(self.tinymodel(torch.tensor(param)).cpu()).numpy()
        # print(Lresult)
        # result is the biased Legendre coefficients reconstructed

        for j in range(self.order):
            meanA = self.mean_spectrum_transformed[j]
            varA = self.variance_spectrum_transformed[j]
            if (varA > 0.0):
                Lresult[j] = (Lresult[j] * (1.1 * varA) + meanA)
            else:
                Lresult[j] = (Lresult[j] + meanA)

        # Recall all of it is logscale:
        #        b = param[0]
        #        gmin2 = param[3]
        #
        #        nuGrid = self.energy_grid_photon - b - gmin2
        #
        #        ## This is the equivalent of the new grid in set in [-1,1]
        #        nuGrid_transform = self.__frequency_compress(nuGrid)
        #
        #        if np.min(nuGrid_transform) < -1.0  or np.max(nuGrid_transform) > 1.0 :
        #            raise TypeError("nuGrid does not have the proper bounds", nuGrid_transform)

        xx = np.linspace(-1, 1, 150 - self.low_energy_index)
        result = np.polynomial.legendre.legval(xx, Lresult) - 35
        # print(result)
        #        result = np.polynomial.legendre.legval(nuGrid_transform, Lresult) - 35
        resultt = -100 * np.ones(150)
        for i in range(150 - self.low_energy_index):
            resultt[i + self.low_energy_index] = result[i]

        return resultt

    def compute_nuFnu_spectrum(self, parameters):

        if self.transform == "normal" or self.transform == "normalenhanced":
            Nnu = self.compute_spectrum(parameters)
        else:
            Nnu = self.compute_spectrum_Legendre(parameters)

        # print("Nnu = ", Nnu)

        nuFnu = np.zeros(150)
        for i in range(self.low_energy_index, 150, 1):
            nuFnu[i] = 10 ** Nnu[i - self.low_energy_index] * (10 ** self.energy_grid_photon[i]) * (
                    10 ** self.energy_grid_photon[i]) / np.sqrt((10 ** self.energy_grid_photon[i]))
        return nuFnu

    def __train_one_epoch(self, epoch_index, device):  # , tb_writer):

        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.data_loader):
            # Every data instance is an input + label pair
            parameters, spectrum = data

            parameters_gpu = parameters.to(device)
            spectrum_gpu = spectrum.to(device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.tinymodel(parameters_gpu)

            # Compute the loss and its gradients
            # print("outputs = ", outputs)
            # print("spectrum_gpu = ", spectrum_gpu)
            loss = self.loss_fn(outputs, spectrum_gpu)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 20 == 0:
                last_loss = running_loss / 100  # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.data_loader) + i + 1
                # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def time_update_coeff(self, x):

        #        from 0 to 500 keep all to 0:
        #        from 500 to 1500 increase second order only to 1
        #        from 1500 to 2500 increase 8th order to 1
        #        from 2500 to 3000 increase 2nd derivative to 1

        reconstruct = {"coeff_der_2": 1.0, "coeff_der_8": 1.0, "coeff_der_2_4": 4.0}
        #        if x < 500:
        #            reconstruct = {"coeff_der_2":0.0, "coeff_der_8":0.0,"coeff_der_2_4":0.0}
        #        elif x < 1500:
        #            reconstruct["coeff_der_2"] = (x-500.0)/1000.0
        #            reconstruct["coeff_der_8"] = 0.0
        #            reconstruct["coeff_der_2_4"] = 0.0
        #        elif x < 2500:
        #            reconstruct["coeff_der_2"] = 1.0
        #            reconstruct["coeff_der_8"] = 2.0*(1.0*x-1500.0)/1000.0
        #            reconstruct["coeff_der_2_4"] = 0.0
        #        elif x < 3500:
        #            reconstruct["coeff_der_2"] = 1.0
        #            reconstruct["coeff_der_8"] = 2.0
        #            reconstruct["coeff_der_2_4"] = 3.0*(x-2500.0)/1000.0
        #        else:
        #            reconstruct["coeff_der_2"] = 1.0
        #            reconstruct["coeff_der_8"] = 2.0
        #            reconstruct["coeff_der_2_4"] = 3.0

        return reconstruct

    def train(self, EPOCHS=50, save_intermediate=False, decaying_lr=False):
        import matplotlib.pyplot as plt  # noqa
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tinymodel.to(device)

        self.loss_train = np.zeros(EPOCHS)
        self.loss_valid = np.zeros(EPOCHS)

        best_vloss = 1_000_000.

        plt.ion()
        fig = plt.figure()
        plt.show()

        plt.xlim(0, EPOCHS)
        plt.ylim(0, 50)
        time_coefficients = {"coeff_der_2": 0.0, "coeff_der_8": 0.0, "coeff_der_2_4": 0.0}
        self.coeff_reconstruct = time_coefficients

        if decaying_lr == True:
            self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.001, momentum_decay=0.04,
                                               weight_decay=0.1)

        for epoch in range(EPOCHS):
            # print('EPOCH {}:'.format(epoch_number + 1))

            if decaying_lr == True and epoch == 25:
                self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.0001, momentum_decay=0.004,
                                                   weight_decay=0.0)

            if decaying_lr == True and epoch == 75:
                self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.00001, momentum_decay=0.002,
                                                   weight_decay=0.0)

            if (epoch == 0):
                time_coefficients = self.time_update_coeff(epoch)
                self.recomputed_smoothed_learning(time_coefficients)
                self.tinymodel.update_derivative_matrix(time_coefficients["coeff_der_2"],
                                                        time_coefficients["coeff_der_8"],
                                                        time_coefficients["coeff_der_2_4"])

            #            if(epoch%50 == 0  and epoch > 498):
            #                time_coefficients = self.time_update_coeff(epoch)
            #                self.recomputed_smoothed_learning(time_coefficients )
            #                self.tinymodel.update_derivative_matrix(time_coefficients["coeff_der_2"], time_coefficients["coeff_der_8"], time_coefficients["coeff_der_2_4"] )

            # Make sure gradient tracking is on, and do a pass over the data
            self.tinymodel.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, device)  # , writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.tinymodel.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    vinputs_GPU = vinputs.to(device)
                    vlabels_GPU = vlabels.to(device)
                    voutputs = self.tinymodel(vinputs_GPU)
                    vloss = self.loss_fn(voutputs, vlabels_GPU)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            # print('EPOCH {}   LOSS train {} valid {}'.format(epoch_number + 1, avg_loss, avg_vloss))
            self.loss_train[epoch] = avg_loss
            self.loss_valid[epoch] = avg_vloss

            # clear_output(wait=True)

            print("Coeffs = ", time_coefficients["coeff_der_2"], time_coefficients["coeff_der_8"],
                  time_coefficients["coeff_der_2_4"])
            # if(epoch_number == 0):
            plt.plot(self.loss_train[:epoch_number + 1], color="red", label="Training set")
            plt.plot(self.loss_valid[:epoch_number + 1], color="blue", label="Validation set")

            plt.plot((0.3 * self.loss_valid[0] / (self.loss_valid[0] / self.loss_train[0])) * self.loss_valid[
                                                                                              :epoch_number + 1] / self.loss_train[
                                                                                                                   :epoch_number + 1],
                     color="green", label="Ratio")

            # else:
            #    plt.plot(loss_train, color = "red", label = "")
            #    plt.plot(loss_valid, color = "blue", label = "")

            plt.xlim(0, EPOCHS)
            plt.ylim(0, 1.1 * np.max(self.loss_valid[:epoch_number + 1]))
            plt.legend()

            plt.title('EPOCH {}   LOSS train {} valid {}'.format(epoch_number + 1, avg_loss, avg_vloss))
            plt.show()

            # Log the running loss averaged per batch
            # for both training and validation
            # writer.add_scalars('Training vs. Validation Loss',
            #                { 'Training' : avg_loss, 'Validation' : avg_vloss },
            #                epoch_number + 1)
            # writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'Modeldump/model_{}_{}'.format(timestamp, epoch_number)
                if (save_intermediate == True):
                    torch.save(self.tinymodel.state_dict(), model_path)

            epoch_number += 1

        ## Save the model:
        self.save_model()
        # torch.save(self.tinymodel.state_dict(), 'Modeldump/final_model')

    def save_model(self):

        ## Save the model:
        torch.save(self.tinymodel.state_dict(), 'Modeldump/final_model')

    def save_averages(self):
        with open('Modeldump/photonmean.txt', 'w') as f:
            print(self.mean_spectrum_transformed, file=f)
        with open('Modeldump/photonvar.txt', 'w') as f:
            print(self.variance_spectrum_transformed, file=f)
        with open('Modeldump/parammean.txt', 'w') as f:
            print(self.mean_param, file=f)
        with open('Modeldump/paramvar.txt', 'w') as f:
            print(self.variance_param, file=f)

    #        np.savetxt("Modeldump/photonmean.txt", self.mean_spectrum_transformed)
    #        np.savetxt("Modeldump/photonvar.txt", self.variance_spectrum_transformed)
    #        np.savetxt("Modeldump/parammean.txt", self.mean_param)
    #        np.savetxt("Modeldump/paramvar.txt", self.variance_param)

    def load_trained_model(self):

        self.tinymodel.load_state_dict(torch.load('Modeldump/final_model'))
        #        self.tinymodel.load_state_dict(torch.load('/home/cayley38/Desktop/Thesis/Programmation/2023_Neural_net/Database/final_model_V1'))

        self.tinymodel.eval()


## ELECTRONS:
class Soprano_SSC_electron(SSC_data):

    def __init__(self, fname, fraction_training=0.8, data_batch_size=32, which_transform="normal", \
                 single_average=False, smoothed_version=1, update_coeff=False, coeff_reconstruct={}):

        """
        This class provides all necessary elements to load the data from disk, clean them, normalise them, and split them into training set and validation set.

        Arguments:
            1- fname: h5 file containing the model
            2- fraction_training = 0.8: which fraction of the data is used in the training set. 1 - fraction_training will be used for the validation set.
            3- data_batch_size: batchsize of the data to be fed to the learning algorythm
            4- which_transform: changes the method for the transformation of the data, and as such the output layers. Valid choices are:
                (i) "normal": corresponding to the assumption of independence of all outputs
                (ii) "normalenhanced": normal + fit for correlation between outputs
            5- order: order to which to perform the Legendre decomposition.
            6- index at which to cut the low energy. Valid for which_transform == legendre
            7- smoothed version: 1 for derivative order 2, 2 for derivative order 6, 3 for derivative order 8, 4 for derivative order 8 + 2, 5 for derivative order 8 + 2  and second derivative order 4.
            8a- update_coeff: True if to update the coefficient of the derivative. False otherwise
            8b- coeff_reconstruct: dictionnary with coeff_der_2, coeff_der_8, coeff_der_2_4 containing the values of the coefficients to put in front of the derivatives
        """
        super().__init__(fname)

        self.__filter_bad_data()

        self.N = len(self.f_filtered)

        self.dim = len(self.energy_grid_electron)

        self.transform = which_transform

        self.fraction_training = fraction_training

        self.batchsize = data_batch_size

        self.single_average = single_average

        self.smoothed_version = smoothed_version

        self.coeff_reconstruct = coeff_reconstruct

        if which_transform == "normal":
            raise TypeError("transform type == normal is not implemented yet")


        elif which_transform == "normalenhanced":
            self.__initialise_smoothed_learning(fname, fraction_training, data_batch_size)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #            self.tinymodel = SimpleModelSmoothed(data_batch_size, device, self.smoothed_version )
            self.tinymodel = ElectronModelConvSmoothed(data_batch_size, device, self.smoothed_version)

        else:
            raise TypeError("Unknown transform type")

        self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.0001, momentum_decay=0.004,
                                           weight_decay=0.0)
        #        self.optimizer = torch.optim.Adadelta(self.tinymodel.parameters() )
        #        self.loss_fn = torch.nn.HuberLoss( reduction='sum', delta = 0.5 )

        #        self.loss_fn = self.cauchy_loss()
        self.loss_fn = torch.nn.L1Loss(reduction='sum')

    #        self.loss_fn = torch.nn.MSELoss( reduction='sum' )

    #    def __initialise_normal_learning(self, fname, fraction_training, data_batch_size):
    #
    #
    #        print("Beginning data transform for the photon spectrum")
    #        self.full_spectrum_transformed = self.__transform_data(self.f_filtered, self.energy_grid_photon)
    #        print("End data transform for the photon spectrum")
    #        print("Beginning detrending for the photon spectrum")
    #        self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(self.full_spectrum_transformed, "parameters")
    #        print("End data detrending for the photon spectrum")
    #        print("Beginning detrending parameters")
    #        self.params_detrended, self.mean_param, self.variance_param = self.__detrend(self.param_filtered, "parameters")
    #        print("End data detrending")
    #
    #
    #        full_spectrum = self.full_spectrum_transformed_detrended
    #        full_param = self.params_detrended
    #        nb_spectrum = len(full_spectrum)
    #
    #        ## Set the generator for reproducibility:
    #        generator1 = torch.Generator().manual_seed(42)
    #
    #        ## Set the indices of the split:
    #        A = torch.utils.data.random_split(full_spectrum, [fraction_training, 1.0 - fraction_training], generator=generator1)
    #
    #
    #        ## Sort the indices:
    #        A_sorted_train = np.sort(A[0].indices)
    #        A_sorted_valid = np.sort(A[1].indices)
    #
    #        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
    #        self.parameter_train = np.array(full_param)[A_sorted_train]
    #
    #        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
    #        self.parameter_valid = np.array(full_param)[A_sorted_valid]
    #
    #
    #        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
    #        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)
    #
    #
    #        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=data_batch_size, shuffle=True, drop_last = True)
    #        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=data_batch_size, shuffle=True, drop_last = True)

    def __initialise_smoothed_learning(self, fname, fraction_training, data_batch_size):

        print("Beginning data transform for the photon spectrum")
        self.full_spectrum_transformed = self.__transform_data(self.f_filtered, self.energy_grid_electron)
        print("End data transform for the photon spectrum")

        print("Beginning detrending for the photon spectrum")

        if self.single_average == False:
            self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(
                self.full_spectrum_transformed, "parameters")
        else:
            self.full_spectrum_transformed_detrended, self.mean_spectrum_transformed, self.variance_spectrum_transformed = self.__detrend(
                self.full_spectrum_transformed, "model")
        print("End data detrending for the photon spectrum")

        print("Add summs (for now only by 3)")
        self.full_spectrum_transformed_add = self.__add_info_smoothed()
        self.full_spectrum_transformed_add = self.enhanced_output
        print("End Add summs: array self.enhanced_output was created")

        print("Beginning detrending parameters")
        self.params_detrended, self.mean_param, self.variance_param = self.__detrend(self.param_filtered, "parameters")
        print("End data detrending")

        full_spectrum = self.enhanced_output
        full_param = self.params_detrended
        nb_spectrum = len(full_spectrum)

        ## Set the generator for reproducibility:
        generator1 = torch.Generator().manual_seed(42)

        ## Set the indices of the split:
        A = torch.utils.data.random_split(full_spectrum, [fraction_training, 1.0 - fraction_training],
                                          generator=generator1)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)

        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        self.parameter_train = np.array(full_param)[A_sorted_train]

        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        self.parameter_valid = np.array(full_param)[A_sorted_valid]

        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)

        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=data_batch_size, shuffle=True,
                                      drop_last=True)
        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=data_batch_size, shuffle=True,
                                            drop_last=True)

        self.save_averages()
        print("Averages were saved")

    def recomputed_smoothed_learning(self, coeff_reconstruct):

        self.coeff_reconstruct = coeff_reconstruct
        print("Add summs (for now only by 3)")
        self.full_spectrum_transformed_add = self.__add_info_smoothed()
        self.full_spectrum_transformed_add = self.enhanced_output
        print("End Add summs: array self.enhanced_output was created")

        full_spectrum = self.enhanced_output
        full_param = self.params_detrended
        nb_spectrum = len(full_spectrum)

        ## Set the generator for reproducibility:
        generator1 = torch.Generator().manual_seed(42)

        ## Set the indices of the split:
        A = torch.utils.data.random_split(full_spectrum, [self.fraction_training, 1.0 - self.fraction_training],
                                          generator=generator1)

        ## Sort the indices:
        A_sorted_train = np.sort(A[0].indices)
        A_sorted_valid = np.sort(A[1].indices)

        self.spectrum_train = np.array(full_spectrum)[A_sorted_train]
        self.parameter_train = np.array(full_param)[A_sorted_train]

        self.spectrum_valid = np.array(full_spectrum)[A_sorted_valid]
        self.parameter_valid = np.array(full_param)[A_sorted_valid]

        self.Soprano_SSC_Dataset_train = SopranoPhotonDataset(self.parameter_train, self.spectrum_train)
        self.Soprano_SSC_Dataset_valid = SopranoPhotonDataset(self.parameter_valid, self.spectrum_valid)

        self.data_loader = DataLoader(self.Soprano_SSC_Dataset_train, batch_size=self.batchsize, shuffle=True,
                                      drop_last=True)
        self.validation_loader = DataLoader(self.Soprano_SSC_Dataset_valid, batch_size=self.batchsize, shuffle=True,
                                            drop_last=True)

    def zero_condition(self, x):
        if x > 0:
            return np.log10(x)
        else:
            return -350.0

    def __transform_data(self, data, energy_grid):
        result = []
        for idx in range(len(data)):
            A = [self.zero_condition(
                data[idx][i] * (10 ** energy_grid[i]) * (10 ** energy_grid[i]) / np.sqrt((10 ** energy_grid[i]))) for i
                in range(len(energy_grid))]
            A = A - np.max(A) + 40
            B = -50.0 * np.ones(len(energy_grid))
            B[A > 0] = np.log10(data[idx][A > 0])
            if (np.inf in B or -np.inf in B):
                print(B, idx)
            result.append(B)

        return result

    def __add_info_smoothed(self):

        ## I want to transform self.full_spectrum_transformed_detrended
        ## Derivative order 8
        #        self.enhanced_output = np.zeros((self.N, 292))
        ## Derivative order 8 + 2
        if self.smoothed_version == 4:
            raise TypeError(" Not implemented ")
        ## Derivative order 8 + 2 and 2nd derivative order 4:
        if self.smoothed_version == 5:
            self.enhanced_output = np.zeros((self.N, 4 * self.dim - 14))

        for i in range(self.N):

            for j in range(self.dim):
                self.enhanced_output[i][j] = self.full_spectrum_transformed_detrended[i][j]

            ##  Derivative order 8 and 2 and 2nd derivative order 4:
            if self.smoothed_version == 5:
                coeff_der_2 = self.coeff_reconstruct["coeff_der_2"]
                coeff_der_8 = self.coeff_reconstruct["coeff_der_8"]
                coeff_der_2_4 = self.coeff_reconstruct["coeff_der_2_4"]
                for j in range(4, self.dim - 4, 1):
                    self.enhanced_output[i][j + self.dim - 4] = coeff_der_8 * 2.0 / 280.0 * \
                                                                self.full_spectrum_transformed_detrended[i][j - 4] \
                                                                - coeff_der_8 * 2.0 * (4.0 / 105.0) * \
                                                                self.full_spectrum_transformed_detrended[i][j - 3] \
                                                                + coeff_der_8 * 2.0 * (1.0 / 5.0) * \
                                                                self.full_spectrum_transformed_detrended[i][j - 2] \
                                                                - coeff_der_8 * 2.0 * (4.0 / 5.0) * \
                                                                self.full_spectrum_transformed_detrended[i][j - 1] \
                                                                + coeff_der_8 * 2.0 * (4.0 / 5.0) * \
                                                                self.full_spectrum_transformed_detrended[i][j + 1] \
                                                                - coeff_der_8 * 2.0 * (1.0 / 5.0) * \
                                                                self.full_spectrum_transformed_detrended[i][j + 2] \
                                                                + coeff_der_8 * 2.0 * (4.0 / 105.0) * \
                                                                self.full_spectrum_transformed_detrended[i][j + 3] \
                                                                - coeff_der_8 * 2.0 / 280 * \
                                                                self.full_spectrum_transformed_detrended[i][j + 4]

                for j in range(1, self.dim - 1, 1):
                    self.enhanced_output[i][j + 2 * self.dim - 8 - 1] = - coeff_der_2 * 10.0 * \
                                                                        self.full_spectrum_transformed_detrended[i][
                                                                            j - 1] \
                                                                        + coeff_der_2 * 10.0 * \
                                                                        self.full_spectrum_transformed_detrended[i][
                                                                            j + 1]

                for j in range(2, self.dim - 2, 1):
                    self.enhanced_output[i][j + 3 * self.dim - 10 - 2] = - coeff_der_2_4 * 1.0 / 12.0 * \
                                                                         self.full_spectrum_transformed_detrended[i][
                                                                             j - 2] \
                                                                         + coeff_der_2_4 * 1.0 * (4.0 / 3.0) * \
                                                                         self.full_spectrum_transformed_detrended[i][
                                                                             j - 1] \
                                                                         - coeff_der_2_4 * 1.0 * (5.0 / 2.0) * \
                                                                         self.full_spectrum_transformed_detrended[i][j] \
                                                                         + coeff_der_2_4 * 1.0 * (4.0 / 3.0) * \
                                                                         self.full_spectrum_transformed_detrended[i][
                                                                             j + 1] \
                                                                         - coeff_der_2_4 * 1.0 / 12.0 * \
                                                                         self.full_spectrum_transformed_detrended[i][
                                                                             j + 2]

    def __detrend(self, data, data_type):

        if self.single_average == False or data_type == "parameters":
            result = np.zeros(np.shape(data))
            mean = np.zeros(len(data[0]))
            variance = np.zeros(len(data[0]))
            for j in range(len(data[0])):
                A = [data[i][j] for i in range(len(data))]
                # print(j, np.mean(  A  ), np.var(  A  ))
                # varA = np.var(  A  )
                meanA = np.mean(A)
                varA = np.max(np.abs(A - meanA))
                # print(j, meanA,  varA)

                mean[j] = meanA
                variance[j] = varA

                for i in range(len(data)):
                    if (varA > 0.0):
                        result[i][j] = (data[i][j] - meanA) / (1.1 * varA)
                    else:
                        result[i][j] = (data[i][j] - meanA)

            return result, mean, variance

        if self.single_average == True and data_type != "parameters":

            result = np.zeros(np.shape(data))
            meanA = np.mean(np.mean(data))
            maxdata = np.max(np.max(data))
            mindata = np.min(np.min(data))
            varA = maxdata - mindata

            for j in range(len(data[0])):

                for i in range(len(data)):
                    if (varA > 0.0):
                        result[i][j] = (data[i][j] - meanA) / (1.1 * varA)
                    else:
                        result[i][j] = (data[i][j] - meanA)

            return result, meanA, varA

        raise TypeError("I dont know what to do here with the detrend")

    def __filter_bad_data(self):

        self.f_filtered = []
        self.param_filtered = []
        for i in range(len(self.f["parameters"])):
            if np.max(self.f["values"]["output_1"][i]) > 0.0:
                self.f_filtered.append(np.array(self.f["values"]["output_1"][i]))
                self.param_filtered.append(np.array(self.f["parameters"][i]))

    def compute_spectrum(self, parameters):

        param = [parameters["log_B"],
                 parameters["log_electron_luminosity"],
                 parameters["log_gamma_cut"],
                 parameters["log_gamma_min"],
                 parameters["log_radius"],
                 parameters["lorentz_factor"],
                 parameters["spectral_index"]
                 ]

        # Apply detrend to the physical parameter:
        for j in range(7):

            meanA = self.mean_param[j]
            varA = self.variance_param[j]

            if (varA > 0.0):
                param[j] = (param[j] - meanA) / (1.1 * varA)
            else:
                param[j] = (param[j] - meanA)

        # Compute the model to get fully detrended reconstruction;
        result = torch.detach(self.tinymodel(torch.tensor(param))).numpy()

        for j in range(self.dim):
            if self.single_average == False:
                meanA = self.mean_spectrum_transformed[j]
                varA = self.variance_spectrum_transformed[j]
            else:
                meanA = self.mean_spectrum_transformed
                varA = self.variance_spectrum_transformed

            if (varA > 0.0):
                result[j] = (result[j] * (1.1 * varA) + meanA)
            else:
                result[j] = (result[j] + meanA)

        return result

    def compute_nuFnu_spectrum(self, parameters):

        Nnu = self.compute_spectrum(parameters)

        nuFnu = np.zeros(self.dim)
        for i in range(self.dim):
            nuFnu[i] = 10 ** Nnu[i] * (10 ** self.energy_grid_electron[i]) * (
                    10 ** self.energy_grid_electron[i]) / np.sqrt((10 ** self.energy_grid_electron[i]))
        return nuFnu

    def __train_one_epoch(self, epoch_index, device):  # , tb_writer):

        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.data_loader):
            # Every data instance is an input + label pair
            parameters, spectrum = data

            parameters_gpu = parameters.to(device)
            spectrum_gpu = spectrum.to(device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.tinymodel(parameters_gpu)

            # Compute the loss and its gradients
            # print("outputs = ", outputs)
            # print("spectrum_gpu = ", spectrum_gpu)
            loss = self.loss_fn(outputs, spectrum_gpu)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 20 == 0:
                last_loss = running_loss / 100  # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.data_loader) + i + 1
                # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def time_update_coeff(self, x):

        #        from 0 to 500 keep all to 0:
        #        from 500 to 1500 increase second order only to 1
        #        from 1500 to 2500 increase 8th order to 1
        #        from 2500 to 3000 increase 2nd derivative to 1

        reconstruct = {"coeff_der_2": 1.0, "coeff_der_8": 1.0, "coeff_der_2_4": 4.0}
        #        if x < 500:
        #            reconstruct = {"coeff_der_2":0.0, "coeff_der_8":0.0,"coeff_der_2_4":0.0}
        #        elif x < 1500:
        #            reconstruct["coeff_der_2"] = (x-500.0)/1000.0
        #            reconstruct["coeff_der_8"] = 0.0
        #            reconstruct["coeff_der_2_4"] = 0.0
        #        elif x < 2500:
        #            reconstruct["coeff_der_2"] = 1.0
        #            reconstruct["coeff_der_8"] = 2.0*(1.0*x-1500.0)/1000.0
        #            reconstruct["coeff_der_2_4"] = 0.0
        #        elif x < 3500:
        #            reconstruct["coeff_der_2"] = 1.0
        #            reconstruct["coeff_der_8"] = 2.0
        #            reconstruct["coeff_der_2_4"] = 3.0*(x-2500.0)/1000.0
        #        else:
        #            reconstruct["coeff_der_2"] = 1.0
        #            reconstruct["coeff_der_8"] = 2.0
        #            reconstruct["coeff_der_2_4"] = 3.0

        return reconstruct

    def train(self, EPOCHS=50, save_intermediate=False, decaying_lr=False):
        import matplotlib.pyplot as plt # noqa
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tinymodel.to(device)

        self.loss_train = np.zeros(EPOCHS)
        self.loss_valid = np.zeros(EPOCHS)

        best_vloss = 1_000_000.

        plt.ion()
        fig = plt.figure()
        plt.show()

        plt.xlim(0, EPOCHS)
        plt.ylim(0, 50)
        time_coefficients = {"coeff_der_2": 0.0, "coeff_der_8": 0.0, "coeff_der_2_4": 0.0}
        self.coeff_reconstruct = time_coefficients

        if decaying_lr == True:
            self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.001, momentum_decay=0.04,
                                               weight_decay=0.1)

        for epoch in range(EPOCHS):
            # print('EPOCH {}:'.format(epoch_number + 1))

            if decaying_lr == True and epoch == 25:
                self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.0001, momentum_decay=0.004,
                                                   weight_decay=0.0)

            if decaying_lr == True and epoch == 75:
                self.optimizer = torch.optim.NAdam(self.tinymodel.parameters(), lr=0.00001, momentum_decay=0.002,
                                                   weight_decay=0.0)

            if (epoch == 0):
                time_coefficients = self.time_update_coeff(epoch)
                self.recomputed_smoothed_learning(time_coefficients)
                self.tinymodel.update_derivative_matrix(time_coefficients["coeff_der_2"],
                                                        time_coefficients["coeff_der_8"],
                                                        time_coefficients["coeff_der_2_4"])

            #            if(epoch%50 == 0  and epoch > 498):
            #                time_coefficients = self.time_update_coeff(epoch)
            #                self.recomputed_smoothed_learning(time_coefficients )
            #                self.tinymodel.update_derivative_matrix(time_coefficients["coeff_der_2"], time_coefficients["coeff_der_8"], time_coefficients["coeff_der_2_4"] )

            # Make sure gradient tracking is on, and do a pass over the data
            self.tinymodel.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, device)  # , writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.tinymodel.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    vinputs_GPU = vinputs.to(device)
                    vlabels_GPU = vlabels.to(device)
                    voutputs = self.tinymodel(vinputs_GPU)
                    vloss = self.loss_fn(voutputs, vlabels_GPU)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            # print('EPOCH {}   LOSS train {} valid {}'.format(epoch_number + 1, avg_loss, avg_vloss))
            self.loss_train[epoch] = avg_loss
            self.loss_valid[epoch] = avg_vloss

            # clear_output(wait=True)

            print("Coeffs = ", time_coefficients["coeff_der_2"], time_coefficients["coeff_der_8"],
                  time_coefficients["coeff_der_2_4"])
            # if(epoch_number == 0):
            plt.plot(self.loss_train[:epoch_number + 1], color="red", label="Training set")
            plt.plot(self.loss_valid[:epoch_number + 1], color="blue", label="Validation set")

            plt.plot((0.3 * self.loss_valid[0] / (self.loss_valid[0] / self.loss_train[0])) * self.loss_valid[
                                                                                              :epoch_number + 1] / self.loss_train[
                                                                                                                   :epoch_number + 1],
                     color="green", label="Ratio")

            # else:
            #    plt.plot(loss_train, color = "red", label = "")
            #    plt.plot(loss_valid, color = "blue", label = "")

            plt.xlim(0, EPOCHS)
            plt.ylim(0, 1.1 * np.max(self.loss_valid[:epoch_number + 1]))
            plt.legend()

            plt.title('EPOCH {}   LOSS train {} valid {}'.format(epoch_number + 1, avg_loss, avg_vloss))
            plt.show()

            # Log the running loss averaged per batch
            # for both training and validation
            # writer.add_scalars('Training vs. Validation Loss',
            #                { 'Training' : avg_loss, 'Validation' : avg_vloss },
            #                epoch_number + 1)
            # writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'Modeldump/model_{}_{}'.format(timestamp, epoch_number)
                if (save_intermediate == True):
                    torch.save(self.tinymodel.state_dict(), model_path)

            epoch_number += 1

        ## Save the model:
        self.save_model()
        # torch.save(self.tinymodel.state_dict(), 'Modeldump/final_model')

    def save_model(self):

        ## Save the model:
        torch.save(self.tinymodel.state_dict(), 'Modeldump/Electron_final_model')

    def save_averages(self):
        with open('Modeldump/electron_spectrum_mean.txt', 'w') as f:
            print(self.mean_spectrum_transformed, file=f)
        with open('Modeldump/electron_spectrumvar.txt', 'w') as f:
            print(self.variance_spectrum_transformed, file=f)

    #        with open('Modeldump/parammean.txt', 'w') as f:
    #            print(self.mean_param, file=f)
    #        with open('Modeldump/paramvar.txt', 'w') as f:
    #            print(self.variance_param, file=f)

    #        np.savetxt("Modeldump/photonmean.txt", self.mean_spectrum_transformed)
    #        np.savetxt("Modeldump/photonvar.txt", self.variance_spectrum_transformed)
    #        np.savetxt("Modeldump/parammean.txt", self.mean_param)
    #        np.savetxt("Modeldump/paramvar.txt", self.variance_param)

    def load_trained_model(self):

        self.tinymodel.load_state_dict(torch.load('Modeldump/Electron_final_model'))

        self.tinymodel.eval()
