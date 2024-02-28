import csv
import torch
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

from .data_utils import SSC_data
from .learning_utils import SimpleModelConvSmoothed
from .electron_utils import ElectronModelConvSmoothed


c = 2.99792458e10  # cm s^-1
Hz_to_erg = 6.62606957030463e-27  # erg
HztoeV  = 0.413566553853599e-14

delta_nu_photon = [4.96171763e-03, 8.10896411e-03, 1.32525274e-02, 2.16586830e-02,
       3.53969122e-02, 5.78493804e-02, 9.45435804e-02, 1.54513126e-01,
       2.52521704e-01, 4.12697697e-01, 6.74474257e-01, 1.10229722e+00,
       1.80149079e+00, 2.94418692e+00, 4.81170188e+00, 7.86379250e+00,
       1.28518420e+01, 2.10038405e+01, 3.43266993e+01, 5.61003252e+01,
       9.16850892e+01, 1.49841477e+02, 2.44886801e+02, 4.00219930e+02,
       6.54081770e+02, 1.06896966e+03, 1.74702336e+03, 2.85517048e+03,
       4.66622180e+03, 7.62603355e+03, 1.24632712e+04, 2.03687969e+04,
       3.32888438e+04, 5.44041520e+04, 8.89130234e+04, 1.45311074e+05,
       2.37482737e+05, 3.88119424e+05, 6.34305843e+05, 1.03664974e+06,
       1.69420273e+06, 2.76884543e+06, 4.52514026e+06, 7.39546317e+06,
       1.20864487e+07, 1.97529537e+07, 3.22823676e+07, 5.27592620e+07,
       8.62247701e+07, 1.40917646e+08, 2.30302532e+08, 3.76384773e+08,
       6.15127833e+08, 1.00530701e+09, 1.64297912e+09, 2.68513038e+09,
       4.38832427e+09, 7.17186399e+09, 1.17210192e+10, 1.91557302e+10,
       3.13063218e+10, 5.11641046e+10, 8.36177951e+10, 1.36657051e+11,
       2.23339417e+11, 3.65004914e+11, 5.96529664e+11, 9.74911916e+11,
       1.59330424e+12, 2.60394643e+12, 4.25564486e+12, 6.95502526e+12,
       1.13666384e+13, 1.85765636e+13, 3.03597864e+13, 4.96171763e+13,
       8.10896411e+13, 1.32525274e+14, 2.16586830e+14, 3.53969122e+14,
       5.78493804e+14, 9.45435804e+14, 1.54513126e+15, 2.52521704e+15,
       4.12697697e+15, 6.74474257e+15, 1.10229722e+16, 1.80149079e+16,
       2.94418692e+16, 4.81170188e+16, 7.86379250e+16, 1.28518420e+17,
       2.10038405e+17, 3.43266993e+17, 5.61003252e+17, 9.16850892e+17,
       1.49841477e+18, 2.44886801e+18, 4.00219930e+18, 6.54081770e+18,
       1.06896966e+19, 1.74702336e+19, 2.85517048e+19, 4.66622180e+19,
       7.62603355e+19, 1.24632712e+20, 2.03687969e+20, 3.32888438e+20,
       5.44041520e+20, 8.89130234e+20, 1.45311074e+21, 2.37482737e+21,
       3.88119424e+21, 6.34305843e+21, 1.03664974e+22, 1.69420273e+22,
       2.76884543e+22, 4.52514026e+22, 7.39546317e+22, 1.20864487e+23,
       1.97529537e+23, 3.22823676e+23, 5.27592620e+23, 8.62247701e+23,
       1.40917646e+24, 2.30302532e+24, 3.76384773e+24, 6.15127833e+24,
       1.00530701e+25, 1.64297912e+25, 2.68513038e+25, 4.38832427e+25,
       7.17186399e+25, 1.17210192e+26, 1.91557302e+26, 3.13063218e+26,
       5.11641046e+26, 8.36177951e+26, 1.36657051e+27, 2.23339417e+27,
       3.65004914e+27, 5.96529664e+27, 9.74911916e+27, 1.59330424e+28,
       2.60394643e+28, 4.25564486e+28, 6.95502526e+28, 1.13666384e+29,
       1.85765636e+29, 3.03597864e+29]

delta_E_electron = [2.90186041e-01, 3.69355119e-01, 4.70123249e-01, 5.98383121e-01,
       7.61635082e-01, 9.69425738e-01, 1.23390622e+00, 1.57054273e+00,
       1.99902103e+00, 2.54439755e+00, 3.23856468e+00, 4.12211574e+00,
       5.24671878e+00, 6.67813805e+00, 8.50007972e+00, 1.08190868e+01,
       1.37707695e+01, 1.75277357e+01, 2.23096842e+01, 2.83962524e+01,
       3.61433692e+01, 4.60040683e+01, 5.85549811e+01, 7.45300566e+01,
       9.48634810e+01, 1.20744307e+02, 1.53685986e+02, 1.95614874e+02,
       2.48982877e+02, 3.16910834e+02, 4.03371019e+02, 5.13419427e+02,
       6.53491441e+02, 8.31778153e+02, 1.05870537e+03, 1.34754328e+03,
       1.71518246e+03, 2.18312163e+03, 2.77872481e+03, 3.53682152e+03,
       4.50174354e+03, 5.72991733e+03, 7.29316370e+03, 9.28289777e+03,
       1.18154747e+04, 1.50389938e+04, 1.91419592e+04, 2.43643030e+04,
       3.10114160e+04, 3.94720064e+04, 5.02408304e+04, 6.39476243e+04,
       8.13939304e+04, 1.03599969e+05, 1.31864299e+05, 1.67839755e+05,
       2.13630100e+05, 2.71913049e+05, 3.46096857e+05, 4.40519625e+05,
       5.60702983e+05, 7.13674982e+05, 9.08381077e+05, 1.15620724e+06,
       1.47164578e+06, 1.87314284e+06, 2.38417704e+06, 3.03463252e+06,
       3.86254644e+06, 4.91633332e+06, 6.25761623e+06, 7.96483037e+06,
       1.01378098e+07, 1.29036253e+07, 1.64240156e+07, 2.09048451e+07,
       2.66081427e+07, 3.38674241e+07, 4.31071958e+07, 5.48677786e+07,
       6.98369049e+07, 8.88899352e+07, 1.13141048e+08, 1.44008392e+08,
       1.83297021e+08, 2.33304445e+08, 2.96954984e+08, 3.77970778e+08,
       4.81089447e+08, 6.12341137e+08, 7.79401149e+08, 9.92038776e+08,
       1.26268858e+09, 1.60717756e+09, 2.04565064e+09, 2.60374873e+09,
       3.31410815e+09, 4.21826911e+09, 5.36910489e+09, 6.83391377e+09,
       8.69835444e+09, 1.10714552e+10, 1.40919895e+10, 1.79365915e+10,
       2.28300847e+10, 2.90586296e+10, 3.69864573e+10, 4.70771693e+10,
       5.99208475e+10, 7.62685611e+10, 9.70762874e+10, 1.23560815e+11,
       1.57270899e+11, 2.00177830e+11, 2.54790706e+11, 3.24303165e+11,
       4.12780140e+11, 5.25395561e+11, 6.68734925e+11, 8.51180393e+11,
       1.08340096e+12, 1.37897637e+12, 1.75519118e+12, 2.23404561e+12,
       2.84354195e+12, 3.61932220e+12, 4.60675220e+12, 5.86357464e+12,
       7.46328564e+12, 9.49943268e+12]


class SSC(SSC_data):

    def __init__(self, smoothed_version = 5, low_energy_index = 42,
        fname = "/home/cayley38/Desktop/Thesis/Programmation/2023_Neural_net/Database/final_model_V1",
        database = "/home/cayley38/Desktop/Thesis/Programmation/2023_Neural_net/Database/database.h5",
        do_ebl = False,
        z = 0.0,
        on_gpu = True,
        inference_folder=None
        ):



        super().__init__(database, inference_folder=inference_folder)
        """
        This class provides all necessary elements to load the CNN and use it
        """
        self.inference_folder = inference_folder
        self.name = fname
        self.smoothed_version = smoothed_version
        self.low_energy_index = low_energy_index

        device = torch.device("cpu")
        self.tinymodel = SimpleModelConvSmoothed(1, device, self.smoothed_version,  self.low_energy_index)
        ## Load model:
        if on_gpu:
            self.tinymodel.load_state_dict(torch.load(fname))
        else:
            self.tinymodel.load_state_dict(torch.load(fname, map_location=device))

        self.tinymodel.eval()


        ## Load parameters and spectrum transform utils:
        self.mean_spectrum_transformed = np.asarray( np.loadtxt(f"{inference_folder}/photonmean1.txt") )
        self.variance_spectrum_transformed = np.asarray( np.loadtxt(f"{inference_folder}/photonvar1.txt") )

        self.mean_param =np.zeros(7)
        self.variance_param =np.zeros(7)

        with open(f"{inference_folder}/parammean1.txt") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
#                    print(f'The line is {", ".join(row)}')
#                    print(f'\t{row[0]} , {row[1]} ,  {row[2]} ,  {row[3]},  {row[4]},  {row[5]},  {row[6]}.')
                    self.mean_param[0] = row[0]
                    self.mean_param[1] = row[1]
                    self.mean_param[2] = row[2]
                    self.mean_param[3] = row[3]
                    self.mean_param[4] = row[4]
                    self.mean_param[5] = row[5]
                    self.mean_param[6] = row[6]
                    line_count += 1
                else:
                    line_count += 1
#            print(f'Processed {line_count} lines.')


        with open(f"{inference_folder}/paramvar1.txt") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
#                    print(f'The line is {", ".join(row)}')
#                    print(f'\t{row[0]} , {row[1]} ,  {row[2]} ,  {row[3]},  {row[4]},  {row[5]},  {row[6]}.')
                    self.variance_param[0] = row[0]
                    self.variance_param[1] = row[1]
                    self.variance_param[2] = row[2]
                    self.variance_param[3] = row[3]
                    self.variance_param[4] = row[4]
                    self.variance_param[5] = row[5]
                    self.variance_param[6] = row[6]
                    line_count += 1
                else:
                    line_count += 1
#            print(f'Processed {line_count} lines.')

        print("All model components are now loaded")


        self.do_ebl = do_ebl

        if do_ebl == True:
            self.funcEBL = self.init_EBLAbsorption(z)


    def __compute_spectrum(self, parameters):


        param = [parameters["log_B"],
                 parameters["log_electron_luminosity"],
                 parameters["log_gamma_cut"],
                 parameters["log_gamma_min"],
                 parameters["log_radius"],
                 parameters["lorentz_factor"],
                 parameters["spectral_index"]
        ]

        # Apply detrend to the physical parameter:
        for j in range( 7 ):

            meanA = self.mean_param[j]
            varA = self.variance_param[j]

            if( varA > 0.0):
                param[j] = (param[j] - meanA)/(1.1*varA)
            else:
                param[j] = (param[j] - meanA)



        #Compute the model to get fully detrended reconstruction;
        result = torch.detach(self.tinymodel( torch.tensor(param ))).numpy(  )

        for j in range( 150 - self.low_energy_index ):

            meanA = self.mean_spectrum_transformed
            varA = self.variance_spectrum_transformed

            if( varA > 0.0):
                result[j ] = (result[j]*(1.1*varA) + meanA)
            else:
                result[j ] = (result[j] + meanA)

        return result



    def init_EBLAbsorption(self, z):
        self.z = z
        if self.z>4.995 or self.z<0:
            raise ValueError('The redshift is out of range: please give the value in the range of [0,4.995]')
        else:
            r_z = "%.2f" %round(self.z, 2) #round z for EBL
            print( "tau_modelC_total_z{}.dat".format(r_z))#
            #print("tau_modelC_total_z{}".format(r_z))
            EBL_energy, EBL_flux = np.loadtxt(f"{self.inference_folder}/ebl_files/tau_modelC_total_z{r_z}.dat",
                                              unpack= True)
        return InterpolatedUnivariateSpline(EBL_energy*1e12, EBL_flux)


    def eval_nuFnu_spectrum(self, parameters, z = 1, dL = 1):

        """
        1) dL is in Mpc
        2) z is the redshift equal to 1
        """
        Nnu = self.__compute_spectrum(parameters)

        tdyn = 10**parameters["log_radius"]/(c*parameters["lorentz_factor"])

        nuFnu = np.zeros(150)
        for i in range(self.low_energy_index, 150, 1):
            nuFnu[i] = 10**Nnu[i-self.low_energy_index]* (10**self.energy_grid_photon[i]) * (10**self.energy_grid_photon[i])/np.sqrt(delta_nu_photon[i])

        Vb = (4.0 / 3.0) * np.pi * np.power(10.0, 3 * parameters["log_radius"])

        num = Vb*Hz_to_erg*parameters["lorentz_factor"]**4.0
        denom = (tdyn*4.0*np.pi*dL**2)

        if self.do_ebl == True:
            return   parameters["lorentz_factor"]*10**np.asarray(self.energy_grid_photon)/(1.0+z), (num/denom)*np.asarray( nuFnu )*np.exp(-self.funcEBL(parameters["lorentz_factor"]*10**np.asarray(self.energy_grid_photon)/(1.0+z)*HztoeV))
        else:
            return   parameters["lorentz_factor"]*10**np.asarray(self.energy_grid_photon)/(1.0+z), (num/denom)*np.asarray( nuFnu )

    def eval_nuFnu_spectrum_no_EBL(self, parameters, z = 1, dL = 1):

        """
        1) dL is in Mpc
        2) z is the redshift equal to 1
        """
        Nnu = self.__compute_spectrum(parameters)

        tdyn = 10**parameters["log_radius"]/(c*parameters["lorentz_factor"])

        nuFnu = np.zeros(150)
        for i in range(self.low_energy_index, 150, 1):
            nuFnu[i] = 10**Nnu[i-self.low_energy_index]* (10**self.energy_grid_photon[i]) * (10**self.energy_grid_photon[i])/np.sqrt(delta_nu_photon[i])

        Vb = (4.0 / 3.0) * np.pi * np.power(10.0, 3 * parameters["log_radius"])

        num = Vb*Hz_to_erg*parameters["lorentz_factor"]**4.0
        denom = (tdyn*4.0*np.pi*dL**2)

        return   parameters["lorentz_factor"]*10**np.asarray(self.energy_grid_photon)/(1.0+z), (num/denom)*np.asarray( nuFnu )



class SSC_electron(SSC_data):

    def __init__(self, smoothed_version = 5 ,
        fname = "/home/cayley38/Desktop/Thesis/Programmation/2023_Neural_net/Tenor/Modeldump/Electron_final_model",
        database = "/home/cayley38/Desktop/Thesis/Programmation/2023_Neural_net/Database/database.h5",
        on_gpu = True,
        inference_folder="/home/mherkh/ICRANet/Tenor/data/inference",
        ):

        super().__init__(database)
        """
        This class provides all necessary elements to load the CNN and use it
        """
        self.inference_folder = inference_folder
        self.name = fname
        self.smoothed_version = smoothed_version
        self.dim = 130

        device = torch.device("cpu")
        self.tinymodel = ElectronModelConvSmoothed(1, device, self.smoothed_version)

        ## Load model:
        if on_gpu == True:
            self.tinymodel.load_state_dict(torch.load(fname))
        else:
            self.tinymodel.load_state_dict(torch.load(fname, map_location=device))

        self.tinymodel.eval()


        ## Load parameters and spectrum transform utils:
        self.mean_spectrum_transformed = np.asarray( np.loadtxt(f"Modeldump/electron_spectrum_mean.txt") )
        self.variance_spectrum_transformed = np.asarray( np.loadtxt("Modeldump/electron_spectrumvar.txt") )

        self.mean_param =np.zeros(7)
        self.variance_param =np.zeros(7)

        with open("Modeldump/parammean1.txt") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
#                    print(f'The line is {", ".join(row)}')
#                    print(f'\t{row[0]} , {row[1]} ,  {row[2]} ,  {row[3]},  {row[4]},  {row[5]},  {row[6]}.')
                    self.mean_param[0] = row[0]
                    self.mean_param[1] = row[1]
                    self.mean_param[2] = row[2]
                    self.mean_param[3] = row[3]
                    self.mean_param[4] = row[4]
                    self.mean_param[5] = row[5]
                    self.mean_param[6] = row[6]
                    line_count += 1
                else:
                    line_count += 1
#            print(f'Processed {line_count} lines.')


        with open("Modeldump/paramvar1.txt") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
#                    print(f'The line is {", ".join(row)}')
#                    print(f'\t{row[0]} , {row[1]} ,  {row[2]} ,  {row[3]},  {row[4]},  {row[5]},  {row[6]}.')
                    self.variance_param[0] = row[0]
                    self.variance_param[1] = row[1]
                    self.variance_param[2] = row[2]
                    self.variance_param[3] = row[3]
                    self.variance_param[4] = row[4]
                    self.variance_param[5] = row[5]
                    self.variance_param[6] = row[6]
                    line_count += 1
                else:
                    line_count += 1
#            print(f'Processed {line_count} lines.')

        print("All model components are now loaded")



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
        for j in range( 7 ):

            meanA = self.mean_param[j]
            varA = self.variance_param[j]

            if( varA > 0.0):
                param[j] = (param[j] - meanA)/(1.1*varA)
            else:
                param[j] = (param[j] - meanA)



        #Compute the model to get fully detrended reconstruction;
        result = torch.detach(self.tinymodel( torch.tensor(param ))).numpy(  )

        for j in range( self.dim ):

            meanA = self.mean_spectrum_transformed
            varA = self.variance_spectrum_transformed

            if( varA > 0.0):
                result[j ] = (result[j]*(1.1*varA) + meanA)
            else:
                result[j ] = (result[j] + meanA)

        return result



    def compute_electron_spectrum(self, parameters):

        Nnu = self.compute_spectrum(parameters)


        nuFnu = np.zeros(self.dim)
        for i in range(self.dim):
            nuFnu[i] = 10**Nnu[i]* (10**self.energy_grid_electron[i]) * (10**self.energy_grid_electron[i])/np.sqrt((10**self.energy_grid_electron[i]))
        return nuFnu
