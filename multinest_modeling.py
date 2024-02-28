import argparse
import csv
import logging
import os
import sys
import time
import traceback

import numpy as np
import pymultinest
import requests
from astropy.table import Table
import json

from files.utils import get_spectrum, initialize_model, fcube_linear
from files.constants import (
    X_RAY_LIMIT,
    PARAMETER_NAMES,
    etolerance,
    N_POINT_SAMPLER,
    PARAM_MIN,
    PARAM_MAX,
    PARAMETER_NAMES_FOR_SPECTRUM_EVAL
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    NUFNU_LIMIT = 1e-19  # or any other value you need

    # Set API_BASE_URL based on DEBUG mode
    if DEBUG:
        API_BASE_URL = "http://backend:8000/api/tenor/"
    else:
        API_BASE_URL = "https://mmdc.am/api/tenor/"

class DataHandler:
    def __init__(self, uuid):
        self.uuid = uuid
        self.api_url = Config.API_BASE_URL + f"{uuid}/"

    def read_data(self):
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()

            data = response.json()
            # Process data
            if 'file_url' in data:
                filename = self.download_file(data['file_url'])
                self.process_data(filename)
                self.rename_csv_columns(filename, ['x', 'y', 'dy'])

            return {
                'ebl_value': data['ebl_value'],
                'z_value': data['z_value'],
                'data': filename if 'file_url' in data else None
            }

        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"HTTP error occurred: {http_err} - UUID: {self.uuid}")
        except requests.exceptions.RequestException as req_err:
            raise ValueError(f"Request error occurred: {req_err} - UUID: {self.uuid}")
        except json.JSONDecodeError as json_err:
            raise ValueError(f"JSON decode error: {json_err} - Response was: '{response.text}' - UUID: {self.uuid}")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = (
                f"An unexpected error occurreddd: {e.__class__.__name__}: {e.args} - UUID: {self.uuid}\n"
                f"Traceback:\n{tb_str}"
            )
            raise ValueError(error_message)

    def process_data(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            # Skip the header
            next(reader, None)

            data = []
            for row in reader:
                # Ensure row has enough columns
                if len(row) >= 3 and row[2] != '0':  # Assuming the 'flux_err' is in the third column (index 2)
                    # Append a dictionary with values converted as needed
                    data.append({'frequency': row[0], 'flux': row[1], 'flux_err': row[2]})

        # Re-write the processed data back to the file
        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['frequency', 'flux', 'flux_err'])
            writer.writeheader()
            writer.writerows(data)

    def post_results(self, results):
        try:
            response = requests.post(self.api_url, json=results)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"HTTP error occurred: {http_err} - UUID: {self.uuid}")
        except requests.exceptions.RequestException as req_err:
            raise ValueError(f"Request error occurred: {req_err} - UUID: {self.uuid}")
        except json.JSONDecodeError as json_err:
            raise ValueError(f"JSON decode error: {json_err} - Response was: '{response.text}' - UUID: {self.uuid}")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = (
                f"An unexpected error occurreddd: {e.__class__.__name__}: {e.args} - UUID: {self.uuid}\n"
                f"Traceback:\n{tb_str}"
            )
            raise ValueError(error_message)

    def rename_csv_columns(self, filename, new_columns):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

        if data:
            data[0] = new_columns

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def download_file(self, file_url):
        response = requests.get(file_url)
        if response.status_code == 200:
            filename = f"{self.uuid}.csv"
            with open(filename, 'wb') as file:
                file.write(response.content)

            return filename
        else:
            raise ValueError("Failed to download the file")


class ModelInitializer:
    def __init__(self, ebl, redshift):


        self.ebl = ebl
        self.redshift = redshift

    def initialize(self):
        return initialize_model(ebl=self.ebl, model_path=os.getenv("MODEL_PATH"),
                                inference_folder=os.getenv("INFERENCE_PATH"),
                                redshift=self.redshift)


class MultinestRunner:
    def __init__(self, data, model, ebl, redshift):
        self.data = data
        self.model = model
        self.ebl = ebl
        self.redshift = redshift

    def run(self):

        def prior_(cube, ndim, nparams):
            for i in range(7):
                cube[i] = fcube_linear(cube[i], PARAM_MIN[i], PARAM_MAX[i])

        def loglike_(cube, ndim, nparams):
            params = dict(zip(PARAMETER_NAMES_FOR_SPECTRUM_EVAL, cube))
            spectrum = get_spectrum(parameters=params, ebl=self.ebl, redshift=self.redshift, model=self.model)
            nu_m2, nu_fnu = spectrum["nu"], spectrum["nuFnu"]
            model_comp = np.interp(np.log10(self.data['x'][self.data['x'] > X_RAY_LIMIT]),
                                   np.log10(nu_m2), np.log10(nu_fnu))
            return -0.5 * np.sum((np.log10(self.data['y'][self.data['x'] > X_RAY_LIMIT]) - model_comp) ** 2 /
                                 (self.data['dy'][self.data['x'] > X_RAY_LIMIT] /
                                  (np.log(10) * self.data['y'][self.data['x'] > X_RAY_LIMIT])))

        output_file_path = "output_file"

        pymultinest.run(
            loglike_,
            prior_,
            len(PARAMETER_NAMES_FOR_SPECTRUM_EVAL),
            outputfiles_basename=output_file_path,
            resume=False,
            verbose=False,
            n_live_points=N_POINT_SAMPLER,
            evidence_tolerance=etolerance
        )
        return pymultinest.Analyzer(outputfiles_basename=output_file_path,
                                    n_params=len(PARAMETER_NAMES_FOR_SPECTRUM_EVAL))


class ResultHandler:
    def __init__(self, analysis, ebl, redshift, model):
        self.analysis = analysis
        self.ebl = ebl
        self.redshift = redshift
        self.model = model

    @staticmethod
    def filter_spectrum(spectrum):
        filtered_spectrum = {"nu": [], "nuFnu": []}
        for nu, nuFnu in zip(spectrum["nu"], spectrum["nuFnu"]):
            if nuFnu > Config.NUFNU_LIMIT:
                filtered_spectrum["nu"].append(nu)
                filtered_spectrum["nuFnu"].append(nuFnu)
        return filtered_spectrum

    def process_results(self):
        params = self.analysis.get_equal_weighted_posterior()[::10, :-1]
        bestresult = self.analysis.get_best_fit()

        multinest_results = {}
        for index, param_values in enumerate(params):
            param_dict = dict(zip(PARAMETER_NAMES_FOR_SPECTRUM_EVAL, param_values))
            spectrum = get_spectrum(parameters=param_dict, ebl=self.ebl, redshift=self.redshift, model=self.model)
            filtered_spectrum = self.filter_spectrum(spectrum)
            if filtered_spectrum["nu"]:
                multinest_results[str(index)] = filtered_spectrum

        best_params = dict(zip(PARAMETER_NAMES_FOR_SPECTRUM_EVAL, bestresult['parameters']))
        best_spectrum = get_spectrum(parameters=best_params, redshift=self.redshift, model=self.model)
        best_filtered_spectrum = self.filter_spectrum(best_spectrum)
        if best_filtered_spectrum["nu"]:
            multinest_results.update({"best": best_filtered_spectrum})

        analysis_statistics = self.analysis.get_stats()
        best_parameters = {
            name: {"value": bestresult['parameters'][i],
                   "error": analysis_statistics['modes'][0]['sigma'][i]
                   } for i, name in enumerate(PARAMETER_NAMES)
        }

        equal_weighted_posterior_values = self.analysis.get_equal_weighted_posterior()[:, :-1].tolist()

        return {
            "data": multinest_results,
            "best_parameters": best_parameters,
            "equal_weighted_posterior": equal_weighted_posterior_values
        }


class MultinestModeling:
    def __init__(self, uuid):
        self.uuid = uuid

    def run(self):
        try:
            logging.info(f"Starting Multinest modeling for UUID: {self.uuid}")
            # Create an instance of DataHandler
            data_handler = DataHandler(self.uuid)

            batch_data = data_handler.read_data()
            # Load data from CSV file
            data = Table.read(batch_data['data'], format='csv')
            ebl = batch_data['ebl_value']
            redshift = batch_data['z_value']

            model_initializer = ModelInitializer(ebl, redshift)
            model = model_initializer.initialize()
            logging.info("Model initialized successfully")

            runner = MultinestRunner(data, model, ebl, redshift)
            analysis = runner.run()
            logging.info("Multinest completed successfully")

            result_handler = ResultHandler(analysis, ebl, redshift, model)
            results = result_handler.process_results()
            logging.info("Result handling completed successfully")

            # Post the results to the server
            post_response = data_handler.post_results(results)
            logging.info(f"Results posted successfully: {post_response}")

            return True

        except Exception as e:
            logging.error(f"Error during modeling: {e}")
            return False

    def debug_run(self):
        try:
            logging.info(f"Starting DEBUG RUN")

            data_handler = DataHandler(self.uuid)

            batch_data = data_handler.read_data()

            # Read results from response.json
            inference_path = os.getenv("INFERENCE_PATH")
            with open(os.path.join(inference_path, "response.json"), "r") as file:
                results = json.load(file)
            time.sleep(20)
            logging.info("Result handling completed successfully")

            # Post the results to the server
            data_handler = DataHandler(self.uuid)
            post_response = data_handler.post_results(results)
            logging.info(f"Results posted successfully: {post_response}")

            return True

        except Exception as e:
            logging.error(f"Error during modeling: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", type=str, required=True)
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    args = parser.parse_args()

    try:
        modeling = MultinestModeling(args.uuid)
        result = modeling.debug_run() if args.debug else modeling.run()
        exit_code = 0 if result else 1
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        exit_code = 1

    sys.exit(exit_code)
