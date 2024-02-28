from . import use_model as um
from .cosmocalc import cosmocalc


def execute_validations(validations):
    for validate, param in validations:
        valid, message = validate(param)
        if not valid:
            return False, message
    return True, "Success"


def initialize_model(*, ebl, model_path, inference_folder, redshift):
    model = um.SSC(do_ebl=ebl, z=redshift, on_gpu=False,
                   fname=model_path,
                   inference_folder=inference_folder,
                   database=None)
    return model


def get_spectrum(*, parameters: dict, redshift: float, ebl: bool = None,
                 model_path: str = None, inference_folder: str = None,
                 model=None) -> dict:
    dL = cosmocalc(redshift)["DL_cm"]
    if not model:
        model = initialize_model(ebl=ebl, model_path=model_path, inference_folder=inference_folder,
                                 redshift=redshift)
    nu, nu_fnu = model.eval_nuFnu_spectrum(parameters, z=redshift, dL=dL)
    nu = nu[nu_fnu > 0]
    nu_fnu = nu_fnu[nu_fnu > 0]
    return {"nu": nu.tolist(), "nuFnu": nu_fnu.tolist()}


def fcube_linear(x, tmin, tmax):
    return (tmax - tmin) * x + tmin
