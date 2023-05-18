import os
import joblib
import numpy as np

from azureml.core import Model
from azureml.monitoring import ModelDataCollector
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


def init():
    global xgbc_model, rfc_model
    global inputs_dc, xgbc_prediction_dc, rfc_prediction_dc

    model_path = os.environ.get("AZUREML_MODEL_DIR")
    xgbc_model_path = os.path.join(model_path, "xbgc_model_file.pkl")
    rfc_model_path = os.path.join(model_path, "rfc_model_file.pkl")

    xgbc_model = joblib.load(xgbc_model_path)
    rfc_model = joblib.load(rfc_model_path)

    input_features = ["feat{}".format(i) for i in range(1, 33)]
    inputs_dc = ModelDataCollector("sample-model", designation="inputs", feature_names=input_features)
    xgbc_prediction_dc = ModelDataCollector("sample-model", designation="xgbc_predictions", feature_names=["xgbc_prediction"])
    rfc_prediction_dc = ModelDataCollector("sample-model", designation="rfc_predictions", feature_names=["rfc_prediction"])


@input_schema('data', NumpyParameterType(np.array([[1,5,0,0,0,2,0,0,0,0,1,0,4,11,6,25,28,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,4,2,1]])))
@output_schema(StandardPythonParameterType({'xgbc_predict': [['Autres']], 'rfc_predict': [['Méfait volontaire']]}))
def run(data):
    xgbc_result = xgbc_model.predict(data)
    rfc_result = rfc_model.predict(data)

    inputs_dc.collect(data)
    xgbc_prediction_dc.collect(xgbc_result)
    rfc_prediction_dc.collect(rfc_result)

    return { "xgbc_predict": xgbc_result.tolist(), "rfc_predict": rfc_result.tolist() }
