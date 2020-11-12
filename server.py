from flask import Flask
from flask_cors import CORS
from functools import reduce
import h5py
import json
import numpy as np
from datetime import datetime

path = "db/test_dataset_1.hdf5"

app = Flask(__name__)
CORS(app)
INTERNAL = u'internal'
TIME = u'time'
MEASUREMENT = u'measurement'
GLUCOSE = u'glucose'


def glucose_mapper(times_collection, glucose_collection):
    def mapper(time, glucose):
        return {
            'date': datetime.fromtimestamp(time[0]).strftime("%d %b, %Y"),
            TIME: datetime.fromtimestamp(time[0]).strftime("%H:%M:%S"),
            GLUCOSE: glucose[0]
        }

    data = map(mapper, times_collection, glucose_collection)
    return list(data)


def measurement_mapper_separate(times_collection, measurement_collection):
    def mapper(measurements, date):
        return {
            TIME: datetime.fromtimestamp(date[0]).strftime("%d %b, %Y, %H:%M:%S"),
            'ms': [{'i': i, 'm': measurement} for i, measurement in enumerate(measurements)]
        }

    data = map(mapper, measurement_collection, times_collection)
    return list(data)


def measurement_mapper_all(measurement_collection):
    data_num = [
        *measurement_collection
    ]
    data_tr = [*np.transpose(data_num).tolist()]
    return [{'i': i, 'm': m} for i, m in enumerate(data_tr)]


def get_data(data_key):
    with h5py.File(path, "r") as f:
        def func(res, key):
            res[key] = f[INTERNAL][key][()]
            return res

        data = reduce(func, data_key, {})
    return data


@app.route(f"/api/{MEASUREMENT}-separate", methods=['GET'])
def measurement_separate_route():
    data = get_data([TIME, MEASUREMENT])
    return json.dumps(measurement_mapper_separate(data[TIME], data[MEASUREMENT]))


@app.route(f"/api/{GLUCOSE}", methods=['GET'])
def glucose_route():
    data = get_data([TIME, GLUCOSE])
    return json.dumps(glucose_mapper(data[TIME], data[GLUCOSE]))


@app.route(f"/api/{MEASUREMENT}-all", methods=['GET'])
def measurement_all_route():
    data = get_data([MEASUREMENT])
    return json.dumps(measurement_mapper_all(data[MEASUREMENT]))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4567)
