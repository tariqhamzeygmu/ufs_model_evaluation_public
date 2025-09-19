# ---------------------------------------------------------------------------------------------------------------------
#  Filename: datareader.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 19 Sept. 2025
#  Purpose: This is the API endpoint for DataReader object construction.
# ---------------------------------------------------------------------------------------------------------------------

import importlib


def getDataReader(datasource, **kwargs):

    module = importlib.import_module('src.datareader.DataReader_Factory')
    fact = getattr(module, 'DataReader_Factory')

    data_reader = fact.create_DataReader(datasource, **kwargs)

    return data_reader
