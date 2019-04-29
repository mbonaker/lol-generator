#!/usr/bin/env python3
import http.server
import io
import logging
import logging.config
import sys
from http.client import HTTPMessage

import numpy as np
from scipy.special import expit
import pandas as pd
import yaml

import dataprovider
from nn import Generator, WinEstimator
from config import ApplicationConfiguration


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            self.server.logger.debug("Request incoming...")
            self.send_response(200, "Calculate...")
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.server_version = "lfapi/{:d}".format(self.server.config.code_version)
            self.send_header("Access-Control-Allow-Method", "POST")
            self.send_header("Access-Control-Allow-Origin", "http://lnn.softwar3.com")
            self.end_headers()

            self.headers: HTTPMessage
            self.rfile: io.BufferedIOBase
            self.wfile: io.BufferedIOBase
            self.server.interesting_data: dataprovider.DataProvider
            content_length = self.headers['content-length']
            if content_length or not content_length.isnumeric():
                self.server.logger.debug("Prepare input stream...")
                input_stream = io.BytesIO()
                input_stream.write(self.rfile.read(int(content_length)))
                input_stream.write(b'\n')
                input_stream.seek(0, io.SEEK_SET)
                output_stream = io.TextIOWrapper(self.wfile, encoding='utf-8')
                self.server.logger.debug("Get known slice...")
                known_nd = self.server.known_data.load_get(input_stream)
                self.server.logger.debug("Create interesting dp...")
                self.server.logger.debug("Make NaN data...")
                interesting_nd = self.server.interesting_data.get_nan_data(known_nd.shape[0])
                self.server.logger.debug("Write known data to it...")
                interesting_nd[:, self.server.slice_known] = known_nd
                self.server.logger.debug("Predict and write unknown without win...")
                interesting_nd[:, self.server.slice_unknown_without_win] = self.server.generator.predict(known_nd)
                self.server.logger.debug("Predict and write win...")
                interesting_nd[:, self.server.slice_win] = self.server.win_estimator.predict(interesting_nd[:, self.server.slice_interesting_without_win])
                pd.DataFrame(expit(interesting_nd[:, self.server.slice_win][:, 0])).to_csv(output_stream, header=False, index=False)
                self.server.logger.debug("Output probabilities: {!s}".format(expit(interesting_nd[:, self.server.slice_win][:, 0])))
                output_stream.write("\n")
                self.server.logger.debug("Print...")
                self.server.interesting_data.write_as_csv(output_stream, nd=interesting_nd)
            self.server.logger.debug("Done.")
        except:
            self.server.logger: logging.Logger
            self.server.logger.exception("Server error!")


class Server(http.server.HTTPServer):
    def __init__(self, addr, config: ApplicationConfiguration):
        super().__init__(addr, RequestHandler)
        self.known_data = dataprovider.KnownStreamProvider(None, "../data", np.dtype(np.float16), True)
        self.win_data = dataprovider.DataProvider("../data", np.dtype(np.float16), dataprovider.PORTION_WIN, True)
        self.generator = Generator(self.known_data.fields, self.known_data.columns, config)
        self.generator.load_weights(config.g_weights)
        self.interesting_without_win_data = dataprovider.DataProvider("../data", np.dtype(np.float16), dataprovider.PORTION_INTERESTING - dataprovider.PORTION_WIN, True)
        self.interesting_data = dataprovider.DataProvider("../data", np.dtype(np.float16), dataprovider.PORTION_INTERESTING, True)
        self.win_estimator = WinEstimator(self.interesting_without_win_data.fields, self.interesting_without_win_data.columns, config)
        self.win_estimator.load_weights(config.w_weights)
        self.slice_win = self.interesting_data.columns.win_slice
        self.slice_known = self.interesting_data.columns.known_slice
        self.slice_interesting_without_win = self.interesting_data.columns.interesting_without_win_slice
        self.slice_unknown_without_win = self.interesting_data.columns.unknown_without_win_slice
        self.config = config
        self.logger = logging.getLogger("server")


if __name__ == '__main__':
    with open("logging.yaml", 'rt') as logging_yaml_file:
        logging.config.dictConfig(yaml.safe_load(logging_yaml_file.read()))
    server = Server(('', 33166), ApplicationConfiguration(sys.argv[1:]))
    logger = logging.getLogger("server")
    logger.info('start listening...')
    server.serve_forever()
