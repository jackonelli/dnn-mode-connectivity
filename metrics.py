"""Metrics"""
import numpy as np
from tabulate import tabulate


class TestCurve:
    """Data container for test_curve"""
    def __init__(self, num_samples, columns):
        self.ts = np.linspace(0.0, 1.0, num_samples)
        self.tr_loss = np.zeros(num_samples)
        self.tr_nll = np.zeros(num_samples)
        self.tr_acc = np.zeros(num_samples)
        self.te_loss = np.zeros(num_samples)
        self.te_nll = np.zeros(num_samples)
        self.te_acc = np.zeros(num_samples)
        self.dl = np.zeros(num_samples)
        self.columns = columns

    def add_meas(self, ind, tr_res, te_res):
        """Add measurement
        From train and test results
        """
        self.tr_loss[ind] = tr_res["loss"]
        self.tr_nll[ind] = tr_res["nll"]
        self.tr_acc[ind] = tr_res["accuracy"]
        self.te_loss[ind] = te_res["loss"]
        self.te_nll[ind] = te_res["nll"]
        self.te_acc[ind] = te_res["accuracy"]

    def values(self, ind):
        t = self.ts[ind]
        tr_err = 100 - self.tr_acc[ind]
        te_err = 100 - self.te_acc[ind]
        values = [
            t, self.tr_loss[ind], self.tr_nll[ind], tr_err, self.te_nll[ind],
            te_err
        ]
        return values

    def table(self, ind, with_header):
        table = tabulate([self.values(ind)],
                         self.columns,
                         tablefmt="simple",
                         floatfmt="10.4f")
        if with_header:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]

        return table
