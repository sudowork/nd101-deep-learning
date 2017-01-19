import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression


def parse_args():
    parser = argparse.ArgumentParser(
        description='Perform linear regression on dataset'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='file to perform regression on',
    )
    parser.add_argument(
        '--format',
        type=str,
        default='csv',
        help='file format (csv or fwf). default: csv',
    )
    parse_has_header_arg(parser)
    return parser.parse_args()


def parse_has_header_arg(parser):
    has_header_parser = parser.add_mutually_exclusive_group(required=False)
    has_header_parser.add_argument(
        '--header',
        dest='has_header',
        action='store_true',
    )
    has_header_parser.add_argument(
        '--no-header',
        dest='has_header',
        action='store_false',
        help='default',
    )
    parser.set_defaults(has_header=False)
    return parser


def read_xy_from_file(filepath, format, has_header):
    assert format == 'csv' or format == 'fwf'
    read_file = pd.read_csv if format == 'csv' else pd.read_fwf
    header = 'infer' if has_header else None
    dataframe = read_file(filepath, header=header)
    return dataframe.ix[:, 0], dataframe.ix[:, 1]  # Pair of pd.Series


def fit_linear_regression(x, y):
    regression = LinearRegression()
    regression.fit(x.values.reshape([-1, 1]), y)
    return regression


def plot_with_line_of_best_fit(x, y, regression, has_header):
    _, ax = plt.subplots()
    xlabel, ylabel = get_axes_labels(x, y, has_header)
    plot_xy_with_regression_line(ax, x, y, regression)
    label_plot(ax, xlabel, ylabel)
    annotate_regression_formula(ax, xlabel, ylabel, regression)
    plt.show()


def plot_xy_with_regression_line(ax, x, y, regression):
    y_predicted = regression.predict(x.values.reshape([-1, 1]))
    ax.scatter(x, y)
    ax.plot(x, y_predicted)


def get_axes_labels(x, y, has_header):
    if has_header:
        xlabel = x.name
        ylabel = y.name
    else:
        xlabel = 'x'
        ylabel = 'y'
    return xlabel, ylabel


def annotate_regression_formula(ax, xlabel, ylabel, regression):
    m = regression.coef_[0]
    b = regression.intercept_
    b_abs = abs(b)
    sign = '+' if b >= 0 else '-'
    formula = '$\\mathrm{{{ylabel}}} = {m:.3f} \\cdot \\mathrm{{{xlabel}}} {sign} {b:.3f}$'.format(  # nopep8
        ylabel=ylabel,
        xlabel=xlabel,
        m=m,
        sign=sign,
        b=b_abs,
    )
    annotation = AnchoredText(formula, loc=2)
    ax.add_artist(annotation)


def label_plot(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Plot of {ylabel} vs {xlabel}'.format(
        ylabel=ylabel,
        xlabel=xlabel,
    ))


if __name__ == '__main__':
    args = parse_args()
    filepath = args.input_file
    format = args.format
    has_header = args.has_header
    assert os.path.isfile(filepath)

    x, y = read_xy_from_file(filepath, format, has_header)
    regression = fit_linear_regression(x, y)
    plot_with_line_of_best_fit(x, y, regression, has_header)
