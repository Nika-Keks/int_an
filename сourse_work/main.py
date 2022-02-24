from generator import Generator, QuadraticGenerator
from interval_data import IntervalData
from interval_regression import QuadraticIntervalRegression


import numpy as np
import numpy.typing as npt


def work(regression_type: QuadraticIntervalRegression.RegressionType, datas: npt.ArrayLike, additional_information: npt.ArrayLike) -> None:
    for data, info in zip(datas, additional_information):
        regression = QuadraticIntervalRegression.create(regression_type, data)

        data.plot(True, info)
        data.save_as_csv(f'{info}{data.size()}')

        params = regression.build_model()
        print(f'{info}: params = {params}')

        regression.plot(info, True)
        print('###############')

    print('\n')

def data(case: int) -> tuple:
    if case == 0:
        data = QuadraticGenerator().generate(25, 5, 25, [10, 5, -1])
        em_data = data.add_emissions(5, 5, 1)

    elif case == 1:
        data = QuadraticGenerator().generate(50, 0, 10, [-10, -20, 3])
        em_data = data.add_emissions(5, 5, 1)

    else:
        return None

    idata = IntervalData(data) 

    return idata, idata.add_emissions(5, 5, 1)

def work_extended_gap(*args):
    extended_data = QuadraticGenerator().generate(*args)
    data = IntervalData(Generator.DataInfo(extended_data.factors()[5:25], extended_data.responses()[5:25]))
    extended_data = IntervalData(extended_data)

    extended_data.plot()

    for regression_pyte in [QuadraticIntervalRegression.RegressionType.UndifinedCenter, QuadraticIntervalRegression.RegressionType.Tol]:
        regression = QuadraticIntervalRegression.create(regression_pyte, data)

        regression.plot("Gap")
        params = regression.build_model()
        print(f'params = {params}')

        regression.data = extended_data

        name = "ExtendedGap"
        regression.plot(name)
        regression.additional_plot(name)


def main():
    cases = []#[0, 1]

    work_extended_gap(35, 0, 30, [10, 5, -1])
    work_extended_gap(60, 0, 20, [-10, -20, 3])

    for case in cases:
        print(f'Case: {case}')
        idata, em_idata = data(case)

        work(QuadraticIntervalRegression.RegressionType.UndifinedCenter, np.array([idata, em_idata]), np.array(['ValidDataMS', 'DataWithEstimsMS']))
        work(QuadraticIntervalRegression.RegressionType.Tol, np.array([idata, em_idata]), np.array(['ValidDataMS', 'DataWithEstimsMS']))


    return


if __name__ == '__main__':
    main()
