import argparse
from datetime import datetime
import numpy as np
import os
from tqdm import tqdm
import yaml


class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def _read_EOD_data(self):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            # read data and transform to numpy array, skip the first row
            single_EOD = np.genfromtxt(
                os.path.join(self.data_path, 'google_finance', self.market_name.upper() + '_' + ticker +
                             '_30Y.csv'), dtype=str, delimiter=',',
                skip_header=True
            )
            self.data_EOD.append(single_EOD)
            # if index > 99:
            #     break
        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def add_return(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        for row, daily_EOD in enumerate(selected_EOD_str):
            # remove timezone
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            selected_EOD[row][0] = tra_date_index[date_str]
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row][col] = float(daily_EOD[col])
        # add feature,(volume(today/yesterday))-1
        add_return = np.divide(
            (selected_EOD[1:])[:, -1], (selected_EOD[:-1])[:, -1])-1
        selected_EOD[:, -1] = np.insert(add_return, 0, add_return.mean())
        return selected_EOD

    '''
        Transform the original EOD data collected from Google Finance to a
        friendly format to fit machine learning model via the following steps:
            Calculate moving average (5-days, 10-days, 20-days, 30-days),
            ignoring suspension days (market open, only suspend this stock)
            Normalize features by (feature - min) / (max - min)
    '''

    def generate_feature(self, selected_tickers_fname, begin_date, end_date, opath):

        # read trading dates, a file with all trading dates
        trading_dates = np.genfromtxt(
            os.path.join(self.data_path, self.market_name.upper() +
                         '_aver_line_dates.csv'),
            dtype=str, delimiter=',', skip_header=False
        )

        print('#trading dates:', len(trading_dates))
        # begin_date = datetime.strptime(trading_dates[29], self.date_format)
        print('begin date:', begin_date)
        print('end date:', end_date)
        # transform the trading dates into a dictionary with index
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            tra_dates_index[date] = index
            index_tra_dates[index] = date
        self.tickers = np.genfromtxt(
            os.path.join(self.data_path, selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )

        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data()
        # generate features for each stock
        for stock_index, single_EOD in enumerate(tqdm(self.data_EOD)):
            # select data within the begin_date
            cur_list = []
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date:
                    cur_list.append(date_index)
                    if end_date <= cur_date:
                        break
            selected_EOD_str = single_EOD[cur_list[0]: cur_list[-1]]
            # add return feature
            selected_EOD = self.add_return(selected_EOD_str,
                                           tra_dates_index)

            # fix missing
            if len(selected_EOD) != len(tra_dates_index):
                miss_index = np.setdiff1d(
                    np.arange(0, len(tra_dates_index), 1, float), selected_EOD[:, 0])
                for each_miss in range(len(miss_index)):
                    miss_value = np.array(
                        [miss_index[each_miss], -1234, -1234, -1234, -1234, -1234])
                    selected_EOD = np.insert(
                        selected_EOD, each_miss, miss_value, axis=0)

            np.savetxt(os.path.join(opath, self.market_name.upper() + '_' +
                                    self.tickers[stock_index] + '_' +
                                    '1' + '.csv'), selected_EOD,
                       fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    # desc = "pre-process EOD data market by market, including listing all " \
    #        "trading days, all satisfied stocks (5 years & high price), " \
    #        "normalizing and compansating data"
    # parser = argparse.ArgumentParser(description=desc)
    current_path = os.path.dirname(os.path.abspath(__file__))
    path_default = os.path.join(current_path, '..', 'data')
    # # parser.add_argument('-market', help='market name', default='NYSE')
    # parser.add_argument('-market', help='market name', default='NASDAQ')

    # get config by ../experiments/parameters_dgrcl.yaml
    path_parameters = os.path.join(current_path, '..', 'experiments')
    with open(f'{path_parameters}/parameters_dgrcl.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # # NASDAQ NYSE
    # args = parser.parse_args()
    output_path = os.path.join(path_default, 'generated_data')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(config['stock_name'])
    market = config['stock_name']

    processor = EOD_Preprocessor(path_default, market)
    processor.generate_feature(
        processor.market_name.upper() + '_tickers_qualify_dr-0.98_min-5_smooth.csv',
        datetime.strptime('2013-01-01 00:00:00', processor.date_format),
        datetime.strptime('2017-01-01 00:00:00', processor.date_format),
        # os.path.join(processor.data_path, '..', '2013-01-01-2016-12-30')
        output_path
    )

