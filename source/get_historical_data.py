import os
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = ['AAPL', 'AMD']
SP500 = 'SPY'


def symbol_to_path(symbol, base_dir='data'):
    """
    Return CSV file path given the ticker symbol.
    :param symbol: symbol of the stock
    :param base_dir: base directory (default 'data')
    :return: file path
    """
    return os.path.join(base_dir, '{}.csv'.format(symbol))


def get_data(symbols, dates, columns=['Adj Close']):
    """
    Read stock data for given symbols, given columns and given dates from CSV files.
    :param symbols: symbols to get data for
    :param columns: columns to get data for
    :param dates: date range
    :return:    returns a DataFrame object
    """
    df = pd.DataFrame(index=dates)
    if SP500 not in symbols:
        symbols.insert(0, SP500)

    for symbol in TICKERS:
        # Join the 2 dataframes
        df_tmp = get_data_from_csv(symbol, columns).rename(columns={'Adj Close': symbol})
        df = df.join(df_tmp, how='inner')
        if symbol == SP500:
            df = df.dropna(subset=[SP500])

    return df


def get_data_from_csv(symbol, columns):
    """
    Returns the content of the CSV file from the stock indicated by symbol.

    Note: Data for a stock is stored in the file: data/<symbol>.csv
    :param columns: columns to parse
    :param symbol: symbol of the stock to read data from
    :return:    a Pandas DataFrame
    """
    if 'Date' not in columns:
        columns.insert(0, 'Date')

    return pd.read_csv(symbol_to_path(symbol), index_col="Date",
                       parse_dates=True, usecols=columns, na_values=['nan'])


def get_mean_volume(data):
    """
    Return the mean volume for a stock indicated by symbol.
    :param data: symbol of the stock to analyze
    :return: the mean volume of the stock
    """
    return data['Volume'].mean()


def get_max_close(data):
    """
    Returns the maximum closing value for a stock indicated by symbol.
    :param data: symbol of the stock to analyze
    :return: the maximum closing value of the stock
    """
    return data['Close'].max()


def plot_data(symbol, data, columns):
    """
    Plots symbols.
    :param symbol:
    :param data:
    :param columns:
    :return:
    """
    data[columns].plot()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('{}'.format(symbol))
    plt.show()


def test_run():
    # Define the date range
    start_date = '2010-01-22'
    end_date = '2010-01-26'
    dates = pd.date_range(start_date, end_date)

    df = get_data(TICKERS, dates)
    print(df)

    # for symbol in TICKERS:
    #     df = get_data_from_csv(symbol)
    #     print(symbol)
    #     print('Max close = {}'.format(get_max_close(df)))
    #     print('Mean volume = {}'.format(get_mean_volume(df)))
    #     plot_data(symbol, df, ['Close', 'High', 'Low'])


if __name__ == '__main__':
    test_run()