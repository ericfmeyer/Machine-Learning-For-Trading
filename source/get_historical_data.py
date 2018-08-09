import os
import pandas as pd
import matplotlib.pyplot as plt


AAPLE = 'AAPL'
AMD = 'AMD'
GOLD = 'GLD'
GOOGLE = 'GOOGL'
IBM = 'IBM'
SP500 = 'SPY'

SYMBOLS = [AAPLE, AMD, GOLD, GOOGLE, IBM]


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
    Read stock data for given symbols, given columns and given dates
    from CSV files.
    :param symbols: symbols to get data for
    :param columns: columns to get data for
    :param dates: date range
    :return:    returns a DataFrame object
    """
    df = pd.DataFrame(index=dates)
    if SP500 not in symbols:
        symbols.insert(0, SP500)

    for symbol in symbols:
        # Join the 2 dataframes
        df_tmp = get_data_from_csv(symbol, columns)\
            .rename(columns={'Adj Close': symbol})
        df = df.join(df_tmp, how='inner')
        if symbol == SP500:
            df = df.dropna(subset=[SP500])

    return df


def get_data_from_csv(symbol, columns=['Adj Close']):
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


def plot_data(df, title='Stock prices'):
    """
    Plots the data contained in the dataframe.
    :param df: dataframe to plot
    :param title: title of the plot chart
    :return: nothing
    """
    ax = df.plot(title=title, fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def plot_selected(df, columns, start_index, end_index):
    """
    Plots the selected columns over the selected date range.
    :param df: dataframe containing stock prices
    :param columns: columns to plot from the dataframe
    :param start_index: start index
    :param end_index: end index
    :return: nothing
    """
    plot_data(df.ix[start_index:end_index, columns])


def normalize_data(df):
    """
    Returns the normalized dataframe, by using the first row of the dataframe.
    :param df:
    :return:
    """
    return df/df.ix[0]


def test_run():
    # Define the date range
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    dates = pd.date_range(start_date, end_date)

    df = get_data(SYMBOLS, dates)
    print(df)

    # row slicing using DataFrame.ix[] selector
    # print(df.ix[start_date:'2010-01-31'])

    # slice by column (symbol)
    # print(df['SPY'])
    # print(df[['SPY', 'AAPL']])

    # slice by row and column
    # print(df.ix['2010-03-10':'2010-03-15', ['SPY', 'AMD']])

    plot_data(df)
    plot_selected(df, [SP500, GOOGLE, IBM, GOLD], start_date, end_date)
    plot_data(normalize_data(df))


if __name__ == '__main__':
    test_run()
