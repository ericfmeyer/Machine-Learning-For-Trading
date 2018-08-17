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
DEFAULT_COLUMN = 'Adj Close'


def symbol_to_path(symbol, base_dir='data'):
    """
    Return CSV file path given the ticker symbol.
    :param symbol: symbol of the stock
    :param base_dir: base directory (default 'data')
    :return: file path
    """
    return os.path.join(base_dir, '{}.csv'.format(symbol))


def get_data(symbols, dates, columns=DEFAULT_COLUMN):
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
        df_tmp = get_data_from_csv(symbol, columns)\
            .rename(columns={'Adj Close': symbol})
        df = df.join(df_tmp, how='inner')
        if symbol == SP500:
            df = df.dropna(subset=[SP500])

    return df


def get_data_from_csv(symbol, columns=DEFAULT_COLUMN):
    """
    Returns the content of the CSV file from the stock indicated by symbol.

    Note: Data for a stock is stored in the file: data/<symbol>.csv
    :param columns: columns to parse
    :param symbol: symbol of the stock to read data from
    :return:    a Pandas DataFrame
    """
    if not isinstance(columns, list):
        columns = [columns]

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


def plot_data(list_df, labels, title='Stock prices'):
    """
    Plots the data contained in the dataframe.
    :param list_df: list of dataframes to plot
    :param labels: labels to use for each values
    :param title: title of the plot chart
    :return: nothing
    """
    ax = list_df[0].plot(title=title, fontsize=10, label=labels[0])
    for i, data in enumerate(list_df[1:]):
        data.plot(label=labels[i + 1], ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    plt.show()


def plot_mean(df, window=20):
    """
    Plots the values and the mean over the specified window.
    :param df: dataframe containing all the values
    :param window: the time window
    :return: nothing
    """
    rm = get_rolling_mean(df, window)
    plot_data([df, rm], labels=[df.name, 'Rolling mean ({})'.format(window)], title='Rolling mean')


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


def get_rolling_mean(values, window=20):
    """
    Returns the rolling mean of given values, using given window size.
    :param values: Pandas dataframe with values
    :param window: size of the window (int)
    :return: rolling mean
    """
    return values.rolling(window).mean()


def get_rolling_std(values, window=20):
    """
    Returns the rolling standard deviation of given values, using given window size.
    :param values: Pandas dataframe with values
    :param window: size of the window (int)
    :return: rolling standard deviation
    """
    return values.rolling(window).std()


def get_bollinger_bands(rm, rstd):
    """
    Return upper and lower Bollinger Bands.
    :param rm: rolling mean values
    :param rstd: rolling standard deviation values
    :return: upper and lower bands
    """
    upper = rm + 2 * rstd
    lower = rm - 2 * rstd
    return upper, lower


def get_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    print(daily_returns.head())
    # daily_returns.ix[0, :] = 0
    print(daily_returns.head())
    return daily_returns


def test_run():
    # Define the date range
    start_date = '2012-01-01'
    end_date = '2012-12-31'

    dates = pd.date_range(start_date, end_date)
    df = get_data(SYMBOLS, dates)

    rolling_mean = get_rolling_mean(df[SP500])
    rolling_std = get_rolling_std(df[SP500])

    upper, lower = get_bollinger_bands(rolling_mean, rolling_std)

    plot_data([df[SP500], rolling_mean, upper, lower], labels=[SP500, 'Rolling mean', 'Upper band', 'Lower band'])
    # plot_mean(df[SP500], window=26)

    daily_returns = get_daily_returns(df[SP500])
    plot_data([daily_returns], 'Daily Returns')

    # print("Mean: ", df.mean())
    # print("Median: ", df.median())
    # print("Std: ", df.std())

    # plot_selected(df, [SP500, GOOGLE, IBM, GOLD], start_date, end_date)
    # plot_data(normalize_data(df))


if __name__ == '__main__':
    test_run()
