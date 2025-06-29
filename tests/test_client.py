import pytest
import pandas as pd
import os
from unittest import mock
import time
from datetime import datetime, timedelta

from tushare_db import TushareDBClient, ProBarAsset, ProBarAdj, ProBarFreq
from tushare_db.tushare_client import TushareClientError
from tushare_db.client import TushareDBClientError
from tushare_db.duckdb_manager import DuckDBManagerError

# Mock Tushare API responses
def mock_tushare_pro_bar(ts_code, start_date, end_date, **kwargs):
    """
    模拟 tushare.pro_bar 的响应，根据 start_date 和 end_date 生成日期。
    """
    # Determine date format based on input string
    date_format = '%Y%m%d'
    if ' ' in start_date:
        date_format = '%Y-%m-%d %H:%M:%S'

    start_dt = datetime.strptime(start_date, date_format)
    end_dt = datetime.strptime(end_date, date_format)

    trade_dates = []
    current_dt = start_dt
    while current_dt <= end_dt:
        if date_format == '%Y%m%d':
            trade_dates.append(current_dt.strftime('%Y%m%d'))
        else:
            trade_dates.append(current_dt.strftime('%Y-%m-%d %H:%M:%S'))
        current_dt += timedelta(days=1)
    trade_dates.sort(reverse=True) # Tushare usually returns latest date first

    num_rows = len(trade_dates)
    if num_rows == 0:
        return pd.DataFrame()

    data = {
        'ts_code': [ts_code] * num_rows,
        'trade_date': trade_dates,
        'open': [10.0] * num_rows,
        'high': [10.5] * num_rows,
        'low': [9.5] * num_rows,
        'close': [10.2] * num_rows,
        'pre_close': [10.1] * num_rows,
        'change': [0.1] * num_rows,
        'pct_chg': [1.0] * num_rows,
        'vol': [1000.0] * num_rows,
        'amount': [10000.0] * num_rows,
    }
    if 'adjfactor' in kwargs and kwargs['adjfactor']:
        data['adj_factor'] = [1.0] * num_rows
    if 'ma' in kwargs and kwargs['ma']:
        for m in kwargs['ma']:
            data[f'ma{m}'] = [f'ma_val_{m}'] * num_rows
    if 'factors' in kwargs and kwargs['factors']:
        if 'tor' in kwargs['factors']:
            data['turnover_rate'] = [0.1] * num_rows
        if 'vr' in kwargs['factors']:
            data['volume_ratio'] = [1.0] * num_rows
    
    df = pd.DataFrame(data)
    df['trade_date'] = df['trade_date'].astype(str)
    return df

def mock_tushare_fetch(api_name, **kwargs):
    """
    模拟 TushareClient.fetch 的响应。
    """
    if api_name == 'stock_basic':
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'name': ['平安银行', '万科A'],
            'list_status': ['L', 'L']
        })
    elif api_name == 'trade_cal':
        return pd.DataFrame({
            'exchange': ['SSE', 'SSE'],
            'cal_date': ['20230101', '20230102'],
            'is_open': [0, 1]
        })
    return pd.DataFrame() # 默认返回空DataFrame

@pytest.fixture
def tushare_token():
    """Fixture for a mock Tushare token."""
    return "mock_token_123"

@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture for a temporary DuckDB database path."""
    return str(tmp_path / "test_tushare.db")

@pytest.fixture
def client(tushare_token, temp_db_path):
    """Fixture for a TushareDBClient instance with a temporary database."""
    # Ensure the client is initialized with the mock token and temp db path
    with mock.patch('tushare_db.tushare_client.ts.pro_api'):
        with mock.patch('tushare_db.tushare_client.TushareClient.fetch', side_effect=mock_tushare_fetch):
            _client = TushareDBClient(tushare_token=tushare_token, db_path=temp_db_path)
            yield _client
            _client.close()
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

@pytest.fixture
def pro_bar_client(tushare_token, temp_db_path):
    """Fixture for a TushareDBClient instance specifically for pro_bar tests."""
    with mock.patch('tushare.pro_bar', side_effect=mock_tushare_pro_bar) as mock_pro_bar:
        _client = TushareDBClient(tushare_token=tushare_token, db_path=temp_db_path)
        yield _client, mock_pro_bar
        _client.close()
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

class TestTushareDBClient:
    def test_client_initialization(self, tushare_token, temp_db_path):
        """测试客户端初始化。"""
        with mock.patch('tushare_db.tushare_client.ts.pro_api'):
            client = TushareDBClient(tushare_token=tushare_token, db_path=temp_db_path)
            assert client.tushare_token == tushare_token
            assert client.duckdb_manager.db_path == temp_db_path
            client.close()

    def test_client_initialization_no_token(self, temp_db_path):
        """测试没有提供 Token 时客户端初始化失败。"""
        with mock.patch.dict(os.environ, {}, clear=True): # 清除环境变量
            with pytest.raises(TushareDBClientError, match="Tushare token not provided"):
                TushareDBClient(db_path=temp_db_path)

    def test_get_data_stock_basic_from_api_and_cache(self, client):
        """测试 stock_basic 数据从 API 获取并缓存。"""
        df = client.get_data('stock_basic', list_status='L')
        assert not df.empty
        assert 'ts_code' in df.columns
        assert len(df) == 2 # Based on mock data

        # Second call should hit cache
        df_cached = client.get_data('stock_basic', list_status='L')
        assert not df_cached.empty
        assert len(df_cached) == 2
        pd.testing.assert_frame_equal(df, df_cached)

    def test_get_data_trade_cal_incremental_update(self, client):
        """测试 trade_cal 数据的增量更新。"""
        # First fetch
        df1 = client.get_data('trade_cal', start_date='20230101', end_date='20230102')
        assert not df1.empty
        assert len(df1) == 2

        # Simulate new data coming in for incremental update
        with mock.patch('tushare_db.tushare_client.TushareClient.fetch') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'exchange': ['SSE'],
                'cal_date': ['20230103'],
                'is_open': [1]
            })
            # Fetching a range that includes new data
            df2 = client.get_data('trade_cal', start_date='20230101', end_date='20230103')
            assert not df2.empty
            assert len(df2) == 3 # Should have 2 old + 1 new

    def test_pro_bar_basic_stock_daily(self, pro_bar_client):
        """测试 pro_bar 接口获取股票日线数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105')
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SZ', start_date='20230101', end_date='20230105',
            asset='E', freq='D', adjfactor=False
        )
        assert not df.empty
        assert 'ts_code' in df.columns
        assert 'trade_date' in df.columns
        assert len(df) == 5

    def test_pro_bar_qfq_adj(self, pro_bar_client):
        """测试 pro_bar 接口获取前复权数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105', adj=ProBarAdj.QFQ)
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SZ', start_date='20230101', end_date='20230105',
            adj=ProBarAdj.QFQ, asset='E', freq='D', adjfactor=False
        )
        assert not df.empty

    def test_pro_bar_index_asset(self, pro_bar_client):
        """测试 pro_bar 接口获取指数数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SH', start_date='20230101', end_date='20230105', asset=ProBarAsset.INDEX)
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SH', start_date='20230101', end_date='20230105',
            asset=ProBarAsset.INDEX, freq='D', adjfactor=False
        )
        assert not df.empty

    def test_pro_bar_min_freq(self, pro_bar_client):
        """测试 pro_bar 接口获取分钟数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='2023-01-01 09:30:00', end_date='2023-01-01 10:00:00', freq=ProBarFreq.MIN5)
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SZ', start_date='2023-01-01 09:30:00', end_date='2023-01-01 10:00:00',
            asset='E', freq=ProBarFreq.MIN5, adjfactor=False
        )
        assert not df.empty

    def test_pro_bar_with_ma(self, pro_bar_client):
        """测试 pro_bar 接口获取带均线数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105', ma=[5, 10])
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SZ', start_date='20230101', end_date='20230105',
            asset='E', freq='D', ma=[5, 10], adjfactor=False
        )
        assert 'ma5' in df.columns
        assert 'ma10' in df.columns
        assert not df.empty

    def test_pro_bar_with_factors(self, pro_bar_client):
        """测试 pro_bar 接口获取带因子数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105', factors=['tor', 'vr'])
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SZ', start_date='20230101', end_date='20230105',
            asset='E', freq='D', factors=['tor', 'vr'], adjfactor=False
        )
        assert 'turnover_rate' in df.columns
        assert 'volume_ratio' in df.columns
        assert not df.empty

    def test_pro_bar_with_adjfactor(self, pro_bar_client):
        """测试 pro_bar 接口获取带复权因子数据。"""
        client, mock_pro_bar = pro_bar_client
        df = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105', adjfactor=True)
        
        mock_pro_bar.assert_called_once_with(
            ts_code='000001.SZ', start_date='20230101', end_date='20230105',
            asset='E', freq='D', adjfactor=True
        )
        assert 'adj_factor' in df.columns
        assert not df.empty

    def test_pro_bar_caching(self, pro_bar_client):
        """测试 pro_bar 接口的缓存机制。"""
        client, mock_pro_bar = pro_bar_client
        
        # First call - should fetch from API
        df1 = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105')
        assert mock_pro_bar.call_count == 1
        assert not df1.empty

        # Second call with same parameters - should hit cache
        df2 = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230105')
        assert mock_pro_bar.call_count == 1 # Still 1, indicating cache hit
        pd.testing.assert_frame_equal(df1, df2)

        # Third call with different date range - should fetch new data incrementally
        # Mock new data for incremental update
        mock_pro_bar.side_effect = [
            mock_tushare_pro_bar('000001.SZ', '20230106', '20230107'), # New data
            mock_tushare_pro_bar('000001.SZ', '20230101', '20230107') # Full range for re-query
        ]
        df3 = client.get_data('pro_bar', ts_code='000001.SZ', start_date='20230101', end_date='20230107')
        assert mock_pro_bar.call_count == 2 # Original call + 1 for incremental
        assert len(df3) == 7 # 5 old + 2 new
        assert '20230106' in df3['trade_date'].values
        assert '20230107' in df3['trade_date'].values

    @mock.patch('tushare_db.client.TushareDBClient.get_data')
    def test_initialize_basic_data(self, mock_get_data, client):
        """测试基础数据初始化方法。"""
        client.initialize_basic_data()

        # Verify that get_data was called for each basic API
        mock_get_data.assert_any_call('stock_basic', list_status='L')
        mock_get_data.assert_any_call('trade_cal', start_date='19900101', end_date=mock.ANY) # end_date is dynamic
        mock_get_data.assert_any_call('hs_const', is_new='1')
        mock_get_data.assert_any_call('stock_company')
        assert mock_get_data.call_count == 4

    @mock.patch('tushare_db.client.TushareDBClient.get_data')
    def test_get_all_stock_qfq_daily_bar(self, mock_get_data, client):
        """测试获取所有股票前复权日线数据。"""
        # Mock stock_basic to return some stock codes
        mock_get_data.side_effect = [
            pd.DataFrame({'ts_code': ['000001.SZ', '000002.SZ']}), # For stock_basic call
            mock_tushare_pro_bar('000001.SZ', '20230101', '20230105'), # For 000001.SZ pro_bar call
            mock_tushare_pro_bar('000002.SZ', '20230101', '20230105')  # For 000002.SZ pro_bar call
        ]

        start_date = '20230101'
        end_date = '20230105'
        all_qfq_df = client.get_all_stock_qfq_daily_bar(start_date, end_date)

        assert not all_qfq_df.empty
        assert len(all_qfq_df) == 10 # 2 stocks * 5 days each
        assert '000001.SZ' in all_qfq_df['ts_code'].values
        assert '000002.SZ' in all_qfq_df['ts_code'].values
        # Verify that get_data was called for stock_basic and then for each pro_bar
        mock_get_data.assert_any_call('stock_basic', list_status='L')
        mock_get_data.assert_any_call('pro_bar', ts_code='000001.SZ', start_date=start_date, end_date=end_date, adj='qfq', freq='D', asset='E')
        mock_get_data.assert_any_call('pro_bar', ts_code='000002.SZ', start_date=start_date, end_date=end_date, adj='qfq', freq='D', asset='E')
        assert mock_get_data.call_count == 3 # 1 for stock_basic, 2 for pro_bar

    @mock.patch('tushare_db.client.TushareDBClient.get_data')
    @mock.patch('tushare_db.client.TushareDBClient.get_all_stock_qfq_daily_bar')
    @mock.patch('tushare_db.client.datetime')
    def test_daily_update(self, mock_datetime, mock_get_all_stock_qfq_daily_bar, mock_get_data, client):
        """测试每日更新方法。"""
        # Mock datetime.now() to control dates
        mock_datetime.now.return_value = datetime(2023, 1, 6)
        mock_datetime.strptime = datetime.strptime # Ensure original strptime is used
        mock_datetime.timedelta = timedelta # Ensure original timedelta is used

        client.daily_update()

        # Verify calls
        mock_get_data.assert_any_call('trade_cal', start_date='20230105', end_date='20230106')
        mock_get_all_stock_qfq_daily_bar.assert_called_once_with(start_date='20230105', end_date='20230106')
        assert mock_get_data.call_count == 1 # Only trade_cal is called via get_data directly
        assert mock_get_all_stock_qfq_daily_bar.call_count == 1
