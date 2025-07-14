
import os
import pytest
from src.tushare_db.client import TushareDBClient
from src.tushare_db.api import cyq_chips

# Mock Tushare Pro API token
MOCK_TUSHARE_TOKEN = "your_tushare_token"

@pytest.fixture(scope="module")
def client():
    """
    Pytest fixture to initialize and close the TushareDBClient.
    """
    # Set the Tushare token as an environment variable
    os.environ["TUSHARE_TOKEN"] = MOCK_TUSHARE_TOKEN
    
    # Initialize the client
    db_path = "test_cyq_chips.db"
    client = TushareDBClient(db_path=db_path)
    yield client
    
    # Teardown: close the client and remove the test database
    client.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(f"{db_path}.wal"):
        os.remove(f"{db_path}.wal")

def test_cyq_chips_data_fetching(client):
    """
    Test fetching data using the cyq_chips API.
    
    Note: This test requires a valid Tushare token with access to the cyq_chips API.
    """
    try:
        # Fetch data for a specific stock and date range
        df = cyq_chips(client, ts_code='600000.SH', start_date='20220101', end_date='20220105')
        
        # Perform assertions on the returned DataFrame
        assert not df.empty, "The DataFrame should not be empty."
        assert 'ts_code' in df.columns, "The 'ts_code' column is missing."
        assert 'trade_date' in df.columns, "The 'trade_date' column is missing."
        assert 'price' in df.columns, "The 'price' column is missing."
        assert 'percent' in df.columns, "The 'percent' column is missing."
        
        # Check if the data for the correct stock is returned
        assert df['ts_code'].unique() == ['600000.SH'], "The ts_code should be '600000.SH'."
        
        print(f"Successfully fetched {len(df)} rows of data for 600000.SH.")
        print(df.head())

    except Exception as e:
        pytest.fail(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
