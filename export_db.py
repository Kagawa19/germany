import os
import shutil
import pandas as pd
from sqlalchemy import create_engine, inspect, text
import logging

def export_tables_to_csv(db_url=None, output_dir='./data'):
    """
    Export all tables from PostgreSQL database to CSV files.
    Removes any existing CSV files in the output directory before creating new ones.
    
    Parameters:
    -----------
    db_url : str, optional
        Database connection string. If None, will use environment variables.
        Example: 'postgresql://postgres:postgres@postgres:5432/appdb'
    output_dir : str, optional
        Directory to save CSV files, defaults to './data'
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Get database URL from environment variables if not provided
        if db_url is None:
            db_user = os.environ.get('POSTGRES_USER', 'postgres')
            db_password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
            db_host = os.environ.get('POSTGRES_HOST', 'postgres')
            db_port = os.environ.get('POSTGRES_PORT', '5432')
            db_name = os.environ.get('POSTGRES_DB', 'appdb')
            db_url = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        
        logger.info(f"Connecting to database: {db_url.replace('://'+db_url.split('://')[1].split(':')[0]+':', '://USER:')}")
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        else:
            # Remove existing CSV files in the output directory
            logger.info(f"Cleaning output directory: {output_dir}")
            for file in os.listdir(output_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(output_dir, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed existing file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
        
        # Get list of tables in the database
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            logger.warning("No tables found in the database.")
            return True
        
        logger.info(f"Found {len(tables)} tables in database: {', '.join(tables)}")
        
        # Export each table to CSV
        for table in tables:
            output_file = os.path.join(output_dir, f"{table}.csv")
            try:
                # Read table data
                logger.info(f"Exporting table '{table}' to {output_file}")
                query = text(f"SELECT * FROM {table}")
                df = pd.read_sql(query, engine)
                
                # Save to CSV
                df.to_csv(output_file, index=False)
                logger.info(f"Successfully exported {len(df)} rows from '{table}' table")
                
            except Exception as e:
                logger.error(f"Error exporting table '{table}': {str(e)}")
                continue
        
        logger.info(f"Export complete! CSV files saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error in export_tables_to_csv: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    export_tables_to_csv()