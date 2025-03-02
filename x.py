import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def export_tables_to_csv(output_dir='database_exports'):
    # Load environment variables
    load_dotenv()

    # Explicitly define connection parameters
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('POSTGRES_DB', 'appdb'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'port': os.getenv('DB_PORT', '5432')
    }

    # Construct connection URL
    DATABASE_URL = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL)

        # List of tables to export
        tables = ['content_data', 'benefits', 'content_benefits']

        # Export each table to a CSV
        for table in tables:
            # Read table into DataFrame
            df = pd.read_sql_table(table, engine)
            
            # Generate CSV filename
            csv_filename = os.path.join(output_dir, f'{table}_export.csv')
            
            # Export to CSV
            df.to_csv(csv_filename, index=False)
            print(f"Exported {table} to {csv_filename}")

        print("Database export completed successfully!")

    except Exception as e:
        print(f"Error exporting database: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()

# Run the export function
if __name__ == '__main__':
    export_tables_to_csv()