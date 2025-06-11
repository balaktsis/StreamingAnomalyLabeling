def connect_to_db() -> tuple:
    """
    Connects to the PostgreSQL database and returns a tuple
    (connection, connection_cursor).

    example: conn, cursor = connect_to_db()
    """
    import psycopg2
    from dotenv import load_dotenv
    import os

    load_dotenv()
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            port=os.getenv("POSTGRES_MAPPED_PORT", "5432"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "postgres"),
        )
        return conn, conn.cursor()
    except psycopg2.OperationalError:
        raise "Error connecting to the database. Please check your connection settings."


def insert_experiment_result(result: dict):
    import uuid
    from psycopg2.extensions import AsIs
    import psycopg2
    import numpy as np

    conn, cursor = connect_to_db()

    result['experiment_id'] = result.get('experiment_id', str(uuid.uuid4()))
    columns = result.keys()
    values = result.values()
    values = [value if not type(value) == np.int64 else int(value) for value in values]

    try:
        query = """
            INSERT INTO experiments (%s) VALUES %s
        """
        cursor.execute(
            query,
            (AsIs(','.join(columns)), tuple(values))
        )
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def experiment_exists(files: list[str], method_name: str, tag: str, extra_info: dict = None) -> bool:
    """
    Checks if an experiment with the given files, method name, and tag already exists in the database.

    :param files: List of file paths associated with the experiment.
    :param method_name: Name of the method used in the experiment.
    :param tag: Tag associated with the experiment.
    :return: True if the experiment exists, False otherwise.
    """
    import json
    conn, cursor = connect_to_db()

    query = """
        SELECT EXISTS (
            SELECT 1 FROM experiments
            WHERE files = %s AND method_name = %s AND tag = %s
    """
    values = (json.dumps(files), method_name, tag)

    if extra_info:
        query += " AND extra_info = %s"
        values = (*values, json.dumps(extra_info))

    query += ")" # closing the EXISTS clause

    cursor.execute(query, values)
    exists = cursor.fetchone()[0]

    if exists:
        print("Experiment already exists in the database.")

    cursor.close()
    return exists
