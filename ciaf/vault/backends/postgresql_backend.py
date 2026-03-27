"""
PostgreSQL Backend for CIAF Vault

Enterprise-grade storage backend using PostgreSQL for:
- Metadata storage
- Receipt storage
- Audit trails
- Compliance events

Features:
- Connection pooling
- Transaction management
- Prepared statements
- Full ACID compliance
- Concurrent access support

Requirements:
    pip install psycopg2-binary

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2 import pool, extras
    from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    pool = None


class PostgreSQLBackend:
    """
    PostgreSQL storage backend for CIAF vault.

    Provides enterprise-grade storage with:
    - Connection pooling for performance
    - Transaction management
    - Full ACID compliance
    - Concurrent access support
    - Advanced indexing and querying
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "ciaf_vault",
        user: str = "ciaf_user",
        password: str = None,
        min_connections: int = 2,
        max_connections: int = 20,
        schema: str = "public",
    ):
        """
        Initialize PostgreSQL backend.

        Args:
            host: PostgreSQL host address
            port: PostgreSQL port (default: 5432)
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
            schema: Database schema name
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for PostgreSQL backend. "
                "Install it with: pip install psycopg2-binary"
            )

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema

        # Create connection pool
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            options=f"-c search_path={schema}",
        )

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema with all required tables."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Create metadata table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.ciaf_metadata (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    metadata_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    stage TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata_hash TEXT,
                    details TEXT,
                    metadata_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # Create indexes on metadata table
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_metadata_model_name
                ON {self.schema}.ciaf_metadata(model_name);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_metadata_stage
                ON {self.schema}.ciaf_metadata(stage);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_metadata_timestamp
                ON {self.schema}.ciaf_metadata(timestamp);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_metadata_json
                ON {self.schema}.ciaf_metadata USING GIN(metadata_json);
            """)

            # Create audit trail table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.ciaf_audit_trail (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    audit_id TEXT UNIQUE NOT NULL,
                    parent_id TEXT,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_id TEXT,
                    details TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_parent_id
                ON {self.schema}.ciaf_audit_trail(parent_id);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON {self.schema}.ciaf_audit_trail(timestamp);
            """)

            # Create compliance events table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.ciaf_compliance_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_id TEXT UNIQUE NOT NULL,
                    framework TEXT NOT NULL,
                    requirement TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    details TEXT,
                    evidence JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_compliance_framework
                ON {self.schema}.ciaf_compliance_events(framework);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_compliance_timestamp
                ON {self.schema}.ciaf_compliance_events(timestamp);
            """)

            # Create inference receipts table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.ciaf_inference_receipts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    receipt_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    query_hash TEXT,
                    output_hash TEXT,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    receipt_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_receipt_model_name
                ON {self.schema}.ciaf_inference_receipts(model_name);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_receipt_timestamp
                ON {self.schema}.ciaf_inference_receipts(timestamp);
            """)

            # Create training snapshots table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.ciaf_training_snapshots (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    snapshot_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    epoch INTEGER,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metrics JSONB,
                    snapshot_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_snapshot_model_name
                ON {self.schema}.ciaf_training_snapshots(model_name);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp
                ON {self.schema}.ciaf_training_snapshots(timestamp);
            """)

            # Create provenance capsules table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.ciaf_provenance_capsules (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    capsule_id TEXT UNIQUE NOT NULL,
                    dataset_id TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    capsule_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_capsule_dataset_id
                ON {self.schema}.ciaf_provenance_capsules(dataset_id);
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_capsule_timestamp
                ON {self.schema}.ciaf_provenance_capsules(timestamp);
            """)

            conn.commit()
            cursor.close()
        finally:
            self.return_connection(conn)

    def get_connection(self):
        """Get a connection from the pool."""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool."""
        self.connection_pool.putconn(conn)

    def store_metadata(
        self,
        metadata_id: str,
        model_name: str,
        stage: str,
        event_type: str,
        metadata_json: Dict[str, Any],
        model_version: Optional[str] = None,
        metadata_hash: Optional[str] = None,
        details: Optional[str] = None,
    ) -> bool:
        """
        Store metadata in PostgreSQL.

        Args:
            metadata_id: Unique metadata identifier
            model_name: Model name
            stage: Training/inference stage
            event_type: Type of event
            metadata_json: Full metadata as JSON
            model_version: Model version (optional)
            metadata_hash: Hash of metadata (optional)
            details: Additional details (optional)

        Returns:
            True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.ciaf_metadata
                (metadata_id, model_name, model_version, stage, event_type, metadata_hash, details, metadata_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (metadata_id)
                DO UPDATE SET
                    metadata_json = EXCLUDED.metadata_json,
                    updated_at = NOW()
            """,
                (
                    metadata_id,
                    model_name,
                    model_version,
                    stage,
                    event_type,
                    metadata_hash,
                    details,
                    json.dumps(metadata_json),
                ),
            )
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error storing metadata: {e}")
            return False
        finally:
            self.return_connection(conn)

    def retrieve_metadata(
        self,
        metadata_id: Optional[str] = None,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metadata from PostgreSQL.

        Args:
            metadata_id: Specific metadata ID (optional)
            model_name: Filter by model name (optional)
            stage: Filter by stage (optional)
            limit: Maximum number of results

        Returns:
            List of metadata records
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=extras.DictCursor)

            query = f"SELECT * FROM {self.schema}.ciaf_metadata WHERE 1=1"
            params = []

            if metadata_id:
                query += " AND metadata_id = %s"
                params.append(metadata_id)
            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)
            if stage:
                query += " AND stage = %s"
                params.append(stage)

            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            # Convert to list of dicts
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error retrieving metadata: {e}")
            return []
        finally:
            self.return_connection(conn)

    def store_receipt(
        self,
        receipt_id: str,
        model_name: str,
        receipt_data: Dict[str, Any],
        model_version: Optional[str] = None,
        query_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
    ) -> bool:
        """
        Store inference receipt in PostgreSQL.

        Args:
            receipt_id: Unique receipt identifier
            model_name: Model name
            receipt_data: Full receipt as JSON
            model_version: Model version (optional)
            query_hash: Hash of query (optional)
            output_hash: Hash of output (optional)

        Returns:
            True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self.schema}.ciaf_inference_receipts
                (receipt_id, model_name, model_version, query_hash, output_hash, receipt_data)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (receipt_id) DO NOTHING
            """,
                (
                    receipt_id,
                    model_name,
                    model_version,
                    query_hash,
                    output_hash,
                    json.dumps(receipt_data),
                ),
            )
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error storing receipt: {e}")
            return False
        finally:
            self.return_connection(conn)

    def retrieve_receipts(
        self,
        receipt_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve inference receipts from PostgreSQL.

        Args:
            receipt_id: Specific receipt ID (optional)
            model_name: Filter by model name (optional)
            limit: Maximum number of results

        Returns:
            List of receipt records
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=extras.DictCursor)

            query = f"SELECT * FROM {self.schema}.ciaf_inference_receipts WHERE 1=1"
            params = []

            if receipt_id:
                query += " AND receipt_id = %s"
                params.append(receipt_id)
            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)

            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error retrieving receipts: {e}")
            return []
        finally:
            self.return_connection(conn)

    def close(self):
        """Close all connections in the pool."""
        self.connection_pool.closeall()


def create_postgresql_vault(
    host: str = "localhost",
    port: int = 5432,
    database: str = "ciaf_vault",
    user: str = "ciaf_user",
    password: str = None,
    create_database: bool = False,
) -> PostgreSQLBackend:
    """
    Create a PostgreSQL vault backend with optional database creation.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        user: Database user
        password: Database password
        create_database: Whether to create the database if it doesn't exist

    Returns:
        PostgreSQLBackend instance
    """
    if create_database and PSYCOPG2_AVAILABLE:
        # Connect to postgres database to create our database
        try:
            conn = psycopg2.connect(
                host=host, port=port, database="postgres", user=user, password=password
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {database}")
                print(f"✅ Created database: {database}")

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"⚠️  Warning: Could not create database: {e}")

    return PostgreSQLBackend(
        host=host, port=port, database=database, user=user, password=password
    )


if __name__ == "__main__":
    # Example usage
    print("CIAF PostgreSQL Backend")
    print("=" * 50)

    if not PSYCOPG2_AVAILABLE:
        print("❌ psycopg2 not available. Install with:")
        print("   pip install psycopg2-binary")
    else:
        print("✅ psycopg2 available")
        print("\nTo use:")
        print("  from ciaf.vault.backends import create_postgresql_vault")
        print("  vault = create_postgresql_vault(")
        print("      host='localhost',")
        print("      database='ciaf_vault',")
        print("      user='ciaf_user',")
        print("      password='your_password'")
        print("  )")
