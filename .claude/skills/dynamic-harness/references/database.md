# Database Setup Guide for Dynamic Harness Agent

Setup for the optional storage layers behind the agent. **Build order matters:** the agent works end-to-end with only the in-memory `SessionMemory` from `implementation.md`. Add layers later, in this order: (1) Redis if you need multi-process sessions, (2) Postgres for the audit trail, (3) vector DB for retrieval, (4) analytics.

Security: never commit real credentials. Put them in `.env` (git-ignored) and substitute the placeholders below.

---

## Layer 1: Session Memory (Redis)

### Installation

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**Docker:**
```bash
docker run -d --name agent-redis -p 6379:6379 redis:latest
```

### Python Client Setup

```bash
pip install redis
```

### Configuration

```python
# config.py
import redis
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TTL_HOURS = 24

redis_client = redis.from_url(REDIS_URL)

# Test connection
try:
    redis_client.ping()
    print("✓ Redis connected")
except redis.ConnectionError:
    print("✗ Redis connection failed")
```

### Key Naming Convention

```
session:{session_id}:task          # Task object
session:{session_id}:plan          # Execution plan
session:{session_id}:harness       # All harnesses (step_id -> harness)
session:{session_id}:execution_log # Execution logs (step_id -> log)
session:{session_id}:review        # Plan review decision
session:{session_id}:state         # Current execution state
```

---

## Layer 2: Temporal Event Store (PostgreSQL)

### Installation

**macOS (Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Docker:**
```bash
docker run -d --name agent-postgres \
  -e POSTGRES_USER=agent \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  -e POSTGRES_DB=agent_db \
  -p 5432:5432 \
  postgres:16
```

### Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE agent_db OWNER postgres;

# Connect to new database
\c agent_db

# (JSONB is a built-in type in PostgreSQL; no extension needed)

# Exit
\q
```

### Schema Creation

```sql
-- Save the SQL in this section as database_setup.sql (used by the setup script)
-- session_id is TEXT because ids look like task-2026-11-06-ab12cd

-- Main events table (immutable append-only log)
CREATE TABLE IF NOT EXISTS agent_events (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  event_type VARCHAR(100) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  model_used VARCHAR(100),
  
  -- Event details stored as JSONB for flexibility
  event_data JSONB NOT NULL,
  tool_calls JSONB,
  validation_results JSONB,
  confidence_score FLOAT,
  
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_agent_events_session_id 
  ON agent_events(session_id);

CREATE INDEX IF NOT EXISTS idx_agent_events_timestamp 
  ON agent_events(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_agent_events_event_type 
  ON agent_events(event_type);

CREATE INDEX IF NOT EXISTS idx_agent_events_session_timestamp 
  ON agent_events(session_id, timestamp DESC);


-- Task summaries (denormalized for fast queries)
CREATE TABLE IF NOT EXISTS task_summaries (
  session_id TEXT PRIMARY KEY,
  user_prompt TEXT NOT NULL,
  task_category VARCHAR(100),
  status VARCHAR(50),  -- completed, failed, rolled_back, needs_human_review
  
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  duration_seconds INT,
  
  plan_version INT,
  harness_adjustments INT,
  verification_passed BOOLEAN,
  
  final_confidence FLOAT,
  model_calls INT,
  tool_calls_total INT,
  retries INT,
  
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_task_summaries_status 
  ON task_summaries(status);

CREATE INDEX IF NOT EXISTS idx_task_summaries_category 
  ON task_summaries(task_category);

CREATE INDEX IF NOT EXISTS idx_task_summaries_completed_at 
  ON task_summaries(completed_at DESC);


-- Execution steps (detailed log per step)
CREATE TABLE IF NOT EXISTS execution_steps (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  step_id INT NOT NULL,
  
  goal TEXT,
  status VARCHAR(50),  -- executing, completed, failed, rolled_back
  
  tool_calls JSONB,  -- array of tool call objects
  guardrail_checks JSONB,
  outputs JSONB,
  
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  duration_ms INT,
  
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_execution_steps_session_step 
  ON execution_steps(session_id, step_id);

CREATE INDEX IF NOT EXISTS idx_execution_steps_status 
  ON execution_steps(status);


-- Verification results (what verifier decided)
CREATE TABLE IF NOT EXISTS verifications (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  step_id INT NOT NULL,
  
  verified BOOLEAN,
  findings JSONB,
  passed_criteria TEXT[],
  failed_criteria TEXT[],
  
  next_action VARCHAR(50),  -- proceed, retry, rollback, escalate
  
  verified_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_verifications_session_step 
  ON verifications(session_id, step_id);

CREATE INDEX IF NOT EXISTS idx_verifications_verified 
  ON verifications(verified);
```

### Python Client Setup

```bash
pip install psycopg2-binary
```

### Configuration

```python
# config.py
import psycopg2
from urllib.parse import urlparse
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_db"
)

def get_db_connection():
    """Get a new database connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        print(f"✗ Database connection failed: {e}")
        raise

# Test connection
try:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    print("✓ PostgreSQL connected")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"✗ PostgreSQL error: {e}")
```

### Queries Reference

```sql
-- Find all tasks completed in last 24 hours
SELECT session_id, user_prompt, status, final_confidence
FROM task_summaries
WHERE completed_at > NOW() - INTERVAL '24 hours'
ORDER BY completed_at DESC;

-- Find all failed tasks
SELECT session_id, user_prompt, status, retries
FROM task_summaries
WHERE status IN ('failed', 'rolled_back')
ORDER BY completed_at DESC;

-- Get full event timeline for a session
SELECT event_type, timestamp, model_used, confidence_score
FROM agent_events
WHERE session_id = 'task-2026-11-06-001'
ORDER BY timestamp ASC;

-- Find tasks with low confidence
SELECT session_id, user_prompt, final_confidence
FROM task_summaries
WHERE final_confidence < 0.70
AND completed_at > NOW() - INTERVAL '7 days'
ORDER BY final_confidence ASC;

-- Average confidence vs success rate
SELECT
  status,
  COUNT(*) as task_count,
  AVG(final_confidence) as avg_confidence,
  SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM task_summaries
GROUP BY status;
```

---

## Layer 3: Vector Database (Pinecone)

### Setup

**Install client:**
```bash
pip install pinecone
```

**Create Pinecone account:**
1. Go to https://www.pinecone.io/
2. Sign up for free account
3. Create a new project
4. Get your API key and environment

### Configuration

```python
# config.py
import os
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
index_name = "agent-tasks"
try:
    index = pc.Index(index_name)
    print(f"✓ Pinecone index '{index_name}' connected")
except:
    print(f"✗ Index '{index_name}' not found")
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": PINECONE_ENVIRONMENT
            }
        }
    )
    print(f"✓ Index created")
```

### Usage Example

```python
from anthropic import Anthropic

class VectorDB:
    def __init__(self, pc_index):
        self.index = pc_index
        self.client = Anthropic()
    
    def embed_text(self, text: str) -> list:
        """Get embedding using Claude (or external service)."""
        # For production, use OpenAI or dedicated embedding service
        # This is a placeholder—Claude isn't an embedding API
        # Use: OpenAI's text-embedding-3-small or similar
        return [0.1] * 1536  # Placeholder - MUST be replaced before real use
    
    def upsert_task(self, session_id: str, task_summary: str, metadata: dict):
        """Index a new task."""
        embedding = self.embed_text(task_summary)
        
        self.index.upsert(
            vectors=[{
                "id": session_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
        print(f"✓ Task {session_id} indexed")
    
    def find_similar_tasks(self, query: str, top_k: int = 5) -> list:
        """Find similar tasks."""
        query_embedding = self.embed_text(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results["matches"]

# Usage
from pinecone import Pinecone
pc = Pinecone(api_key="...")
vector_db = VectorDB(pc.Index("agent-tasks"))

# Index a task
vector_db.upsert_task(
    "task-2026-11-06-001",
    "Refactor auth module to support OAuth2",
    {
        "task_type": "code_refactor",
        "complexity": 4,
        "success": True,
        "confidence": 0.82
    }
)

# Find similar tasks
similar = vector_db.find_similar_tasks("Add OAuth2 support to authentication")
```

### Alternative: Qdrant (Self-Hosted)

**Docker setup:**
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Python client:**
```bash
pip install qdrant-client
```

**Usage:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create collection
client.recreate_collection(
    collection_name="agent-tasks",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Upsert points
client.upsert(
    collection_name="agent-tasks",
    points=[
        PointStruct(
            id="task-1",
            vector=[0.1, 0.2, ...],
            payload={
                "task_type": "code_refactor",
                "success": True
            }
        )
    ]
)

# Search
results = client.search(
    collection_name="agent-tasks",
    query_vector=[0.1, 0.2, ...],
    limit=5
)
```

---

## Layer 4: Analytics Database (Optional)

### Setup for Metrics

```sql
-- Analytics table
CREATE TABLE IF NOT EXISTS agent_metrics (
  session_id TEXT PRIMARY KEY,
  
  -- Task info
  task_category VARCHAR(100),
  created_date DATE,
  
  -- Execution metrics
  status VARCHAR(50),
  completed BOOLEAN,
  time_to_completion_seconds INT,
  
  -- Confidence tracking
  initial_confidence FLOAT,
  confidence_after_analysis FLOAT,
  confidence_after_review FLOAT,
  final_confidence FLOAT,
  
  -- Harness effectiveness
  harness_versions INT,
  verification_attempts INT,
  retries INT,
  rollbacks INT,
  
  -- Model performance
  analysis_accuracy FLOAT,
  review_effectiveness FLOAT,
  verification_accuracy FLOAT,
  
  -- Resource usage
  total_model_calls INT,
  model_calls_analysis INT,
  model_calls_review INT,
  model_calls_execution INT,
  model_calls_verification INT,
  total_tool_calls INT,
  
  -- Outcome
  success BOOLEAN,
  failure_reason VARCHAR(255),
  required_human_intervention BOOLEAN,
  
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_status ON agent_metrics(status);
CREATE INDEX IF NOT EXISTS idx_metrics_category ON agent_metrics(task_category);
CREATE INDEX IF NOT EXISTS idx_metrics_success ON agent_metrics(success);
```

---

## Complete Setup Script

```bash
#!/bin/bash
# setup_databases.sh - Complete database setup
# Requires: export POSTGRES_PASSWORD=<choose one> before running

echo "🚀 Setting up Dynamic Harness Agent databases..."

# 1. Redis
echo "1️⃣  Setting up Redis..."
docker run -d --name agent-redis -p 6379:6379 redis:latest
echo "✓ Redis running on localhost:6379"

# 2. PostgreSQL
echo "2️⃣  Setting up PostgreSQL..."
docker run -d --name agent-postgres \
  -e POSTGRES_USER=agent \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  -e POSTGRES_DB=agent_db \
  -p 5432:5432 \
  postgres:16

echo "✓ PostgreSQL running on localhost:5432"

# Wait for PostgreSQL to start
sleep 10

# 3. Create PostgreSQL schema
echo "3️⃣  Creating PostgreSQL schema..."
PGPASSWORD="$POSTGRES_PASSWORD" psql -h localhost -U agent -d agent_db -f database_setup.sql
echo "✓ Schema created"

# 4. Create Pinecone index (via API)
echo "4️⃣  Pinecone setup..."
echo "⚠️  Pinecone requires manual setup:"
echo "   1. Visit https://www.pinecone.io/"
echo "   2. Create free account"
echo "   3. Create index 'agent-tasks' with dimension 1536"
echo "   4. Set PINECONE_API_KEY environment variable"

# 5. Environment file
echo "5️⃣  Creating .env file..."
cat > .env << EOF
# Database connections
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_db

# Pinecone (set after creating index)
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Anthropic
ANTHROPIC_API_KEY=your_api_key_here
EOF

echo "✓ .env file created"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Set ANTHROPIC_API_KEY in .env"
echo "   2. Set PINECONE_API_KEY in .env (after creating Pinecone index)"
echo "   3. Run: python -m agent"
```

**Run it:**
```bash
chmod +x setup_databases.sh
./setup_databases.sh
```

---

## Environment Configuration

```bash
# .env file
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_db
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=us-east-1
ANTHROPIC_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

---

## Health Checks

```python
# health_check.py
import redis
import psycopg2
from pinecone import Pinecone

def check_databases():
    """Verify all databases are accessible."""
    
    checks = {}
    
    # Redis
    try:
        r = redis.from_url("redis://localhost:6379/0")
        r.ping()
        checks["redis"] = "✓"
    except Exception as e:
        checks["redis"] = f"✗ {e}"
    
    # PostgreSQL
    try:
        conn = psycopg2.connect(
            "postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        checks["postgresql"] = "✓"
    except Exception as e:
        checks["postgresql"] = f"✗ {e}"
    
    # Pinecone
    try:
        pc = Pinecone(api_key="your_api_key")
        index = pc.Index("agent-tasks")
        index.describe_index_stats()
        checks["pinecone"] = "✓"
    except Exception as e:
        checks["pinecone"] = f"✗ {e}"
    
    # Print results
    print("\n📊 Database Health Check:")
    for name, status in checks.items():
        print(f"  {name:15} {status}")
    
    return all("✓" in v for v in checks.values())

if __name__ == "__main__":
    all_good = check_databases()
    exit(0 if all_good else 1)
```

---

## Backup & Recovery

```bash
# Backup PostgreSQL
pg_dump -h localhost -U agent agent_db > backup.sql

# Restore PostgreSQL
psql -h localhost -U agent agent_db < backup.sql

# Backup Redis (RDB)
redis-cli --rdb /path/to/backup.rdb

# View Pinecone backup (automatic)
# Pinecone provides automatic backups in managed service
```

---

## Monitoring

```python
# monitor_databases.py
import redis
import psycopg2
from datetime import datetime

def monitor_postgres():
    """Monitor PostgreSQL stats."""
    conn = psycopg2.connect("postgresql://agent:${POSTGRES_PASSWORD}@localhost:5432/agent_db")
    cursor = conn.cursor()
    
    # Row counts
    tables = ["agent_events", "task_summaries", "execution_steps", "verifications"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:20} {count:10,} rows")
    
    # Disk usage
    cursor.execute("""
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
        FROM pg_tables
        WHERE schemaname='public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        LIMIT 5
    """)
    
    print("\n  Top 5 tables by size:")
    for schema, table, size in cursor.fetchall():
        print(f"    {table:20} {size}")
    
    cursor.close()
    conn.close()

def monitor_redis():
    """Monitor Redis stats."""
    r = redis.from_url("redis://localhost:6379/0")
    info = r.info()
    
    print(f"\n  Memory used:       {info['used_memory_human']}")
    print(f"  Connected clients: {info['connected_clients']}")
    print(f"  Total commands:    {info['total_commands_processed']}")

if __name__ == "__main__":
    print(f"\n📊 Database Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPostgreSQL:")
    monitor_postgres()
    print("\nRedis:")
    monitor_redis()
```

---

This guide provides everything needed to set up and manage all four database layers for your dynamic harness agent.
