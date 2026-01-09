-- PostgreSQL Initialization Script for R3MES
-- Creates optimized database schema with proper indexing

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create optimized indexes for performance
-- These will be created after tables are initialized by the application

-- Performance monitoring view
CREATE OR REPLACE VIEW connection_stats AS
SELECT 
    state,
    count(*) as connections,
    max(now() - state_change) as max_duration
FROM pg_stat_activity 
WHERE pid != pg_backend_pid()
GROUP BY state;

-- Grant permissions for monitoring
GRANT SELECT ON connection_stats TO PUBLIC;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'R3MES PostgreSQL database initialized successfully';
END $$;