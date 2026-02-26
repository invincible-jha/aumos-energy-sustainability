-- ============================================================================
-- Migration 001: Initial schema for aumos-energy-sustainability
-- Table prefix: esg_
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- esg_carbon_records — per-inference carbon footprint records
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS esg_carbon_records (
    id                              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id                       UUID NOT NULL,
    inference_id                    UUID NOT NULL,
    model_id                        VARCHAR(255) NOT NULL,
    region                          VARCHAR(100) NOT NULL,
    energy_kwh                      DOUBLE PRECISION NOT NULL,
    carbon_intensity_gco2_per_kwh   DOUBLE PRECISION NOT NULL,
    carbon_gco2                     DOUBLE PRECISION NOT NULL,
    renewable_percentage            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    tokens_input                    INTEGER,
    tokens_output                   INTEGER,
    inference_duration_ms           INTEGER,
    metadata                        JSONB NOT NULL DEFAULT '{}',
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_esg_carbon_records_tenant_id
    ON esg_carbon_records (tenant_id);
CREATE INDEX IF NOT EXISTS idx_esg_carbon_records_inference_id
    ON esg_carbon_records (inference_id);
CREATE INDEX IF NOT EXISTS idx_esg_carbon_records_model_id
    ON esg_carbon_records (model_id);
CREATE INDEX IF NOT EXISTS idx_esg_carbon_records_region
    ON esg_carbon_records (region);
CREATE INDEX IF NOT EXISTS idx_esg_carbon_records_created_at
    ON esg_carbon_records (created_at DESC);
-- Composite for reporting queries
CREATE INDEX IF NOT EXISTS idx_esg_carbon_records_tenant_created
    ON esg_carbon_records (tenant_id, created_at DESC);

-- Row-level security for tenant isolation
ALTER TABLE esg_carbon_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY esg_carbon_records_tenant_isolation ON esg_carbon_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- ---------------------------------------------------------------------------
-- esg_energy_profiles — regional energy source profiles
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS esg_energy_profiles (
    id                              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id                       UUID NOT NULL,
    region                          VARCHAR(100) NOT NULL,
    display_name                    VARCHAR(255) NOT NULL DEFAULT '',
    carbon_intensity_gco2_per_kwh   DOUBLE PRECISION NOT NULL,
    renewable_percentage            DOUBLE PRECISION NOT NULL,
    solar_percentage                DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    wind_percentage                 DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    hydro_percentage                DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    nuclear_percentage              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    estimated_latency_ms            INTEGER NOT NULL DEFAULT 50,
    last_refreshed_at               TIMESTAMPTZ,
    is_active                       BOOLEAN NOT NULL DEFAULT TRUE,
    source_metadata                 JSONB NOT NULL DEFAULT '{}',
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_esg_energy_profiles_tenant_region UNIQUE (tenant_id, region)
);

CREATE INDEX IF NOT EXISTS idx_esg_energy_profiles_tenant_id
    ON esg_energy_profiles (tenant_id);
CREATE INDEX IF NOT EXISTS idx_esg_energy_profiles_region
    ON esg_energy_profiles (region);

ALTER TABLE esg_energy_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY esg_energy_profiles_tenant_isolation ON esg_energy_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- ---------------------------------------------------------------------------
-- esg_routing_decisions — workload routing decisions based on energy profiles
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS esg_routing_decisions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           UUID NOT NULL,
    workload_id         UUID NOT NULL,
    workload_type       VARCHAR(100) NOT NULL,
    selected_region     VARCHAR(100) NOT NULL,
    selected_profile_id UUID REFERENCES esg_energy_profiles (id) ON DELETE SET NULL,
    candidate_regions   JSONB NOT NULL DEFAULT '[]',
    renewable_score     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    latency_score       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    composite_score     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    carbon_saved_gco2   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    override_reason     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_esg_routing_decisions_tenant_id
    ON esg_routing_decisions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_esg_routing_decisions_workload_id
    ON esg_routing_decisions (workload_id);
CREATE INDEX IF NOT EXISTS idx_esg_routing_decisions_workload_type
    ON esg_routing_decisions (workload_type);
CREATE INDEX IF NOT EXISTS idx_esg_routing_decisions_selected_region
    ON esg_routing_decisions (selected_region);
CREATE INDEX IF NOT EXISTS idx_esg_routing_decisions_created_at
    ON esg_routing_decisions (created_at DESC);

ALTER TABLE esg_routing_decisions ENABLE ROW LEVEL SECURITY;
CREATE POLICY esg_routing_decisions_tenant_isolation ON esg_routing_decisions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- ---------------------------------------------------------------------------
-- esg_sustainability_reports — generated ESG sustainability reports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS esg_sustainability_reports (
    id                          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id                   UUID NOT NULL,
    title                       VARCHAR(255) NOT NULL,
    report_type                 VARCHAR(50) NOT NULL DEFAULT 'quarterly',
    period_start                TIMESTAMPTZ NOT NULL,
    period_end                  TIMESTAMPTZ NOT NULL,
    status                      VARCHAR(20) NOT NULL DEFAULT 'generating',
    total_inferences            INTEGER NOT NULL DEFAULT 0,
    total_energy_kwh            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_carbon_kg_co2         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    average_renewable_percentage DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    carbon_saved_kg_co2         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    routing_optimisation_rate   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    per_model_breakdown         JSONB NOT NULL DEFAULT '{}',
    per_region_breakdown        JSONB NOT NULL DEFAULT '{}',
    esg_score                   DOUBLE PRECISION,
    generated_at                TIMESTAMPTZ,
    error_message               TEXT,
    requested_by                UUID,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_esg_sustainability_reports_tenant_id
    ON esg_sustainability_reports (tenant_id);
CREATE INDEX IF NOT EXISTS idx_esg_sustainability_reports_report_type
    ON esg_sustainability_reports (report_type);
CREATE INDEX IF NOT EXISTS idx_esg_sustainability_reports_status
    ON esg_sustainability_reports (status);
CREATE INDEX IF NOT EXISTS idx_esg_sustainability_reports_period_start
    ON esg_sustainability_reports (period_start);
CREATE INDEX IF NOT EXISTS idx_esg_sustainability_reports_created_at
    ON esg_sustainability_reports (created_at DESC);

ALTER TABLE esg_sustainability_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY esg_sustainability_reports_tenant_isolation ON esg_sustainability_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- ---------------------------------------------------------------------------
-- esg_optimizations — energy optimization recommendations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS esg_optimizations (
    id                          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id                   UUID NOT NULL,
    category                    VARCHAR(50) NOT NULL,
    title                       VARCHAR(255) NOT NULL,
    description                 TEXT NOT NULL,
    target_resource             VARCHAR(255),
    projected_savings_kg_co2    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    projected_savings_kwh       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    priority                    VARCHAR(20) NOT NULL DEFAULT 'medium',
    status                      VARCHAR(20) NOT NULL DEFAULT 'active',
    implementation_effort       VARCHAR(20) NOT NULL DEFAULT 'medium',
    evidence                    JSONB NOT NULL DEFAULT '{}',
    implemented_at              TIMESTAMPTZ,
    expires_at                  TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_esg_optimizations_tenant_id
    ON esg_optimizations (tenant_id);
CREATE INDEX IF NOT EXISTS idx_esg_optimizations_category
    ON esg_optimizations (category);
CREATE INDEX IF NOT EXISTS idx_esg_optimizations_priority
    ON esg_optimizations (priority);
CREATE INDEX IF NOT EXISTS idx_esg_optimizations_status
    ON esg_optimizations (status);
CREATE INDEX IF NOT EXISTS idx_esg_optimizations_savings
    ON esg_optimizations (projected_savings_kg_co2 DESC);

ALTER TABLE esg_optimizations ENABLE ROW LEVEL SECURITY;
CREATE POLICY esg_optimizations_tenant_isolation ON esg_optimizations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
