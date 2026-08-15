BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mindbridge_runtime') THEN
        CREATE ROLE mindbridge_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
    ELSIF EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = 'mindbridge_runtime' AND (rolsuper OR rolbypassrls)
    ) THEN
        RAISE EXCEPTION 'mindbridge_runtime must not bypass row-level security';
    END IF;
    IF current_user <> 'mindbridge_runtime' THEN
        EXECUTE format('GRANT mindbridge_runtime TO %I', current_user);
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO mindbridge_runtime;

DO $$
DECLARE
    protected_table text;
BEGIN
    FOR protected_table IN
        SELECT column_name.table_name
        FROM information_schema.columns AS column_name
        WHERE column_name.table_schema = 'public'
          AND column_name.column_name = 'tenant_id'
        ORDER BY column_name.table_name
    LOOP
        EXECUTE format(
            'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %I TO mindbridge_runtime',
            protected_table
        );
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', protected_table);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', protected_table);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I '
            'USING (tenant_id = current_setting(''mindbridge.tenant_id'', true)) '
            'WITH CHECK (tenant_id = current_setting(''mindbridge.tenant_id'', true))',
            protected_table
        );
    END LOOP;
END
$$;

INSERT INTO schema_migrations (version) VALUES (5);

COMMIT;
