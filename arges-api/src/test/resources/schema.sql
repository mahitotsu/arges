DROP TABLE IF EXISTS v_table;
CREATE TABLE v_table (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    value BIGINT NOT NULL
);