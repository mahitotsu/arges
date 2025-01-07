DROP TABLE IF EXISTS v_table;
DROP TABLE IF EXISTS e_table;

CREATE OR REPLACE FUNCTION uuid_generate_v7() RETURNS UUID AS $$
  SELECT (
    lpad(to_hex((extract(epoch FROM clock_timestamp()) * 1000)::BIGINT), 12, '0')
    || to_hex(((random() * 65536)::INT & 0x0F) | 0x70) 
    || lpad(to_hex((random() * (2^32 - 1))::BIGINT), 8, '0')
    || lpad(to_hex((random() * (2^32 - 1))::BIGINT), 8, '0')
  )::uuid
$$
 LANGUAGE SQL;

CREATE OR REPLACE FUNCTION uuid_generate_v7() RETURNS UUID AS $$
    SELECT (
      substring(raw_uuid FROM 1 FOR 8) || '-' ||
      substring(raw_uuid FROM 9 FOR 4) || '-' ||
      substring(raw_uuid FROM 13 FOR 4) || '-' ||
      substring(raw_uuid FROM 17 FOR 4) || '-' ||
      substring(raw_uuid FROM 21)
    )::UUID 
    FROM (
      SELECT (
        lpad(to_hex((extract(epoch FROM clock_timestamp()) * 1000)::BIGINT), 12, '0') ||
        lpad(to_hex(((random() * 65536)::INT & 0x0FFF) | 0x7000), 4, '0') ||
        lpad(to_hex((random() * (2^32 - 1))::BIGINT), 8, '0') ||
        lpad(to_hex((random() * (2^32 - 1))::BIGINT), 8, '0')
      ) AS raw_uuid
    )
$$
 LANGUAGE SQL;

CREATE TABLE v_table (
    id UUID NOT NULL DEFAULT uuid_generate_v7(),
    value BIGINT NOT NULL,
    PRIMARY KEY (id)
);
CREATE TABLE e_table (
    vid UUID NOT NULL,
    eid UUID NOT NULL DEFAULT uuid_generate_v7(),
    value BIGINT NOT NULL,
    PRIMARY KEY (vid, eid)
);