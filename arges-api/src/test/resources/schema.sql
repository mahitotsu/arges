DROP TABLE IF EXISTS v_table;
DROP TABLE IF EXISTS e_table;

CREATE TABLE v_table (
    id UUID NOT NULL,
    value BIGINT NOT NULL,
    PRIMARY KEY (id)
);
CREATE TABLE e_table (
    vid UUID NOT NULL,
    eid TIMESTAMP DEFAULT transaction_timestamp(),
    value BIGINT NOT NULL,
    PRIMARY KEY (vid, eid)
);