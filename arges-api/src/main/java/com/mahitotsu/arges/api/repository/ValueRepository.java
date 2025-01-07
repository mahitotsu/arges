package com.mahitotsu.arges.api.repository;

import java.sql.PreparedStatement;
import java.util.UUID;

import javax.sql.DataSource;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

@Repository
public class ValueRepository {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private DataSource dataSource;

    @Transactional
    public UUID insert(final int initialValue) {

        final KeyHolder keyHolder = new GeneratedKeyHolder();
        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        sqlClient.update(con -> {
            final PreparedStatement ps = con.prepareStatement("INSERT INTO v_table (value) VALUES (?)",
                    new String[] { "id" });
            ps.setInt(1, initialValue);
            return ps;
        }, keyHolder);

        final UUID generatedKey = keyHolder.getKeyAs(UUID.class);
        this.logger.debug("A new key is generated: {}", generatedKey);

        return generatedKey;
    }

    @Transactional(readOnly = true)
    public Integer get(final UUID key) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        return sqlClient.queryForObject("SELECT value FROM v_table WHERE id = ?", Integer.class, key);
    }

    @Retryable(retryFor = CannotAcquireLockException.class, maxAttempts = 32, backoff = @Backoff(delay = 10, multiplier = 1.2, random = true))
    @Transactional
    public Integer increment(final UUID key) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        final Integer currentValue = sqlClient.queryForObject("SELECT value FROM v_table WHERE id = ? FOR UPDATE",
                Integer.class, key);
        if (currentValue == null) {
            return null;
        }

        final Integer nextValue = currentValue + 1;
        sqlClient.update("UPDATE v_table SET value = ? WHERE id = ?", nextValue, key);
        return nextValue;
    }

    @Retryable(retryFor = CannotAcquireLockException.class, maxAttempts = 32, backoff = @Backoff(delay = 10, multiplier = 1.2, random = true))
    @Transactional
    public void increment2(final UUID key) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        sqlClient.update("INSERT INTO e_table (vid, value) VALUES (?, ?)", key, 1);
    }

    @Transactional(readOnly = true)
    public Integer get2(final UUID key) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        final Integer incremented = sqlClient.queryForObject("SELECT sum(value) FROM e_table WHERE vid = ?",
                Integer.class, key);
        final Integer initial = sqlClient.queryForObject("SELECT value FROM v_table WHERE id = ?", Integer.class, key);
        return incremented == null || initial == null ? null : incremented + initial;
    }
}
