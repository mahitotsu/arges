package com.mahitotsu.arges.api.repository;

import java.sql.PreparedStatement;
import java.util.Optional;
import java.util.UUID;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

@Repository
public class ValueRepository {

    @Autowired
    private DataSource dataSource;

    @Transactional
    public UUID insert(final int initialValue) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);
        final KeyHolder keyHolder = new GeneratedKeyHolder();

        sqlClient.update(con -> {
            final PreparedStatement ps = con.prepareStatement("INSERT INTO v_table (value) VALUES (?)",
                    PreparedStatement.RETURN_GENERATED_KEYS);
            ps.setInt(1, initialValue);
            return ps;
        }, keyHolder);

        return Optional.ofNullable(keyHolder.getKeys()).map(m -> m.get("id")).map(k -> UUID.class.cast(k)).orElse(null);
    }

    @Transactional(readOnly = true)
    public Integer get(final UUID key) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        return sqlClient.queryForObject("SELECT value FROM v_table WHERE id = ?", Integer.class, key);
    }

    @Retryable(retryFor = CannotAcquireLockException.class, maxAttempts = 16 )
    @Transactional
    public Integer increment(final UUID key) {

        final JdbcTemplate sqlClient = new JdbcTemplate(this.dataSource);

        final Integer currentValue = sqlClient.queryForObject("SELECT value FROM v_table WHERE id = ? FOR UPDATE", Integer.class, key);
        if (currentValue == null) {
            return null;
        }

        final Integer nextValue = currentValue + 1;
        System.out.println(currentValue + ", " + nextValue);
        sqlClient.update("UPDATE v_table SET value = ? WHERE id = ?", nextValue, key);
        return nextValue;
    }
}
