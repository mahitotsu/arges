package com.mahitotsu.arges.api.config;

import java.sql.SQLException;

import com.zaxxer.hikari.SQLExceptionOverride;

public class DsqlExceptionOverride implements SQLExceptionOverride {

    @java.lang.Override
    public Override adjudicate(final SQLException ex) {

        final String sqlState = ex.getSQLState();

        if (sqlState != null
                && (sqlState.equals("0C000") || sqlState.equals("0C001") || sqlState.matches("0A\\d{3}"))) {
            return SQLExceptionOverride.Override.DO_NOT_EVICT;
        }

        return Override.CONTINUE_EVICT;
    }
}
