package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.*;

import java.sql.Connection;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ConfigurableApplicationContext;

public class ApiTest extends TestBase {

    @Autowired(required = false)
    private ConfigurableApplicationContext application;

    @Autowired(required = false)
    private DataSource dataSource;

    @Test
    public void testAppStatus() {

        assertNotNull(this.application);
        assertTrue(this.application.isActive());
        assertTrue(this.application.isRunning());
    }

    @Test
    public void testDataSourceAvailability() {

        assertNotNull(this.dataSource);
        try (final Connection con = this.dataSource.getConnection()) {
            assertNotNull(con);
            assertFalse(con.isReadOnly());
        } catch (SQLException e) {
            fail(e);
        }
    }
}
