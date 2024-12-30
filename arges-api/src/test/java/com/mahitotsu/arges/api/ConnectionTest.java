package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.assertNotSame;

import java.sql.Connection;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

public class ConnectionTest extends TestBase {

    @Autowired
    private DataSource dataSource;
    
    @Test
    public void testRoundRobin() throws SQLException {

        final Connection con1 = this.dataSource.getConnection();
        final Connection con2 = this.dataSource.getConnection();

        assertNotSame(con1, con2);
    }
}
