package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.*;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

public class RdbConfigTest extends TestBase {

    @Autowired
    private DataSource dataSource;

    @Test
    public void testMultiDataSource() {

        try (
                final Connection con1 = this.dataSource.getConnection();
                final Connection con2 = this.dataSource.getConnection();) {

            assertNotNull(con1);
            assertNotNull(con2);
            assertNotSame(con1, con2);

            final String url1 = this.getUrl(con1);
            final String url2 = this.getUrl(con2);
            assertNotNull(url1);
            assertNotNull(url2);
            assertFalse(url1.equals(url2));

            final boolean tblExist1 = this.hasTable(con1, "ut_table");
            final boolean tblExist2 = this.hasTable(con2, "ut_table");
            assertTrue(tblExist1);
            assertTrue(tblExist2);

        } catch (SQLException e) {
            fail(e);
        }
    }

    private String getUrl(final Connection con) throws SQLException {
        return con.getMetaData().getURL();
    }

    private boolean hasTable(final Connection con, final String tableName) throws SQLException {
        final ResultSet rs = con.getMetaData().getTables(null, null, tableName, new String[] { "TABLE" });
        if (rs.next()) {
            return true;
        } else {
            return false;
        }
    }
}
