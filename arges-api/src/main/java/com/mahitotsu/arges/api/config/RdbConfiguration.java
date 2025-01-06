package com.mahitotsu.arges.api.config;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import com.zaxxer.hikari.HikariDataSource;

@Configuration
public class RdbConfiguration {

    @Value("${mahitotsu.dsql.token.expirationSeconds}")
    private int dsqlTokenExpirationSeconds;

    @ConfigurationProperties(prefix = "mahitotsu.dsql.connection.properties")
    @Bean
    @Qualifier("dsql.connection.properties")
    public Properties dsqlConnectionProperties() {
        return new Properties();
    }

    @ConfigurationProperties(prefix = "mahitotsu.dsql.endpoints")
    @Bean
    @Qualifier("dsql.endpoints")
    public List<String> dsqlEndpoints() {
        return new ArrayList<String>();
    }

    @Bean
    @Qualifier("rawDataSource")
    public DataSource auroraDsqlDataSource(
            @Qualifier("dsql.connection.properties") final Properties connecProperties,
            @Qualifier("dsql.endpoints") final List<String> dsqlEndpoints) {

        return new AuroraDSQLDataSource(connecProperties, dsqlEndpoints.toArray(new String[dsqlEndpoints.size()]));
    }

    @Bean
    @Primary
    public DataSource dataSource(@Qualifier("rawDataSource") final DataSource rawDataSource) {

        final HikariDataSource ds = new HikariDataSource();
        ds.setDataSource(rawDataSource);
        ds.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());
        ds.setMaxLifetime(Duration.ofSeconds(this.dsqlTokenExpirationSeconds).toMillis());
        ds.setIdleTimeout(0);
        ds.setMinimumIdle(1);
        ds.setAutoCommit(false);

        return ds;
    }
}
