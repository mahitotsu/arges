package com.mahitotsu.arges.api.config;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.datasource.lookup.AbstractRoutingDataSource;
import org.springframework.lang.Nullable;

import com.zaxxer.hikari.HikariDataSource;

import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class RdbConfiguration {

    @Bean
    @ConfigurationProperties(prefix = "spring.a.datasource")
    @Qualifier("a")
    public DataSourceProperties dataSourcePropertiesA() {
        return new DataSourceProperties();
    }

    @Bean
    @ConfigurationProperties(prefix = "spring.b.datasource")
    @Qualifier("b")
    public DataSourceProperties dataSourcePropertiesB() {
        return new DataSourceProperties();
    }

    @Bean
    @Qualifier("a")
    public DataSource dataSourceA(@Qualifier("a") final DataSourceProperties dataSourceProperties) {

        final String endpoint = dataSourceProperties.getUrl().split("/")[2];
        final String region = endpoint.split("\\.")[2];
        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(Region.US_EAST_1)
                .credentialsProvider(DefaultCredentialsProvider.create()).region(Region.of(region)).build();
        final String token = dsqlUtilities.generateDbConnectAdminAuthToken(builder -> builder.hostname(endpoint));
        dataSourceProperties.setPassword(token);

        final HikariDataSource ds = dataSourceProperties.initializeDataSourceBuilder().type(HikariDataSource.class)
                .build();
        ds.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());

        return ds;
    }

    @Bean
    @Qualifier("b")
    public DataSource dataSourceB(@Qualifier("b") final DataSourceProperties dataSourceProperties) {

        final String endpoint = dataSourceProperties.getUrl().split("/")[2];
        final String region = endpoint.split("\\.")[2];
        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(Region.US_EAST_1)
                .credentialsProvider(DefaultCredentialsProvider.create()).region(Region.of(region)).build();
        final String token = dsqlUtilities.generateDbConnectAdminAuthToken(builder -> builder.hostname(endpoint));
        dataSourceProperties.setPassword(token);

        final HikariDataSource ds = dataSourceProperties.initializeDataSourceBuilder().type(HikariDataSource.class)
                .build();
        ds.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());

        return ds;
    }

    @Bean
    @Primary
    public DataSource dataSource(@Qualifier("a") final DataSource dataSourceA,
            @Qualifier("b") final DataSource dataSourceB) {

        final AbstractRoutingDataSource dataSource = new AbstractRoutingDataSource() {
            private final AtomicBoolean selector = new AtomicBoolean();

            @Override
            @Nullable
            protected Object determineCurrentLookupKey() {
                final boolean current = this.selector.getAndSet(!this.selector.get());
                return current ? "a" : "b";
            }
        };

        dataSource.setTargetDataSources(Map.of("a", dataSourceA, "b", dataSourceB));

        return dataSource;
    }
}
