package com.mahitotsu.arges.api.config;

import javax.sql.DataSource;

import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.zaxxer.hikari.HikariDataSource;

import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class RdbConfiguration {

    @Bean
    public DataSource dataSource(final DataSourceProperties dataSourceProperties) {

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
}
