package com.mahitotsu.arges.api.configs;

import javax.sql.DataSource;

import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class RdbConfiguration {

    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSourceProperties dataSourceProperties() {
        return new DataSourceProperties();
    }

    @Bean
    public DataSource dataSource(final DataSourceProperties dataSourceProperties) {

        final String endpoint = dataSourceProperties.getUrl().split("/")[2];
        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(Region.US_EAST_1)
                .credentialsProvider(ProfileCredentialsProvider.create()).build();
        final String token = dsqlUtilities.generateDbConnectAdminAuthToken(builder -> builder.hostname(endpoint));
        dataSourceProperties.setPassword(token);

        return dataSourceProperties.initializeDataSourceBuilder().build();
    }
}
