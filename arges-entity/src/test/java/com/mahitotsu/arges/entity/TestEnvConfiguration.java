package com.mahitotsu.arges.entity;

import javax.sql.DataSource;

import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;

import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@SpringBootApplication
public class TestEnvConfiguration {

    private Region region = Region.US_EAST_1;

    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSourceProperties dataSourceProperties() {
        return new DataSourceProperties();
    }

    @Bean
    public DataSource dataSource(final DataSourceProperties dataSourceProperties) {

        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(this.region)
                .credentialsProvider(ProfileCredentialsProvider.create()).build();
        final String token = dsqlUtilities.generateDbConnectAdminAuthToken(
                builder -> builder.region(this.region).hostname(dataSourceProperties.getUrl().split("/")[2]));
        dataSourceProperties.setPassword(token);

        return dataSourceProperties.initializeDataSourceBuilder().build();
    }
    
}
