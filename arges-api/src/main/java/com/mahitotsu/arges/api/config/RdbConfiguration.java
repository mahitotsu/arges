package com.mahitotsu.arges.api.config;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.sql.DataSource;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.datasource.AbstractDataSource;
import org.springframework.jdbc.datasource.DriverManagerDataSource;

import com.zaxxer.hikari.HikariDataSource;

import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

@Configuration
public class RdbConfiguration {

    @Bean
    public DataSource dataSource() {

        final Map<String, String> dsqlEndpoints = Stream.of(
            "4iabtwnq2j55iez4j4bkykghgm.dsql.us-east-1.on.aws", 
            "s4abtwnq2jebk7aj6vhlsb2coi.dsql.us-east-2.on.aws"
        ).collect(Collectors.toMap(
                endpoint -> endpoint.split("\\.")[2],
                endpoint -> endpoint));
        final Map<String, DataSource> dsqlClusters = dsqlEndpoints.entrySet().stream()
                .collect(Collectors.toMap(
                        entry -> entry.getKey(),
                        entry -> this.buildRegionalDataSource(entry.getValue(), entry.getKey())));

        final AbstractDataSource dsqlDS = new AbstractDataSource() {

            private final AtomicInteger counter = new AtomicInteger(0);
            private final String[] regions = dsqlClusters.keySet().toArray(new String[dsqlClusters.size()]);
            private final ConcurrentMap<String, String> tokenCache = new ConcurrentHashMap<>();

            @Override
            public Connection getConnection() throws SQLException {
                return this.getConnection("admin", null); // with dummy parameters
            }

            @Override
            public Connection getConnection(final String username, final String password) throws SQLException {

                final int index = this.counter.getAndUpdate(v -> v == Integer.MAX_VALUE ? 0 : v + 1);
                final String region = this.regions[index % this.regions.length];

                final String endpoint = dsqlEndpoints.get(region);
                final String token = this.tokenCache.compute(endpoint,
                        (k, v) -> v == null ? generateAuthToken(endpoint, region) : v);

                System.out.println(region);
                return dsqlClusters.get(region).getConnection(username, token);
            }
        };

        final HikariDataSource ds = new HikariDataSource();
        ds.setDataSource(dsqlDS);
        ds.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());

        return ds;
    }

    private DataSource buildRegionalDataSource(final String endpoint, final String region) {

        final DriverManagerDataSource ds = new DriverManagerDataSource();
        ds.setUrl("jdbc:postgresql://" + endpoint + "/postgres?ssl=required");

        return ds;
    }

    private String generateAuthToken(final String endpoint, final String region) {

        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(Region.of(region))
                .credentialsProvider(DefaultCredentialsProvider.create()).build();
        final String token = dsqlUtilities.generateDbConnectAdminAuthToken(builder -> builder.hostname(endpoint));

        System.out.println(token);
        return token;
    }
}
