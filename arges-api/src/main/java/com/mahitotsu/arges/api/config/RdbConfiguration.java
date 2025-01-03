package com.mahitotsu.arges.api.config;

import java.sql.Connection;
import java.sql.SQLException;
import java.time.Duration;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.sql.DataSource;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Bean
    public DataSource dataSource() {

        final Map<String, String> dsqlEndpoints = Stream.of(
                "4iabtwnq2j55iez4j4bkykghgm.dsql.us-east-1.on.aws",
                "s4abtwnq2jebk7aj6vhlsb2coi.dsql.us-east-2.on.aws").collect(
                        Collectors.toMap(
                                endpoint -> endpoint.split("\\.")[2],
                                endpoint -> endpoint));

        final int loginTimeout = 5;
        final int connectionTImeout = 2;
        final Properties props = new Properties();
        props.setProperty("sslmode", "require");
        props.setProperty("loginTimeout ", String.valueOf(loginTimeout));
        props.setProperty("connectionTimeout", String.valueOf(connectionTImeout));

        final Map<String, DataSource> dsqlClusters = dsqlEndpoints.entrySet().stream()
                .collect(Collectors.toMap(
                        entry -> entry.getKey(),
                        entry -> new DriverManagerDataSource("jdbc:postgresql://" + entry.getValue() + "/postgres",
                                props)));

        final AbstractDataSource dsqlDS = new AbstractDataSource() {

            class CacheEntry {

                CacheEntry(final String token, final long expiresAt) {
                    this.token = token;
                    this.expiresAt = expiresAt;
                }

                private String token;
                private long expiresAt;

                boolean isExpired(final long datetime) {
                    return this.expiresAt < datetime;
                }

                String getToken() {
                    return this.token;
                }
            }

            private final AtomicInteger counter = new AtomicInteger(0);
            private final String[] regions = dsqlClusters.keySet().toArray(new String[dsqlClusters.size()]);
            private final ConcurrentMap<String, CacheEntry> tokenCache = new ConcurrentHashMap<>();

            @Override
            public Connection getConnection() throws SQLException {
                return this.getConnection("admin", null); // with dummy parameters
            }

            @Override
            public Connection getConnection(final String username, final String password) throws SQLException {

                final int index = this.counter.getAndUpdate(v -> v == Integer.MAX_VALUE ? 0 : v + 1);
                final String region = this.regions[index % this.regions.length];

                final String endpoint = dsqlEndpoints.get(region);
                final int expirationSeconds = 30;
                final String token = this.tokenCache.compute(endpoint,
                        (k, v) -> v == null || v.isExpired(System.currentTimeMillis())
                                ? new CacheEntry(generateAuthToken(endpoint, region, expirationSeconds),
                                        expirationSeconds * 900 + System.currentTimeMillis())
                                : v)
                        .getToken();
                this.logger.debug("get a new connection from: " + region);

                final Connection connection = dsqlClusters.get(region).getConnection(username, token);
                if (!connection.isValid(connectionTImeout)) {
                    throw new SQLException("The connection obtained from the DataSource is invalid.");
                }
                return connection;
            }
        };

        final HikariDataSource ds = new HikariDataSource();
        ds.setDataSource(dsqlDS);
        ds.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());
        ds.setMaxLifetime(Duration.ofMinutes(1).toMillis());
        ds.setAutoCommit(false);

        return ds;
    }

    private String generateAuthToken(final String endpoint, final String region, final int expirationSeconds) {

        final DsqlUtilities dsqlUtilities = DsqlUtilities.builder().region(Region.of(region))
                .credentialsProvider(DefaultCredentialsProvider.create()).build();
        final String token = dsqlUtilities.generateDbConnectAdminAuthToken(
                builder -> builder.hostname(endpoint).expiresIn(Duration.ofSeconds(expirationSeconds)));

        this.logger.debug("Generate a new token: " + token);
        return token;
    }
}
