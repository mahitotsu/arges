package com.mahitotsu.arges.api.config;

import java.sql.Connection;
import java.sql.SQLException;
import java.time.Duration;
import java.util.Arrays;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import javax.sql.DataSource;

import org.springframework.jdbc.datasource.AbstractDataSource;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;

import lombok.Value;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dsql.DsqlUtilities;

public class AuroraDSQLDataSource extends AbstractDataSource {

    @Value
    static class CacheEntry {

        CacheEntry(final String token, final long expiresAt) {
            this.token = token;
            this.expiresAt = expiresAt;
        }

        private String token;
        private long expiresAt;

        boolean isExpired(final long datetime) {
            return this.expiresAt < datetime;
        }
    }

    public AuroraDSQLDataSource(final Properties connectionProperties, final String... dsqlEndpoints) {

        final Map<String, String> regionalEdnpoints = Arrays.stream(dsqlEndpoints).collect(
                Collectors.toMap(
                        endpoint -> endpoint.split("\\.")[2],
                        endpoint -> endpoint));
        final Map<String, DataSource> regionalDataSources = regionalEdnpoints.entrySet().stream()
                .collect(Collectors.toMap(
                        entry -> entry.getKey(),
                        entry -> new DriverManagerDataSource("jdbc:postgresql://" + entry.getValue() + "/postgres",
                                connectionProperties)));

        this.regionalEndpoints = regionalEdnpoints;
        this.regionalDataSources = regionalDataSources;
        this.regions = regionalEdnpoints.keySet().toArray(new String[regionalEdnpoints.size()]);
        this.connectionTimeout = Integer.parseInt(connectionProperties.getProperty("connectionTimeout", "10000"));
    }

    private final AtomicInteger counter = new AtomicInteger(0);
    private final ConcurrentMap<String, CacheEntry> tokenCache = new ConcurrentHashMap<>();

    private final Map<String, String> regionalEndpoints;
    private final Map<String, DataSource> regionalDataSources;
    private final String[] regions;
    private final int connectionTimeout;

    @Retryable(retryFor = SQLException.class, maxAttempts = 4, backoff = @Backoff(delay = 0))
    @Override
    public Connection getConnection() throws SQLException {
        return this.getConnection("admin", null); // with dummy password
    }

    @Retryable(retryFor = SQLException.class, maxAttempts = 4, backoff = @Backoff(delay = 0))
    @Override
    public Connection getConnection(final String username, final String password) throws SQLException {

        final int index = this.counter.getAndUpdate(v -> v == Integer.MAX_VALUE ? 0 : v + 1);
        final String region = this.regions[index % this.regions.length];

        final String endpoint = regionalEndpoints.get(region);
        final int expirationSeconds = 30;
        final String token = this.tokenCache.compute(endpoint,
                (k, v) -> v == null || v.isExpired(System.currentTimeMillis())
                        ? new CacheEntry(this.generateAuthToken(endpoint, region, expirationSeconds),
                                expirationSeconds * 900 + System.currentTimeMillis())
                        : v)
                .getToken();

        final Connection connection = regionalDataSources.get(region).getConnection(username, token);
        if (!connection.isValid(this.connectionTimeout)) {
            throw new SQLException("The connection obtained from the DataSource is invalid.");
        }

        this.logger.debug("get a new connection from: " + region);
        return connection;
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
