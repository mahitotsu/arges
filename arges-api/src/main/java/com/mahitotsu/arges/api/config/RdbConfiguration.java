package com.mahitotsu.arges.api.config;

import java.time.Duration;
import java.util.Properties;

import javax.sql.DataSource;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import com.zaxxer.hikari.HikariDataSource;

@Configuration
public class RdbConfiguration {

    @Bean
    @Qualifier("rawDataSource")
    public DataSource auroraDsqlDataSource(
            @Value("""
                    sslmode=require
                    loginTimeout=5
                    connectionTimeout=2
                     """) final Properties conProps,
            @Value("""
                        4iabtwnq2j55iez4j4bkykghgm.dsql.us-east-1.on.aws,
                        s4abtwnq2jebk7aj6vhlsb2coi.dsql.us-east-2.on.aws
                    """) final String[] dsqlEndpoints) {

        return new AuroraDSQLDataSource(conProps, dsqlEndpoints);
    }

    @Bean
    @Primary
    public DataSource dataSource(@Qualifier("rawDataSource") final DataSource rawDataSource) {

        final HikariDataSource ds = new HikariDataSource();
        ds.setDataSource(rawDataSource);
        ds.setExceptionOverrideClassName(DsqlExceptionOverride.class.getName());
        ds.setMaxLifetime(Duration.ofMinutes(1).toMillis());
        ds.setAutoCommit(false);

        return ds;
    }
}
