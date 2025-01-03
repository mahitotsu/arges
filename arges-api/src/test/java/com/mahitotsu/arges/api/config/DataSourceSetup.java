package com.mahitotsu.arges.api.config;

import javax.sql.DataSource;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.datasource.init.DataSourceInitializer;
import org.springframework.jdbc.datasource.init.DatabasePopulator;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;

@Configuration
public class DataSourceSetup {

    @Bean
    public DataSourceInitializer dataSourceInitializer(final DataSource dataSource) {

        final ResourceDatabasePopulator databasePopulator = new ResourceDatabasePopulator();
        databasePopulator.addScript(new ClassPathResource("schema.sql"));
        databasePopulator.addScript(new ClassPathResource("data.sql"));

        DatabasePopulator proxy = (connection) -> {

            final boolean autoCommit = connection.getAutoCommit();
            try {
                connection.setAutoCommit(true);
                databasePopulator.populate(connection);
            } finally {
                connection.setAutoCommit(autoCommit);
            }
        };

        DataSourceInitializer initializer = new DataSourceInitializer();
        initializer.setDataSource(dataSource);
        initializer.setDatabasePopulator(proxy);
        initializer.setEnabled(true);
        return initializer;
    }
}
