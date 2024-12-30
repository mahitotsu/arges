package com.mahitotsu.arges.api.config;

import org.junit.jupiter.api.extension.AfterTestExecutionCallback;
import org.junit.jupiter.api.extension.BeforeTestExecutionCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.ExtensionContext.Namespace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TimingLoggerExtension implements BeforeTestExecutionCallback, AfterTestExecutionCallback {

    private static final String START_TIME = "start_time";

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Override
    public void beforeTestExecution(final ExtensionContext context) throws Exception {

        final long startTime = System.currentTimeMillis();
        context.getStore(Namespace.GLOBAL).put(START_TIME, startTime);
        this.logger.info("Test '{}' is started at {}.", context.getDisplayName(), startTime);
    }

    @Override
    public void afterTestExecution(final ExtensionContext context) throws Exception {

        final long endTime = System.currentTimeMillis();
        final long startTime = context.getStore(Namespace.GLOBAL).remove(START_TIME, long.class);
        this.logger.info("Test '{}' is finished at {} (Duration: {} ms).", context.getDisplayName(), endTime,
                endTime - startTime);
    }
}
