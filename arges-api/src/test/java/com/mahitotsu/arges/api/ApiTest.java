package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ConfigurableApplicationContext;

public class ApiTest extends TestBase {
    
    @Autowired
    private ConfigurableApplicationContext application;

    @Test
    public void testAppStatus() {
        assertTrue(this.application.isActive());
        assertTrue(this.application.isRunning());
    }
}
