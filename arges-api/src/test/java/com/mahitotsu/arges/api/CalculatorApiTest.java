package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.mahitotsu.arges.api.CalculatorApi.CloseRequest;
import com.mahitotsu.arges.api.CalculatorApi.StartRequest;

public class CalculatorApiTest extends TestBase {

    @Test
    public void testSingleThread() {

        final CalculatorApi apiClient = this.apiClient(CalculatorApi.class);
        assertNotNull(apiClient);

        final String calculationId = apiClient.open(new StartRequest(1));
        apiClient.close(new CloseRequest(calculationId));
    }
}
