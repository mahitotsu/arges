package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import com.mahitotsu.arges.api.CalculatorApi.CloseRequest;
import com.mahitotsu.arges.api.CalculatorApi.CurrentRequest;
import com.mahitotsu.arges.api.CalculatorApi.StartRequest;
import com.mahitotsu.arges.api.CalculatorApi.TransactRequest;

import lombok.Value;

public class CalculatorApiTest extends TestBase {

    @Value
    public static class Operation {
        private String operator;
        private int operand;
    }

    @Value
    public static class CalculationTask implements Callable<Integer> {

        private CalculatorApi apiClient;
        private String calculationId;
        private Iterable<Operation> operations;

        @Override
        public Integer call() throws Exception {
            this.operations.forEach(op -> this.apiClient
                    .transact(new TransactRequest(this.calculationId, op.getOperator(), op.getOperand())));
            return this.apiClient.current(new CurrentRequest(this.calculationId)).getCurrentValue();
        }
    }

    @Test
    public void testSingleThread() {

        final List<Operation> operations = new LinkedList<>();
        IntStream.range(0, 10).forEach(i -> operations.add(new Operation("ADD", i + 1)));
        IntStream.range(0, 5).forEach(i -> operations.add(new Operation("SUB", i + 1)));
        final int expectedValue = 40;

        final CalculatorApi apiClient = this.apiClient(CalculatorApi.class);
        final String calculationId = apiClient.open(new StartRequest(0));
        final CalculationTask task = new CalculationTask(apiClient, calculationId, operations);

        try {
            assertEquals(expectedValue, task.call());
        } catch (Exception e) {
            fail(e);
        } finally {
            apiClient.close(new CloseRequest(calculationId));
        }

    }
}
