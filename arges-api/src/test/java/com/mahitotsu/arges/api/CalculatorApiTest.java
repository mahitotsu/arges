package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.RepeatedTest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @RepeatedTest(value = 1, name = "{displayName} - repetition {currentRepetition} of {totalRepetitions}")
    public void testCase1_SingleThread() {
        this.invokeTestCase1(1);
    }

    @RepeatedTest(value = 1, name = "{displayName} - repetition {currentRepetition} of {totalRepetitions}")
    public void testCase1_MultiThread() {
        this.invokeTestCase1(10);
    }

    @RepeatedTest(value = 1, name = "{displayName} - repetition {currentRepetition} of {totalRepetitions}")
    public void testCase2_SingleThread() {
        this.invokeTestCase2(1);
    }

    @RepeatedTest(value = 1, name = "{displayName} - repetition {currentRepetition} of {totalRepetitions}")
    public void testCase2_MultiThread() {
        this.invokeTestCase2(10);
    }

    private void invokeTestCase1(final int numOfThreads) {

        final List<Operation> operations = new LinkedList<>();
        IntStream.range(0, 10).forEach(i -> operations.add(new Operation("ADD", i + 1)));
        IntStream.range(0, 5).forEach(i -> operations.add(new Operation("SUB", i + 1)));
        final int expectedValue = 40;

        final CalculatorApi apiClient = this.apiClient(CalculatorApi.class);
        final String calculationId = apiClient.open(new StartRequest(0));
        final CalculationTask task = new CalculationTask(apiClient, calculationId, operations);

        final int numOfTasks = 10;
        final CompletionService<Integer> completion = new ExecutorCompletionService<>(
                Executors.newFixedThreadPool(numOfThreads));
        IntStream.range(0, numOfTasks).forEach(i -> completion.submit(task));

        try {
            int actual = 0;
            for (int i = 0; i < numOfTasks; i++) {
                actual = completion.take().get();
                this.logger.info("Complete calculation task: {}", i);
            }
            assertEquals(expectedValue * numOfTasks, actual);
        } catch (Exception e) {
            fail(e);
        }
    }

    private void invokeTestCase2(final int numOfThreads) {

        final List<Operation> operations = new LinkedList<>();
        IntStream.range(0, 10).forEach(i -> operations.add(new Operation("ADD", i + 1)));
        IntStream.range(0, 5).forEach(i -> operations.add(new Operation("SUB", i + 1)));
        final int expectedValue = 40;

        final CalculatorApi apiClient = this.apiClient(CalculatorApi.class);

        final int numOfTasks = 10;
        final CompletionService<Integer> completion = new ExecutorCompletionService<>(
                Executors.newFixedThreadPool(numOfThreads));
        IntStream.range(0, numOfTasks).forEach(i -> {
            final String calculationId = apiClient.open(new StartRequest(0));
            final CalculationTask task = new CalculationTask(apiClient, calculationId, operations);
            completion.submit(task);
        });

        try {
            for (int i = 0; i < numOfTasks; i++) {
                final int actual = completion.take().get();
                this.logger.info("Complete calculation task: {}", i);
                assertEquals(expectedValue * numOfTasks, actual);
            }
        } catch (Exception e) {
            fail(e);
        }
    }
}
