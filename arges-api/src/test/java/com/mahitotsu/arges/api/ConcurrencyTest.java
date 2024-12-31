package com.mahitotsu.arges.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.springframework.beans.factory.annotation.Autowired;

import com.mahitotsu.arges.api.repository.ValueRepository;

@Execution(ExecutionMode.SAME_THREAD)
public class ConcurrencyTest extends TestBase {

    @Autowired
    private ValueRepository repository;

    @RepeatedTest(name = RepeatedTest.LONG_DISPLAY_NAME, value = 5)
    public void testIncrements_SingleThread() throws Exception {

        final UUID key = this.insertTask(0).call();

        final Collection<Callable<Integer>> tasks = IntStream.range(0, 10).mapToObj(i -> this.incrementTask(key))
                .collect(Collectors.toList());
        this.invokeTasks(1, tasks);

        this.assertTask(key, 10);
    }

    @RepeatedTest(name = RepeatedTest.LONG_DISPLAY_NAME, value = 5)
    public void testIncrements_MultiThread() throws Exception {

        final UUID key = this.insertTask(0).call();

        final Collection<Callable<Integer>> tasks = IntStream.range(0, 10).mapToObj(i -> this.incrementTask(key))
                .collect(Collectors.toList());
        this.invokeTasks(tasks.size(), tasks);

        this.assertTask(key, 10);
    }

    private <R> Collection<R> invokeTasks(final int numOfThreads, final Collection<Callable<R>> tasks) {

        final CompletionService<R> cs = new ExecutorCompletionService<>(Executors.newFixedThreadPool(numOfThreads));
        tasks.stream().forEach(cs::submit);

        final List<R> results = new ArrayList<>(tasks.size());
        try {
            for (int i = 0; i < tasks.size(); i++) {
                results.add(cs.take().get());
            }
        } catch (Exception e) {
            fail(e);
        }

        return results;
    }

    private Callable<UUID> insertTask(final int initialValue) {
        return () -> {
            return this.repository.insert(initialValue);
        };
    }

    private Callable<Integer> incrementTask(final UUID key) {
        return () -> {
            return this.repository.increment(key);
        };
    }

    private Callable<Boolean> assertTask(final UUID key, final int expectedValue) {
        return () -> {
            final Integer actualValue = this.repository.get(key);
            assertEquals(expectedValue, actualValue);
            return true;
        };
    }
}
