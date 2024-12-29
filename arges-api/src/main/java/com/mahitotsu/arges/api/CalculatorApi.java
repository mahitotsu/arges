package com.mahitotsu.arges.api;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.service.annotation.HttpExchange;
import org.springframework.web.service.annotation.PostExchange;

import lombok.AllArgsConstructor;
import lombok.Value;

@HttpExchange(url = "/calculator", accept = MediaType.APPLICATION_JSON_VALUE, contentType = MediaType.APPLICATION_JSON_VALUE)
public interface CalculatorApi {

    @Value
    @AllArgsConstructor
    static class StartRequest {
        private int initialValue;
    }

    @Value
    @AllArgsConstructor
    class CloseRequest {
        private String calculationId;
    }

    @Value
    @AllArgsConstructor
    class TransactRequest {
        private String calculationId;
        private String operator;
        private int operand;
    }
    @Value
    @AllArgsConstructor
    class CurrentRequest {
        private String calculationId;
    }

    @Value
    @AllArgsConstructor
    class CurrentResponse {
        private String calculationId;
        private int currentValue;
    }

    @PostExchange(url="/open")
    String open(@RequestBody final StartRequest request);

    @PostExchange(url="/close")
    void close(@RequestBody final CloseRequest request);

    @PostExchange(url="/transact")
    void transact(@RequestBody final TransactRequest request);

    @PostExchange(url="/current")
    CurrentResponse current(@RequestBody final CurrentRequest request);
}
