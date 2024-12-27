package com.mahitotsu.arges.api.controller;

import java.math.RoundingMode;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.mahitotsu.arges.api.service.CalculatorService;
import com.mahitotsu.arges.api.service.Operator;

import lombok.AllArgsConstructor;
import lombok.Value;

@RestController
@RequestMapping(path = "/calculator", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
public class CalculatorController {

    @Value
    @AllArgsConstructor
    public static class StartRequest {
        private int initialValue;
    }

    @Value
    @AllArgsConstructor
    public static class ClearRequest {
        private String calculationId;
    }

    @Value
    @AllArgsConstructor
    public static class PushRequest {
        private String calculationId;
        private String operator;
        private int operand;
    }

    @Value
    @AllArgsConstructor
    public static class GetRequest{
        private String calculationId;
    }

    @Value
    @AllArgsConstructor
    public static class GetResponse {
        private String calculationId;
        private int currentValue; 
    }

    @Autowired
    private CalculatorService service;

    @PostMapping("/start")
    public String start(@RequestBody final StartRequest request) {
        return this.service.start(request.getInitialValue(), RoundingMode.HALF_DOWN);
    }

    @PostMapping("/clear")
    public void clear(@RequestBody final ClearRequest request) {
        this.service.clear(request.getCalculationId());
    }

    @PostMapping("/push")
    public void push(@RequestBody final PushRequest request) {
        this.service.push(request.getCalculationId(), Operator.valueOf(request.getOperator()),
                request.getOperand());
    }

    @GetMapping( "/get-current/{caluculationId}")
    public GetResponse getCurrent(@RequestBody final GetRequest request) {
        final int currentValue = this.service.current(request.getCalculationId());
        return new GetResponse(request.getCalculationId(), currentValue);
    }
}
