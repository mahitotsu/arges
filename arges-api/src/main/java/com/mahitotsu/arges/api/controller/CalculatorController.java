package com.mahitotsu.arges.api.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RestController;

import com.mahitotsu.arges.api.CalculatorApi;
import com.mahitotsu.arges.api.service.CalculatorService;
import com.mahitotsu.arges.api.service.Operator;

@RestController
public class CalculatorController implements CalculatorApi{

    @Autowired
    private CalculatorService service;

    public String open(final StartRequest request) {
        return this.service.open(request.getInitialValue());
    }

    public void close(final CloseRequest request) {
        this.service.close(request.getCalculationId());
    }

    public void transact(final TransactRequest request) {
        this.service.transact(request.getCalculationId(), Operator.valueOf(request.getOperator()),
                request.getOperand());
    }

    public CurrentResponse current(final CurrentRequest request) {
        final int currentValue = this.service.current(request.getCalculationId());
        return new CurrentResponse(request.getCalculationId(), currentValue);
    }
}
