package com.mahitotsu.arges.api.service.impl;

import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.mahitotsu.arges.api.repository.CalculationRepository;
import com.mahitotsu.arges.api.service.CalculatorService;
import com.mahitotsu.arges.api.service.Operator;

@Service("CalculatorService")
public class CalculatorServiceImpl implements CalculatorService {

    @Autowired
    private CalculationRepository repository;
    
    public String open(final int initialValue) {
        return this.repository.create(initialValue).toString();
    }

    public void transact(final String calculationId, final Operator oeprator, final int operand) {
        this.repository.apply(UUID.fromString(calculationId), oeprator, operand);
    }

    public int current(final String calculationId) {
        return this.repository.get(UUID.fromString(calculationId)).getCurrent();
    }

    public void close(final String calculationId) {
        this.repository.delete(UUID.fromString(calculationId));
    }
}
