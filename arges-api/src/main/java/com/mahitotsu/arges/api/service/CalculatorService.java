package com.mahitotsu.arges.api.service;

import java.math.RoundingMode;

public interface CalculatorService {

    String start(int initialValue, RoundingMode roundingMode);

    void push(String calculationId, Operator operator, int operand);

    int current(String calculationId);

    void clear(String calculationId);
}
