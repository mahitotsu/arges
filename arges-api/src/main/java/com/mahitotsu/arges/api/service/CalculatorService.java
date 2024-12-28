package com.mahitotsu.arges.api.service;

public interface CalculatorService {

    String open(int initialValue );

    void transact(String calculationId, Operator operator, int operand);

    int current(String calculationId);

    void close(String calculationId);
}
