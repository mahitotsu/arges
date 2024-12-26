package com.mahitotsu.arges.api.repository;

import java.math.RoundingMode;
import java.util.Random;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.arges.api.entity.CalculationEntity;
import com.mahitotsu.arges.api.service.Operator;

import jakarta.persistence.EntityManager;
import jakarta.persistence.LockModeType;

@Repository
public class CalculationRepository  {

    private static final Random random = new Random();
    
    @Autowired
    private EntityManager entityManager;

    @Transactional
    public UUID create(final int initialValue, final RoundingMode roundingMode) {

        final UUID id = new UUID(System.currentTimeMillis(), CalculationRepository.random.nextLong());
        final CalculationEntity calculation = new CalculationEntity(id, initialValue, roundingMode);
        this.entityManager.persist(calculation);

        return calculation.getId();
    }

    @Transactional
    public int apply(final UUID id, final Operator oeprator, final int value) {

        final CalculationEntity calculation = this.entityManager.getReference(CalculationEntity.class, id);
        this.entityManager.lock(calculation, LockModeType.PESSIMISTIC_READ);

        calculation.apply(oeprator, value);
        return calculation.getCurrent();
    }

    @Transactional(readOnly = true)
    public int getCurrent(final UUID id) {

        return this.entityManager.getReference(CalculationEntity.class, id).getCurrent();
    }

    @Transactional
    public void delete(final UUID id) {

        this.entityManager.remove(id);
    }
}