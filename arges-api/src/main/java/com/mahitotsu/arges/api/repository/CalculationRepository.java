package com.mahitotsu.arges.api.repository;

import java.util.Random;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.mahitotsu.arges.api.entity.CalculationEntity;
import com.mahitotsu.arges.api.service.Operator;

import jakarta.persistence.EntityManager;
import jakarta.persistence.EntityNotFoundException;
import jakarta.persistence.LockModeType;

@Repository
public class CalculationRepository {

    private static final Random random = new Random();

    @Autowired
    private EntityManager entityManager;

    @Retryable(retryFor = CannotAcquireLockException.class, maxAttempts = 64, backoff = @Backoff(random = true))
    @Transactional
    public UUID create(final int initialValue) {

        final UUID id = new UUID(System.currentTimeMillis(), CalculationRepository.random.nextLong());
        final CalculationEntity calculation = new CalculationEntity(id, initialValue);
        this.entityManager.persist(calculation);

        return calculation.getId();
    }

    @Retryable(retryFor = CannotAcquireLockException.class, maxAttempts = 64, backoff = @Backoff(random = true))
    @Transactional
    public void apply(final UUID id, final Operator oeprator, final int value) {

        final CalculationEntity calculation = this.entityManager.getReference(CalculationEntity.class, id);
        this.entityManager.lock(calculation, LockModeType.PESSIMISTIC_WRITE);

        calculation.apply(oeprator, value);
    }

    @Transactional(readOnly = true)
    public CalculationEntity get(final UUID id) {

        return this.entityManager.find(CalculationEntity.class, id);
    }

    @Transactional
    public void delete(final UUID id) {

        final CalculationEntity calculation = this.entityManager.getReference(CalculationEntity.class, id);

        try {
            this.entityManager.lock(calculation, LockModeType.PESSIMISTIC_WRITE);
            this.entityManager.remove(calculation);
        } catch (EntityNotFoundException e) {
            //
        }
    }
}