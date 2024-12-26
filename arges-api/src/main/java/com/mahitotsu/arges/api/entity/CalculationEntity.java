package com.mahitotsu.arges.api.entity;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.UUID;

import com.mahitotsu.arges.api.service.Operator;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Version;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Entity
public class CalculationEntity {

    @SuppressWarnings("unused")
    private CalculationEntity() {
    }

    public CalculationEntity(final UUID id, final int initialValue, final RoundingMode mode) {
        this.id = id;
        this.mode = mode;
        this.current = initialValue;
    }

    @Setter(AccessLevel.NONE)
    @Id
    @Column(unique = true, nullable = false, updatable = false)
    private UUID id;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    @Version
    private long modCount;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private RoundingMode mode;

    @Column(nullable = false)
    private int current;

    public void apply(final Operator operator, final int operand) {

        switch (operator) {
            case ADD:
                this.current += operand;
                return;
            case SUB:
                this.current -= operand;
                return;
            case MUL:
                this.current *= operand;
                return;
            case DIV:
                final BigDecimal value = BigDecimal.valueOf(this.current);
                this.current = value.divide(BigDecimal.valueOf(operand), this.mode).intValue();
                return;
            default:
                throw new IllegalArgumentException("The specified operator is an unkonwn value.");
        }
    }
}
