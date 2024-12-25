package com.mahitotsu.arges.api.entity;

import java.util.Random;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EntityManager;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AccessLevel;
import lombok.EqualsAndHashCode;
import lombok.Setter;
import lombok.ToString;

@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Entity(name = "branch")
@Table(name = "branches")
public class BranchEntity extends EntityBase {

    private static final Random RANDOM = new Random();

    protected BranchEntity(final EntityManager entityManager, final String branchCode) {
        super(entityManager);
        this.branchCode = branchCode;
    }

    @Setter(value = AccessLevel.NONE)
    @NotNull
    @Size(min = 3, max = 3)
    @Pattern(regexp = "[0-9]+")
    @Column(name = "branch_code", unique = true, nullable = false, updatable = false, length = 3)
    private String branchCode;

    public AccountEntity openAccount() {

        final String accountNumber = String.format("%07d", RANDOM.nextInt(10000000));
        final AccountEntity account = new AccountEntity(this, accountNumber);

        return account;
    }
}
