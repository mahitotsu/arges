package com.mahitotsu.arges.api.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Setter;
import lombok.ToString;

@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Entity(name = "account")
@Table(name = "accounts", uniqueConstraints = {
        @UniqueConstraint(columnNames = { "branchCode", "accountNumber" })
})
public class AccountEntity extends EntityBase {

    protected AccountEntity(final BranchEntity branch, final String accountNumber) {
        super(branch);
        this.branch = branch;
        this.accountNumber = accountNumber;
    }

    @Setter(value = AccessLevel.NONE)
    @NotNull
    @Valid
    @ManyToOne(optional = false)
    @JoinColumn(name = "branch_code", referencedColumnName = "branch_code", nullable = false, updatable = false)
    private BranchEntity branch;

    @Setter(value = AccessLevel.NONE)
    @NotNull
    @Size(min = 7, max = 7)
    @Pattern(regexp = "[0-9]+")
    @Column(name = "account_number", nullable = false, updatable = false, length = 3)
    private String accountNumber;
}
