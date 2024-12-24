package com.mahitotsu.arges.api.account;

import com.mahitotsu.arges.api.common.EntityBase;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Entity(name = "Account")
@Table(uniqueConstraints = {
        @UniqueConstraint(columnNames = { "branchCode", "accountNumber" })
})
public class AccountEntity extends EntityBase {

    @NotNull
    @Size(min = 3, max = 3)
    @Pattern(regexp = "[0-9]+")
    @Column(nullable = false, updatable = false, length = 3)
    private String branchCode;

    @NotNull
    @Size(min = 7, max = 7)
    @Pattern(regexp = "[0-9]+")
    @Column(nullable = false, updatable = false, length = 3)
    private String accountNumber;
}
