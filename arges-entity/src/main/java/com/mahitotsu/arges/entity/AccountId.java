package com.mahitotsu.arges.entity;

import java.io.Serializable;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import lombok.Data;

@Data
@Embeddable
public class AccountId implements Serializable {

    @Column(nullable = false, updatable = false)
    private String branchCode;

    @Column(nullable = false, updatable = false)
    private String accountNumber;
}
