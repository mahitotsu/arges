package com.mahitotsu.arges.entity;

import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import lombok.Data;

@Data
@Entity(name = "Account")
public class AccountEntity {

    @EmbeddedId
    private AccountId id;

    private long balance;
}
