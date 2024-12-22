package com.mahitotsu.arges.entity;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

public class AccountEntityTest extends EntityTestBase {

    @Autowired
    private AccountRepository accountRepository;

    @Test
    public void testAccountOperations() {
        assertNotNull(this.accountRepository);
    }
}
