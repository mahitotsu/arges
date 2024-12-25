package com.mahitotsu.arges.api.entity;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import jakarta.persistence.EntityManager;

@Repository
public class BankRepository {
    
    @Autowired
    private EntityManager entityManager;

    public BranchEntity openBranch(final String branchCode) {

        final BranchEntity branch = new BranchEntity(this.entityManager, branchCode);
        return branch;
    }
}
