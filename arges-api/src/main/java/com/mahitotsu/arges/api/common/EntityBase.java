package com.mahitotsu.arges.api.common;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.MappedSuperclass;
import jakarta.persistence.Version;
import lombok.Data;

@Data
@MappedSuperclass
public abstract class EntityBase {

    @GeneratedValue
    @Id
    @Column(nullable = false, unique = true, insertable = false, updatable = false, columnDefinition = "UUID DEFAULT gen_random_uuid()")
    private UUID id;

    @Version
    @Column(nullable = false, insertable = false, updatable = false, columnDefinition = "BIGINT DEFAULT 0")
    private long modCount;
}
