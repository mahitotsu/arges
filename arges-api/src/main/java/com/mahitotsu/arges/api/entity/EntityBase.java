package com.mahitotsu.arges.api.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.EntityManager;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.MappedSuperclass;
import jakarta.persistence.Transient;
import jakarta.persistence.Version;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Data
@ToString(exclude = { "entityManager" })
@MappedSuperclass
public abstract class EntityBase {

    protected EntityBase(final EntityManager entityManager) {
        this.setEntityManager(entityManager);
    }

    protected EntityBase(final EntityBase parent) {
        this(parent != null ? parent.getEntityManager() : null);
    }

    @Getter(value = AccessLevel.NONE)
    @Setter(value = AccessLevel.NONE)
    @GeneratedValue
    @Id
    @Column(nullable = false, unique = true, insertable = false, updatable = false, columnDefinition = "UUID DEFAULT gen_random_uuid()")
    private UUID id;

    @Getter(value = AccessLevel.NONE)
    @Setter(value = AccessLevel.NONE)
    @Version
    @Column(nullable = false, insertable = false, updatable = false, columnDefinition = "BIGINT DEFAULT 0")
    private long modCount;

    @Getter(value = AccessLevel.PROTECTED)
    @Transient
    private EntityManager entityManager;

    private void setEntityManager(final EntityManager entityManager) {

        if (this.entityManager != entityManager && this.entityManager != null) {
            this.entityManager.detach(this);
        }

        this.entityManager = entityManager;

        if (this.entityManager != null) {
            this.entityManager.merge(this);
        }
    }
}
