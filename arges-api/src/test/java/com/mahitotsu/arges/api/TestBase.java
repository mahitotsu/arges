package com.mahitotsu.arges.api;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.Extensions;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import com.mahitotsu.arges.api.config.TimingLoggerExtension;

@SpringBootTest
@ActiveProfiles({ "test" })
@Extensions({ @ExtendWith(TimingLoggerExtension.class) })
public abstract class TestBase {

}
