package com.mahitotsu.arges.api;

import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.SpringBootTest.WebEnvironment;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.support.RestClientAdapter;
import org.springframework.web.service.invoker.HttpServiceProxyFactory;

@SpringBootTest(webEnvironment = WebEnvironment.RANDOM_PORT, properties = {
        "DSQL_ENDPOINT=4iabtwnq2j55iez4j4bkykghgm.dsql.us-east-1.on.aws",
})
@ActiveProfiles({ "test" })
public abstract class TestBase {

    @LocalServerPort
    private int localServerPort;

    protected RestClient restClient() {
        return RestClient.builder()
                .baseUrl(String.format("http://localhost:%d", this.localServerPort)).build();
    }

    protected HttpServiceProxyFactory httpServiceProxyFactory() {
        return HttpServiceProxyFactory.builderFor(RestClientAdapter.create(this.restClient())).build();
    }

    protected <A> A apiClient(final Class<A> apiClientType) {
        return this.httpServiceProxyFactory().createClient(apiClientType);
    }
}
