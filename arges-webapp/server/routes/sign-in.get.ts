import { CloudfrontSignedCookiesOutput, getSignedCookies } from "@aws-sdk/cloudfront-signer";
import * as jwt from "jsonwebtoken";
import { JwksClient } from "jwks-rsa";

interface Secret {
    privateKey: string;
    clientId: string;
    clientSecret: string;
    redirectUrl: string;
}

interface Endpoints {
    authorization_endpoint: string;
    end_session_endpoint: string;
    jwks_uri: string;
    revocation_endpoint: string;
    token_endpoint: string;
    userinfo_endpoint: string;
}

interface Tokens {
    id_token: string;
    access_token: string;
    refresh_token: string;
}

const region = process.env.AWS_DEFAULT_REGION!
const awsSessionToken = process.env.AWS_SESSION_TOKEN!
const secretName = process.env.SECRET_NAME!
const secretExtensionPort = process.env.PARAMETERS_SECRETS_EXTENSION_HTTP_PORT!
const keyPairId = process.env.KEY_PAIR_ID!
const userPoolId = process.env.USER_POOL_ID!
const issuer = `https://cognito-idp.${region}.amazonaws.com/${userPoolId}`;

const cognitoEndpoints = (async () => {
    return fetch(`${issuer}/.well-known/openid-configuration`)
        .then(res => res.json())
        .then(json => json as Endpoints);
})();

const jwksClient = (async () => { return new JwksClient({ jwksUri: (await cognitoEndpoints).jwks_uri }); })();
const decodeToken = async (token: string, options?: jwt.VerifyOptions) => {
    return new Promise((resolve, reject) => {
        jwt.verify(
            token,
            (header, callback) => {
                jwksClient.then(client => {
                    client.getSigningKey(header.kid, (err, key) => {
                        callback(err, key?.getPublicKey());
                    });
                });
            },
            options,
            (err, decoded) => err ? reject(err) : resolve(decoded)
        );
    });
}

export default defineEventHandler(async (event) => {

    const endpoints = await cognitoEndpoints;
    const authorizationCode = getQuery(event).code as string;
    const referer = event.node.req.headers.referer;
    if (authorizationCode == undefined
        || referer == undefined || endpoints.authorization_endpoint.startsWith(referer) == false) {
        return sendError(event, createError({ status: 403, }));
    }

    const secretUrl = `http://localhost:${secretExtensionPort}/secretsmanager/get?secretId=${secretName}`;
    const secretValue = await fetch(secretUrl, {
        method: 'GET',
        headers: {
            'X-Aws-Parameters-Secrets-Token': awsSessionToken,
        }
    })
        .then(response => response.json())
        .then(json => JSON.parse(json['SecretString']) as Secret);

    const tokens = await fetch(endpoints.token_endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            'grant_type': 'authorization_code',
            'client_id': secretValue.clientId,
            'client_secret': secretValue.clientSecret,
            'code': authorizationCode,
            'redirect_uri': secretValue.redirectUrl,
        })
    })
        .then(res => res.json())
        .then(json => json as Tokens);
    const idToken = await decodeToken(tokens.id_token, { issuer: issuer, audience: secretValue.clientId, });
    const accessToken = await decodeToken(tokens.access_token, { issuer: issuer, });

    const userInfo = await fetch(endpoints.userinfo_endpoint, {
        method: 'GET',
        headers: {
            'Authorization': `Bearer ${tokens.access_token}`,
        },
    })
        .then(res => res.json())
        .then(json => json);

    const signedOutput = getSignedCookies({
        policy: JSON.stringify({
            Statement: [{
                Condition: {
                    DateLessThan: {
                        'AWS:EpochTime': 2147483647
                    },
                },
            },],
        }),
        keyPairId: keyPairId,
        privateKey: secretValue.privateKey,
    });

    for (const item of Object.keys(signedOutput)) {
        setCookie(event, item, signedOutput[item as keyof CloudfrontSignedCookiesOutput] as string, {
            path: '/',
            secure: true,
            httpOnly: true,
        });
    }

    return send(event, JSON.stringify({
        event,
        idToken,
        accessToken,
        userInfo,
    }), 'text/plain');
});