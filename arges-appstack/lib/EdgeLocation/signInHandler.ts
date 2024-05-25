import { CloudfrontSignedCookiesOutput, getSignedCookies } from "@aws-sdk/cloudfront-signer";
import { APIGatewayProxyHandlerV2, APIGatewayProxyResultV2 } from "aws-lambda";

interface Secret {
    privateKey: string;
    clientSecret: string;
}

interface Endpoints {
    authorization_endpoint: string;
    end_session_endpoint: string;
    jwks_uri: string;
    revocation_endpoint: string;
    token_endpoint: string;
    userinfo_endpoint: string;
}

const region = process.env.AWS_DEFAULT_REGION!
const awsSessionToken = process.env.AWS_SESSION_TOKEN!
const secretName = process.env.SECRET_NAME!
const secretExtensionPort = process.env.PARAMETERS_SECRETS_EXTENSION_HTTP_PORT!
const keyPairId = process.env.KEY_PAIR_ID!
const userPoolId = process.env.USER_POOL_ID!

const cognitoEndpoints = (async () => {
    return fetch(`https://cognito-idp.${region}.amazonaws.com/${userPoolId}/.well-known/openid-configuration`, {
        method: 'GET',
    })
        .then(res => res.json())
        .then(json => json as Endpoints);
})();

const forbiddenResponse = {
    statusCode: 403,
    headers: {
        'Content-Type': 'text/plain',
    },
    body: 'Forbidden',
}

export const handler: APIGatewayProxyHandlerV2 = async (event, context): Promise<APIGatewayProxyResultV2> => {

    const endpoints = await cognitoEndpoints;
    const referer = event.headers.referer;
    if (referer == undefined || endpoints.authorization_endpoint.startsWith(referer) == false) {
        return forbiddenResponse;
    }

    const secretUrl = `http://localhost:${secretExtensionPort}/secretsmanager/get?secretId=${secretName}`;
    const resultJson = await fetch(secretUrl, {
        method: 'GET',
        headers: {
            'X-Aws-Parameters-Secrets-Token': awsSessionToken,
        }
    })
        .then(response => response.json())
        .then(json => JSON.parse(json['SecretString']) as Secret);

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
        privateKey: resultJson.privateKey,
    });

    const options = ['Path=/', 'Secure', 'HttpOnly'];
    const cookieValues = [];
    for (const item of Object.keys(signedOutput)) {
        cookieValues.push(`${item}=${[signedOutput[item as keyof CloudfrontSignedCookiesOutput], ...options].join('; ')}`);
    }

    return {
        statusCode: 200,
        headers: {
            'Content-Type': 'text/plain',
        },
        cookies: cookieValues,
        body: JSON.stringify(event, null, 4),
    }
}