import { getSignedCookies } from "@aws-sdk/cloudfront-signer";
import { APIGatewayProxyHandlerV2 } from "aws-lambda";

interface Secret {
    privateKey: string;
}

const secretName = process.env.SECRET_NAME!
const awsSessionToken = process.env.AWS_SESSION_TOKEN!
const secretExtensionPort = process.env.PARAMETERS_SECRETS_EXTENSION_HTTP_PORT!
const keyPairId = process.env.KEY_PAIR_ID!

export const handler: APIGatewayProxyHandlerV2 = async (event, context) => {

    const secretUrl = `https://localhost:${secretExtensionPort}/secretsmanager/get?secretId=${secretName}`;
    const resultJson = await fetch(secretUrl, {
        method: 'GET',
        headers: {
            'X-Aws-Parameters-Secrets-Token': awsSessionToken,
        }
    })
        .then(response => response.json())
        .then(json => JSON.parse(json['SecretString']) as Secret);

    const cookieHeaders = getSignedCookies({
        policy: JSON.stringify({
            Statement: [{
                // Resource: 'http*://*/*',
                Condition: {
                    DateLessThan: {
                        'AWS:EpochTime': Math.floor(new Date().getTime() / 1000),
                    },
                },
            },],
        }),
        keyPairId: keyPairId,
        privateKey: resultJson.privateKey,
    }) as any;
    const options = ['Path=/', 'Secure', 'HttpOnly'].join('; ');
    for (const key in Object.keys(cookieHeaders)) {
        cookieHeaders[key] = cookieHeaders[key] + options;
    }

    return {
        statusCode: 200,
        headers: {
            'Content-Type': 'text/plain',
            ...cookieHeaders,
        },
        body: JSON.stringify(event, null, 4),
    }
}