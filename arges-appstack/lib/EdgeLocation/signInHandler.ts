import { CloudfrontSignedCookiesOutput, getSignedCookies } from "@aws-sdk/cloudfront-signer";
import { APIGatewayProxyHandlerV2 } from "aws-lambda";

interface Secret {
    privateKey: string;
}

const secretName = process.env.SECRET_NAME!
const awsSessionToken = process.env.AWS_SESSION_TOKEN!
const secretExtensionPort = process.env.PARAMETERS_SECRETS_EXTENSION_HTTP_PORT!
const keyPairId = process.env.KEY_PAIR_ID!

export const handler: APIGatewayProxyHandlerV2 = async (event, context) => {

    const secretUrl = `http://localhost:${secretExtensionPort}/secretsmanager/get?secretId=${secretName}`;
    const resultJson = await fetch(secretUrl, {
        method: 'GET',
        headers: {
            'X-Aws-Parameters-Secrets-Token': awsSessionToken,
        }
    })
        .then(response => response.json())
        .then(json => JSON.parse(json['SecretString']) as Secret);

    const signedResult = getSignedCookies({
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
    for (const item of Object.keys(signedResult)) {
        cookieValues.push(`${item}=${[signedResult[item as keyof CloudfrontSignedCookiesOutput], ...options].join('; ')}`);
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