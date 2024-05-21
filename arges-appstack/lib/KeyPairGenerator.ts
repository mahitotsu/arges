import { DeleteParameterCommand, ParameterType, PutParameterCommand, SSMClient } from "@aws-sdk/client-ssm";
import { CdkCustomResourceHandler } from "aws-lambda";
import { generateKeyPairSync } from "crypto";

const client = new SSMClient();

const putStringParameter = async (name: string, value: string, secure?: boolean) => {
    const command = new PutParameterCommand({
        Name: name,
        Value: value,
        Type: secure ? ParameterType.SECURE_STRING : ParameterType.STRING,
    });
    return client.send(command).then(() => name)
}

const deleteStringParameter = async (name: string) => {
    const command = new DeleteParameterCommand({ Name: name, });
    return client.send(command).then(() => { });
}

const putKeyPairToParameterStore = async (stackName: string, suffix: string, keyPair: { publicKey: string, privateKey: string }) => {
    return Promise.allSettled([
        putStringParameter(`/${stackName}/PublicKey-${suffix}`, keyPair.publicKey),
        putStringParameter(`/${stackName}/PrivateKey-${suffix}`, keyPair.privateKey, true),
    ]).then(results => {
        const item = results.find(item => item.status === 'rejected');
        if (item) {
            throw new Error((item as PromiseRejectedResult).reason);
        }
        return Promise.resolve({
            Data: {
                PublicKey: (results[0] as PromiseFulfilledResult<string>).value,
                PrivateKey: (results[1] as PromiseFulfilledResult<string>).value,
            }
        });
    });
}

const deleteKeyPairFromParameterStore = async (stackName: string, suffix: string, e?: Error) => {
    return Promise.allSettled([
        deleteStringParameter(`/${stackName}/PublicKey-${suffix}`),
        deleteStringParameter(`/${stackName}/PrivateKey-${suffix}`),
    ]).then(() => {
        if (e) { throw e; } else { return {}; }
    });
}

export const handler: CdkCustomResourceHandler = async (event, context) => {

    const stackName = event.StackId.split('/')[1];
    switch (event.RequestType) {

        case 'Create':
            return putKeyPairToParameterStore(
                stackName, event.RequestId,
                generateKeyPairSync('rsa', {
                    modulusLength: 2048,
                    publicKeyEncoding: {
                        type: 'pkcs1',
                        format: 'pem'
                    },
                    privateKeyEncoding: {
                        type: 'pkcs1',
                        format: 'pem',
                    }
                })
            ).catch(e => {
                return deleteKeyPairFromParameterStore(stackName, event.RequestId, e);
            });

        case 'Update':
            return {};

        case 'Delete':
            return deleteKeyPairFromParameterStore(stackName, event.PhysicalResourceId);
    }
}
