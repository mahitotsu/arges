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
    return await client.send(command).then(() => name)
}

const deleteStringParameter = async (name: string) => {
    const command = new DeleteParameterCommand({ Name: name, });
    return await client.send(command).then(() => { });
}

const putKeyPairToParameterStore = async (stackName: string, suffix: string, publicKey: string, privateKey: string) => {
    return await Promise.allSettled([
        putStringParameter(`/${stackName}/PublicKey-${suffix}`, publicKey),
        putStringParameter(`/${stackName}/PrivateKey-${suffix}`, privateKey, true),
    ]).then(results => {
        results.forEach(item => {
            if (item.status === 'rejected') {
                throw new Error((item as PromiseRejectedResult).reason)
            }
        });
        return Promise.resolve({
            Data: {
                PublicKey: (results[0] as PromiseFulfilledResult<string>).value,
                PrivateKey: (results[1] as PromiseFulfilledResult<string>).value,
            }
        });
    });
}

const deleteKeyPairFromParameterStore = async (stackName: string, suffix: string) => {
    return await Promise.allSettled([
        deleteStringParameter(`/${stackName}/PublicKey-${suffix}`),
        deleteStringParameter(`/${stackName}/PrivateKey-${suffix}`),
    ]);
}

export const handler: CdkCustomResourceHandler = async (event, context) => {

    const stackName = event.StackId.split('/')[1];
    switch (event.RequestType) {

        case 'Create':
            const { publicKey, privateKey } = generateKeyPairSync('rsa', {
                modulusLength: 2048,
                publicKeyEncoding: {
                    type: 'pkcs1',
                    format: 'pem'
                },
                privateKeyEncoding: {
                    type: 'pkcs1',
                    format: 'pem',
                }
            });
            try {
                return await putKeyPairToParameterStore(stackName, event.RequestId, publicKey, privateKey);
            } catch (e) {
                await deleteKeyPairFromParameterStore(stackName, event.RequestId);
                throw e;
            }

        case 'Update':
            return {};

        case 'Delete':
            return await deleteKeyPairFromParameterStore(stackName, event.PhysicalResourceId);
    }
}
