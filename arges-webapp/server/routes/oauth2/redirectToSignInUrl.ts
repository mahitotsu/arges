const runtimeConfig = useRuntimeConfig();
const signInUrl = runtimeConfig.signInUrl;

const errorPage = `<!doctype html>
<html>
<head>
    <meta http-equiv="refresh" content="3;url=${signInUrl}" />
</head>
<body>
</body>
</html>
`

export default defineEventHandler(async (event) => {
    await send(event, errorPage, 'text/html'); 
})
