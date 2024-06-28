export default defineEventHandler(async (event) => {
    await send(event, 'Hello');
})