export default defineNuxtConfig({
  devtools: { enabled: true },
  ssr: true,
  nitro: {
    preset: 'aws-lambda',
    serveStatic: false,
  }
})
