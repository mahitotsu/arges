export default defineNuxtConfig({
  devtools: { enabled: false },
  ssr: true,
  nitro: {
    preset: 'aws-lambda',
    minify: true,
  },
  runtimeConfig: {
    signInUrl: '',
    callbackUrl: '',
  }
})
