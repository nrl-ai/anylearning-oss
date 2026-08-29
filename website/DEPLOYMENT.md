# Deploying the website to Vercel

The site is configured for `https://anylearning-oss.nrl.ai` and uses a fixed
light theme.

## Vercel setup

1. Import `nrl-ai/anylearning-oss` into Vercel.
2. Set **Root Directory** to `website/`. Its `package.json`, lockfile, and
   `vercel.json` contain the complete Next.js build configuration.
3. Add `anylearning-oss.nrl.ai` under **Settings → Domains**.
4. Add the DNS record Vercel requests at the `nrl.ai` DNS provider.
5. Set `SITE_URL=https://anylearning-oss.nrl.ai` if overriding the checked-in
   sitemap default.

Every push to the production branch will deploy automatically after the
GitHub repository is connected.

For a direct CLI deployment, run `vercel --prod` from `website/`.
