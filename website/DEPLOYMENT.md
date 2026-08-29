# Deploying the website to Vercel

The site is configured for `https://anylearning-oss.nrl.ai` and uses a fixed
light theme.

## Vercel setup

1. Import `nrl-ai/anylearning-oss` into Vercel.
2. Leave the repository root selected; the root `vercel.json` builds
   `website/`. Alternatively, set **Root Directory** to `website/`, where the
   local `vercel.json` supplies the Next.js framework setting.
3. Add `anylearning-oss.nrl.ai` under **Settings → Domains**.
4. Add the DNS record Vercel requests at the `nrl.ai` DNS provider.
5. Set `SITE_URL=https://anylearning-oss.nrl.ai` if overriding the checked-in
   sitemap default.

Every push to the production branch will deploy automatically after the
GitHub repository is connected.
