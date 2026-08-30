import type { DocsThemeConfig } from "nextra-theme-docs";
import { useConfig } from "nextra-theme-docs";
import { useRouter } from "next/router";
import { Wordmark } from "./components/brand/logo";
import GitHubMark from "./components/icons/github-mark";

// The mark is an inline SVG that takes currentColor for the frame and the
// --mark token for the handle, so one asset is correct in both themes and the
// navbar chrome stays neutral.
const logo = <Wordmark />;

const config: DocsThemeConfig = {
  // Nextra reads next-themes options from HERE, not from tailwind.config.js.
  // The repo had a `nextThemes` block in the Tailwind config, where it is not a
  // recognised option and has never had any effect.
  nextThemes: {
    defaultTheme: "light",
    forcedTheme: "light",
  },
  // The mark, so Nextra's own chrome (active nav, links, focus rings) matches
  // the rest of the system rather than shipping its default blue.
  primaryHue: 196,
  primarySaturation: 90,
  project: {
    link: "https://github.com/nrl-ai/anylearning-oss",
    icon: <GitHubMark className="h-5 w-5" />,
  },
  docsRepositoryBase: "https://github.com/nrl-ai/anylearning-oss/tree/main/website",
  useNextSeoProps() {
    const { asPath } = useRouter();
    if (asPath !== "/") {
      return {
        titleTemplate: "%s – AnyLearning",
      };
    }
  },
  logo,
  head: function useHead() {
    const { title } = useConfig();
    const { route } = useRouter();
    const autoLabeling = route === "/docs/auto-labeling";
    const descriptions: Record<string, string> = {
      "/docs/auto-labeling":
        "Learn how to auto-label images in AnyLearning with SAM 2 using box, include-point and exclude-point prompts.",
      "/docs/samexporter":
        "Convert and run SAM, MobileSAM, EfficientSAM, SAM 2, SAM 2.1 and SAM3 as ONNX with real model downloads, prompts and visual examples.",
      "/docs/tabular-ai":
        "Train local CatBoost classification and regression models from CSV, Excel, Parquet or Hugging Face datasets in AnyLearning.",
      "/docs/text-llm":
        "Use local NLP to train text classifiers, run bounded lexical search and evaluate saved model, LLM or human responses without sending data away.",
      "/docs/machine-learning-basics":
        "Learn what machine learning is, how supervised, unsupervised and reinforcement learning differ, and where LLMs fit.",
      "/docs/choosing-an-ml-task":
        "Choose the right machine-learning task for tables, text and images with a practical decision guide.",
      "/docs/evaluating-ml-models":
        "Learn train, validation and test splits, model metrics, baselines and how to prevent data leakage.",
    };
    const description =
      descriptions[route] ??
      "Simple, Smart, Secured AI Model Builder. Create custom AI models in minutes with no-code tools.";
    const canonical = `https://anylearning-oss.nrl.ai${route === "/" ? "" : route}`;
    const autoLabelingThumbnail =
      "https://anylearning-oss.nrl.ai/auto_labeling/auto-labeling-thumbnail.jpg";
    const socialCard = autoLabeling
      ? autoLabelingThumbnail
      : route === "/" || !title
        ? "https://anylearning-oss.nrl.ai/screenshot.png"
        : `https://anylearning-oss.nrl.ai/api/og?title=${title}`;

    return (
      <>
        {/* Matches --background in styles/tokens.css; the browser chrome has to
            be given a literal colour, so these two are the one place the token
            values are repeated. */}
        <meta
          name="theme-color"
          content="oklch(0.966 0.002 255)"
          media="(prefers-color-scheme: light)"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta httpEquiv="Content-Language" content="en" />
        <meta name="description" content={description} />
        <link rel="canonical" href={canonical} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content={canonical} />
        <meta property="og:description" content={description} />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={title ? title + " – AnyLearning" : "AnyLearning"} />
        <meta name="twitter:description" content={description} />
        <meta name="twitter:image" content={socialCard} />
        <meta
          name="twitter:image:alt"
          content={
            autoLabeling
              ? "Auto-labeling a safety helmet with SAM 2 in AnyLearning"
              : "AnyLearning no-code AI model builder"
          }
        />
        <meta name="twitter:site:domain" content="anylearning-oss.nrl.ai" />
        <meta name="twitter:url" content={canonical} />
        <meta property="og:image" content={socialCard} />
        <meta property="og:image:width" content={autoLabeling ? "1200" : "1600"} />
        <meta property="og:image:height" content={autoLabeling ? "630" : "900"} />
        <meta
          property="og:image:alt"
          content={
            autoLabeling
              ? "Auto-labeling a safety helmet with SAM 2 in AnyLearning"
              : "AnyLearning no-code AI model builder"
          }
        />
        {autoLabeling && (
          <script
            type="application/ld+json"
            dangerouslySetInnerHTML={{
              __html: JSON.stringify({
                "@context": "https://schema.org",
                "@type": "HowTo",
                name: "Auto-label images with SAM 2",
                description,
                image: autoLabelingThumbnail,
                step: [
                  {
                    "@type": "HowToStep",
                    name: "Turn on AI and choose a model",
                    text: "Start labelling, turn on the AI tool and choose the recommended SAM 2 Hiera-Small model.",
                    url: `${canonical}#step-1-turn-on-ai-and-choose-a-model`,
                  },
                  {
                    "@type": "HowToStep",
                    name: "Prompt and refine the object",
                    text: "Draw an include box, then refine the generated mask with include and exclude points.",
                    url: `${canonical}#step-2-prompt-and-refine-the-object`,
                  },
                  {
                    "@type": "HowToStep",
                    name: "Assign a class and save",
                    text: "Finish the object, assign its class, inspect the mask and save the annotation.",
                    url: `${canonical}#step-3-assign-a-class-and-save`,
                  },
                ],
              }),
            }}
          />
        )}
        <meta name="apple-mobile-web-app-title" content="AnyLearning" />
        <link rel="icon" href="/favicon.png" type="image/png" />
        <link
          rel="icon"
          href="/favicon.png"
          type="image/png"
          media="(prefers-color-scheme: light)"
        />
      </>
    );
  },
  // The key is what Nextra remembers when someone dismisses this. Change it
  // and the banner comes back for everyone, so it doubles as the announcement's
  // identity -- never reuse one for a different message.
  banner: {
    key: "anylearning-0263-structured-ai",
    text: (
      <a href="/docs/tabular-ai">
        New in AnyLearning 0.26.5: large-table Tabular AI, Text AI and safe Hugging Face imports.
        Explore the guide →
      </a>
    ),
  },
  editLink: {
    text: null,
    component: () => <></>,
  },
  feedback: {
    content: null,
    labels: "feedback",
  },
  sidebar: {
    titleComponent({ title, type }) {
      if (type === "separator") {
        return <span className="cursor-default">{title}</span>;
      }
      return <>{title}</>;
    },
    defaultMenuCollapseLevel: 2,
    toggleButton: true,
  },
  footer: {
    text: (
      <div className="flex w-full flex-col items-center gap-1 sm:items-start">
        <a
          className="rounded text-current underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
          target="_blank"
          rel="noopener noreferrer"
          title="AnyLearning Team Website"
          href="https://www.nrl.ai"
        >
          Created by <span className="font-medium">NRL.ai</span>.
        </a>
        <p className="text-xs">© {new Date().getFullYear()} The AnyLearning Project.</p>
      </div>
    ),
  },
};

export default config;
