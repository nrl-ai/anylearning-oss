import dynamic from "next/dynamic";
import Head from "next/head";
import Link from "next/link";
import React from "react";

import LabelDemo from "../components/demo/label-demo";
import TrainingDemo from "../components/demo/training-demo";
import SiteFooter, { ThemeSync } from "../components/features/site-chrome";
import Menu from "../components/menu";
import config from "../theme.config";

// Swiper is ~40 kB and the slideshow sits several screens below the fold, so it
// has no business in the first load. The demos above it are the hero and stay
// in the main bundle.
const Slideshow = dynamic(() => import("../components/screenshot/slideshow").then((m) => m.Slideshow), {
  ssr: false,
  loading: () => <div className="h-[420px] rounded-lg border border-line bg-surface-sunken" />,
});

/**
 * The hero is the product, not a picture of the product.
 *
 * Every competitor in this category opens with a gradient headline over a
 * laptop mockup. AnyLearning's most characteristic act is drawing a box on an
 * image, so the page opens with a labeller you can actually use before you have
 * downloaded anything. The screenshots come later, once you already believe it.
 */

const STAGES = [
  {
    n: "01",
    title: "Label",
    body: "Draw boxes, polygons or whole-image classes. Segment Anything can pre-label for you, so you correct instead of starting from scratch.",
  },
  {
    n: "02",
    title: "Train",
    body: "Pick a model size, set epochs, start the run. Training happens in a separate process on your own hardware: GPU if you have one, CPU if you don't.",
  },
  {
    n: "03",
    title: "Ship",
    body: "Every finished run registers a model you can try on a new image and export to ONNX for whatever runs it next.",
  },
];

const TASKS = [
  { name: "Object detection", note: "Boxes around things that matter" },
  { name: "Image segmentation", note: "Outline a region precisely" },
  { name: "Instance segmentation", note: "Separate every occurrence" },
  { name: "Keypoint detection", note: "Named landmarks on every instance" },
  { name: "Image classification", note: "One label per image" },
  { name: "Handpose classification", note: "Gestures from hand landmarks" },
];

function Section({ eyebrow, title, lead, children, className = "" }) {
  return (
    <section className={`mx-auto w-full max-w-6xl px-6 py-20 ${className}`}>
      {eyebrow && <p className="t-eyebrow mb-3">{eyebrow}</p>}
      {title && <h2 className="t-display text-3xl font-semibold sm:text-4xl">{title}</h2>}
      {lead && <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted-foreground">{lead}</p>}
      {children}
    </section>
  );
}

export default function HomePage() {
  const head = config.head();

  return (
    <>
      <Head>
        <title>AnyLearning: label images and train models on your own machine</title>
        {head}
      </Head>
      <ThemeSync />
      <div className="flex min-h-screen flex-col bg-background text-foreground">
        <Menu activePage="home" />

        {/* --- Hero ------------------------------------------------------- */}
        <header className="mx-auto w-full max-w-6xl px-6 pt-20 pb-8">
          <p className="t-eyebrow mb-4">Offline computer vision</p>
          <h1 className="t-display max-w-3xl text-4xl font-semibold sm:text-5xl lg:text-6xl">
            Label images and train models on your own machine.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground">
            AnyLearning is a desktop app for building computer vision models. Your images, your
            annotations, your trained weights — none of it leaves the machine it was created on.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              href="/download"
              className="rounded-md bg-mark px-5 py-2.5 text-sm font-medium text-mark-ink transition-colors hover:bg-mark-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              Download AnyLearning
            </Link>
            <Link
              href="/docs"
              className="rounded-md border border-line px-5 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
            >
              Read the docs
            </Link>
          </div>
        </header>

        {/* The signature moment: a real labeller, before any download. */}
        <div className="mx-auto w-full max-w-6xl px-6">
          <LabelDemo />
          <p className="mt-3 text-xs text-muted-foreground">
            This is running in your browser, with nothing sent anywhere. The desktop app works the
            same way. It just also trains the model.
          </p>
        </div>

        {/* --- How it works ----------------------------------------------- */}
        <Section
          eyebrow="How it works"
          title="Three stages, in order"
          lead="You cannot train without labelled data, and you cannot ship a model without a finished run. The app tracks where each project actually stands rather than where you happen to be looking."
        >
          <ol className="mt-10 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-3">
            {STAGES.map((stage) => (
              <li key={stage.n} className="bg-surface p-6">
                <span className="font-mono text-xs tabular text-mark">{stage.n}</span>
                <h3 className="t-display mt-2 text-lg font-semibold">{stage.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{stage.body}</p>
              </li>
            ))}
          </ol>
        </Section>

        {/* --- Training demo ---------------------------------------------- */}
        <Section
          eyebrow="Training"
          title="Watch a run, without starting one"
          lead="A real run writes progress to the project database from a separate process, and the app polls it. This is a simulation of exactly that, so you can see the shape of it before you install anything."
        >
          <div className="mt-10">
            <TrainingDemo />
          </div>
        </Section>

        {/* --- Offline ----------------------------------------------------- */}
        <Section
          eyebrow="Privacy"
          title="Offline is the feature"
          lead="Plenty of tools will train a model for you if you upload your data first. That is exactly the step some work cannot take: patient scans, factory-floor footage, anything under NDA, anything a data-protection officer has an opinion about."
        >
          <div className="mt-10 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-3">
            {[
              ["No upload step", "Images are read from disk and stay there. There is no bucket, no ingest, no sync."],
              ["No account", "Nothing to sign into and no activation key. Install it and start working."],
              ["No running cost", "Training uses hardware you already own, so a longer run costs time rather than money."],
            ].map(([title, body]) => (
              <div key={title} className="bg-surface p-6">
                <h3 className="t-display text-base font-semibold">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{body}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* --- Tasks ------------------------------------------------------- */}
        <Section
          eyebrow="Tasks"
          title="Six kinds of project"
          lead="Each one comes with its own labelling tools, model variants and training defaults, so the choice you make when you create a project is the only configuration most people need."
        >
          <ul className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {TASKS.map((task, i) => (
              <li key={task.name} className="rounded-lg border border-line bg-surface p-4">
                <span
                  aria-hidden
                  className="mb-3 block h-1 w-8 rounded-full"
                  style={{ backgroundColor: `var(--class-${i + 1})` }}
                />
                <h3 className="text-sm font-medium">{task.name}</h3>
                <p className="mt-1 text-xs text-muted-foreground">{task.note}</p>
              </li>
            ))}
          </ul>
        </Section>

        {/* --- Screenshots -------------------------------------------------- */}
        <Section eyebrow="The app" title="See it in action">
          <div className="mt-10">
            <Slideshow />
          </div>
        </Section>

        {/* --- Open source ------------------------------------------------- */}
        <Section
          eyebrow="Open source"
          title="Apache-2.0, with no activation"
          lead="Use the complete application, including local training, without an account or product key. Inspect the code, adapt it, and contribute improvements upstream."
        >
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="https://github.com/nrl-ai/anylearning-oss"
              className="rounded-md bg-mark px-5 py-2.5 text-sm font-medium text-mark-ink transition-colors hover:bg-mark-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            >
              View on GitHub
            </Link>
            <Link
              href="/download"
              className="rounded-md border border-line px-5 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
            >
              Download AnyLearning
            </Link>
          </div>
        </Section>

        <SiteFooter />
      </div>
    </>
  );
}
