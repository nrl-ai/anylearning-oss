import Image from "next/image";
import { useEffect, useState } from "react";

const SLIDES = [
  {
    src: "/screenshots/1.png",
    title: "Set up the project",
    detail: "Define classes and keep the dataset split visible from the start.",
    alt: "AnyLearning project overview showing class labels and dataset split",
  },
  {
    src: "/screenshots/2.png",
    title: "Inspect the dataset",
    detail: "Browse every image and see annotation coverage at a glance.",
    alt: "AnyLearning dataset browser showing annotated construction images",
  },
  {
    src: "/screenshots/3.png",
    title: "Label precisely",
    detail: "Draw, refine and review annotations without leaving the app.",
    alt: "AnyLearning labeling workspace with boxes around construction equipment",
  },
  {
    src: "/screenshots/4.png",
    title: "Follow every run",
    detail: "Track loss, accuracy and run status while training stays local.",
    alt: "AnyLearning training dashboard with loss and validation accuracy charts",
  },
  {
    src: "/screenshots/5.png",
    title: "Use the model",
    detail: "Try trained models on new images, then export the one that works.",
    alt: "AnyLearning model registry listing trained computer vision models",
  },
];

export function Slideshow({ className = "" }) {
  const [activeSlide, setActiveSlide] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    if (paused || prefersReducedMotion.matches) return undefined;

    const timer = window.setInterval(() => {
      setActiveSlide((current) => (current + 1) % SLIDES.length);
    }, 5000);

    return () => window.clearInterval(timer);
  }, [paused]);

  const showPrevious = () => {
    setActiveSlide((current) => (current - 1 + SLIDES.length) % SLIDES.length);
  };

  const showNext = () => {
    setActiveSlide((current) => (current + 1) % SLIDES.length);
  };

  const active = SLIDES[activeSlide];

  return (
    <section
      aria-label="AnyLearning product tour"
      aria-roledescription="carousel"
      className={`overflow-hidden rounded-xl border border-line bg-surface shadow-lg ${className}`}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocusCapture={() => setPaused(true)}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setPaused(false);
      }}
    >
      <div className="relative aspect-[8/5] overflow-hidden bg-surface-sunken">
        {SLIDES.map((slide, index) => (
          <Image
            key={slide.src}
            src={slide.src}
            alt={slide.alt}
            fill
            loading={index === 0 ? "eager" : "lazy"}
            sizes="(min-width: 1024px) 55vw, 100vw"
            aria-hidden={index !== activeSlide}
            className={`object-cover transition duration-500 ease-out motion-reduce:transition-none ${
              index === activeSlide ? "scale-100 opacity-100" : "scale-[1.01] opacity-0"
            }`}
          />
        ))}
        <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between bg-gradient-to-b from-black/50 to-transparent px-4 pt-3 pb-8 text-white">
          <span className="rounded-full bg-black/45 px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] backdrop-blur-sm">
            Product tour
          </span>
          <span className="font-mono text-[11px] tabular-nums">
            {String(activeSlide + 1).padStart(2, "0")} / {String(SLIDES.length).padStart(2, "0")}
          </span>
        </div>
      </div>

      <div className="flex min-h-[5.5rem] items-center gap-3 border-t border-line px-4 py-3 sm:px-5">
        <button
          type="button"
          aria-label="Show previous screenshot"
          onClick={showPrevious}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-line text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
        >
          <span aria-hidden>←</span>
        </button>

        <div className="min-w-0 flex-1" aria-live={paused ? "polite" : "off"} aria-atomic="true">
          <p className="truncate text-sm font-medium text-foreground">{active.title}</p>
          <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
            {active.detail}
          </p>
        </div>

        <div className="hidden items-center gap-1.5 sm:flex" aria-label="Choose a screenshot">
          {SLIDES.map((slide, index) => (
            <button
              key={slide.src}
              type="button"
              aria-label={`Show slide ${index + 1}: ${slide.title}`}
              aria-current={index === activeSlide ? "true" : undefined}
              onClick={() => setActiveSlide(index)}
              className={`h-1.5 rounded-full transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark focus-visible:ring-offset-2 ${
                index === activeSlide ? "w-5 bg-mark" : "w-1.5 bg-line hover:bg-muted-foreground"
              }`}
            />
          ))}
        </div>

        <button
          type="button"
          aria-label="Show next screenshot"
          onClick={showNext}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-line text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mark"
        >
          <span aria-hidden>→</span>
        </button>
      </div>
    </section>
  );
}
