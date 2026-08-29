import Document, { Head, Html, Main, NextScript } from "next/document";

import { fontClassNames } from "../components/fonts";

export default class MyDocument extends Document {
  render() {
    return (
      // The font variables go on <html>, not on a wrapper inside <body>.
      // Tailwind's preflight sets `font-family: var(--font-sans), …` on html,
      // and an empty var() invalidates the whole declaration rather than
      // falling through to the next entry — which silently rendered the entire
      // site in Times New Roman.
      //
      <Html lang="en" className={fontClassNames} style={{ colorScheme: "light" }}>
        <Head />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}
